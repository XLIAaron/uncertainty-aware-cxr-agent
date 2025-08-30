# file: watch_and_run.py
import time, os, json, hashlib, pathlib
from datetime import datetime

# --- your pipeline imports ---
from helper import init_rexnet, rexnet_predict_prob, single_image_score, route_case, THR_MAHA, FEATURE_IDX
from TTAhelper import get_tta_stats
from MOEhelper import run_committee, cascade_moe_vlm
from VLMhelper import vlm_diag_line
from CAMhelper import save_rexnet_cam_overlay
from pathlib import Path
import shutil

USE_SENTINEL = "done"        # "done" | "sha256" | None
IN_DIR = "incoming"
RUNS_DIR = "runs"
STATE_FP = os.path.join(RUNS_DIR, "seen.json")
LOG_FP   = os.path.join(RUNS_DIR, "log.jsonl")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STAGING = os.path.join(IN_DIR, "_staging")
CAM_DIR = os.path.join(IN_DIR, "_cam")          
LABEL_DIRS = {"Edema", "NoEdema"}               
EXCLUDE_DIRS = {os.path.basename(STAGING), os.path.basename(CAM_DIR), *LABEL_DIRS} 

os.makedirs(STAGING, exist_ok=True)
os.makedirs(CAM_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
for d in LABEL_DIRS: os.makedirs(os.path.join(IN_DIR, d), exist_ok=True)

policy = {
    "mode": "balanced",
    "p_accept": 0.50,
    "std_ok": 0.05,
    "max_auto_accept_frd_multiple": 2.0,
    "use_pvalue_gate": True,
    "pvalue_min": 1e-3,
}

os.makedirs(RUNS_DIR, exist_ok=True)
seen = set()
if os.path.exists(STATE_FP):
    seen = set(json.load(open(STATE_FP)))

def bucket_output(src_fp: str, label: str, diagnosed_by: str, in_root: str):
    """
    label: 'PE_yes' or 'PE_no'
    diagnosed_by: 'vlm' if VLM produced the final label, else 'model'/'moe'
    in_root: base incoming folder (e.g., IN_DIR)
    """
    if label not in ("PE_yes", "PE_no"):
        return
    label_dir = "Edema" if label == "PE_yes" else "NoEdema"
    subroot = Path(in_root) / ("recheck_needed" if diagnosed_by == "vlm" else "")
    dest_dir = subroot / label_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / Path(src_fp).name
    try:
        shutil.copy2(src_fp, dst)   # use shutil.move(...) if you prefer moving
        print(f"Saved to: {dst}")
    except Exception as e:
        print(f"[WARN] Could not copy to {dst}: {e}")

def _unique_dest(dest_dir: str, base_name: str) -> str:
    """Avoid collisions when moving files."""
    stem, ext = os.path.splitext(base_name)
    dest = os.path.join(dest_dir, base_name)
    if not os.path.exists(dest):
        return dest
    i = 1
    while True:
        cand = os.path.join(dest_dir, f"{stem}_{int(time.time())}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

def move_to_label(fp: str, label: str, diagnosed_by: str = "model") -> str:
    """
    label: 'PE_yes' -> Edema, 'PE_no' -> NoEdema.
    diagnosed_by: 'vlm' -> place under incoming/recheck_needed/...
    Returns new path (or original if label unknown).
    """
    if label not in {"PE_yes", "PE_no"}:
        return fp
    sub = "Edema" if label == "PE_yes" else "NoEdema"
    if diagnosed_by == "vlm":
        dest_dir = os.path.join(IN_DIR, "recheck_needed", sub)
    else:
        dest_dir = os.path.join(IN_DIR, sub)
    os.makedirs(dest_dir, exist_ok=True)
    dest_fp = _unique_dest(dest_dir, os.path.basename(fp))
    os.replace(fp, dest_fp)   # atomic within same filesystem
    return dest_fp

def promote_stable(stable_for=5.0):
    now = time.time()
    for n in os.listdir(STAGING):
        sp = os.path.join(STAGING, n)
        try:
            st = os.stat(sp)
            if now - st.st_mtime >= stable_for and file_ready(sp):
                os.replace(sp, os.path.join(IN_DIR, n))  # atomic
        except FileNotFoundError:
            pass

def file_ready(fp, wait_s=0.7):
    """Simple debounce: size must stop changing."""
    try:
        s1 = os.path.getsize(fp)
        time.sleep(wait_s)
        s2 = os.path.getsize(fp)
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False

def fhash(fp):
    h = hashlib.sha1()
    h.update(fp.encode())
    # (optional) also hash mtime+size to reprocess if file changed
    st = os.stat(fp)
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def process_one(fp):
    # 0) Save CAM overlay right away (based on initial ReXNet)
    try:
        cam_out = os.path.join(CAM_DIR, os.path.basename(fp))  # same filename
        save_rexnet_cam_overlay(fp, cam_out, alpha=0.35)
    except Exception as e:
        print("CAM overlay failed:", repr(e))

    # 1) perceive (single model)
    p = rexnet_predict_prob(fp)
    frd = single_image_score(fp)
    d = frd["Mahalanobis"]
    print(f"Score: {p:.4f}")
    out = route_case(
        frd_maha=d, thr_maha=THR_MAHA,
        p=p,
        p_accept=policy["p_accept"], mode=policy["mode"],
        ensemble_std=None, std_ok=policy["std_ok"],
        df_features=len(FEATURE_IDX),
        use_pvalue_gate=policy["use_pvalue_gate"],
        pvalue_min=policy["pvalue_min"],
        max_auto_accept_frd_multiple=policy["max_auto_accept_frd_multiple"],
        may_request_tta=True,
    )
    print("Single Model Suggesion:", out["action"], "-", out["reason"], flush=True)

    stage = "base"
    p_used, std_used = p, None

    if out["action"] == "TTA":
        tta = get_tta_stats(fp, n=8, allow_hflip=False)
        p_used, std_used = tta["p_mean"], tta["p_std"]
        stage = "tta"
        out = route_case(
            frd_maha=d, thr_maha=THR_MAHA,
            p=p_used,
            p_accept=policy["p_accept"], mode=policy["mode"],
            ensemble_std=std_used, std_ok=policy["std_ok"],
            df_features=len(FEATURE_IDX),
            use_pvalue_gate=policy["use_pvalue_gate"],
            pvalue_min=policy["pvalue_min"],
            max_auto_accept_frd_multiple=policy["max_auto_accept_frd_multiple"],
            may_request_tta=False,
        )
        print("after TTA:", out["action"], "-", out["reason"], flush=True)

    result = None
    if out["action"] == "ACCEPT":
        result = {"pe_prob": p_used, "label": ("PE_yes" if p_used >= policy["p_accept"] else "PE_no")}
    elif out["action"] == "RUN_MOE":
        result, stage = cascade_moe_vlm(fp, run_committee_fn=run_committee, vlm_fn=vlm_diag_line, do_prints=True)
    elif out["action"] == "ESCALATE_VLM":
        result, stage = cascade_moe_vlm(fp, run_committee_fn=lambda _: {"label": "undetermined", "ok": False},
                                        vlm_fn=vlm_diag_line, do_prints=True)

    # if result and result.get("label") in ("PE_yes", "PE_no"):
    #         diagnosed_by = "vlm" if stage == "vlm" else "model"   # 'moe','base','tta' → normal bucket
    #         bucket_output(fp, result["label"], diagnosed_by, IN_DIR)

    # 4) log
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "file": fp,
        "stage": stage,
        "p": p_used,
        "p_std": std_used,
        "frd_maha": d,
        "thr_maha": THR_MAHA,
        "decision": out,
        "result": result,
    }
    with open(LOG_FP, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    init_rexnet("cv_models/rexnet_fold5_best.pth")
    print(f"Watching {IN_DIR} …")
    while True:
        promote_stable(stable_for=1.0)
        for root, dirs, files in os.walk(IN_DIR, topdown=True):
            # EXCLUDE destination and helper dirs so moved files aren't reprocessed
            # dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            # dirs[:] = [d for d in dirs if d not in {"_staging", "recheck_needed", "Edema", "NoEdema"}]
            dirs[:] = [d for d in dirs if d not in {"_staging", "recheck_needed", "Edema", "NoEdema","_cam"}]
            for name in files:
                fp = os.path.join(root, name)
                if pathlib.Path(fp).suffix.lower() not in EXTS:
                    continue
                key = fhash(fp)
                if key in seen or not file_ready(fp):
                    continue
                try:
                    rec = process_one(fp)
                    # final label for move
                    final_lbl = (rec.get("result") or {}).get("label")
                    if final_lbl in {"PE_yes", "PE_no"}:
                        diagnosed_by = "vlm" if rec.get("stage") == "vlm" else "model"
                        new_fp = move_to_label(fp, final_lbl, diagnosed_by=diagnosed_by)
                        print(f"Moved -> {final_lbl} : {new_fp}", flush=True)
                    else:
                        print(f"Final: {final_lbl or rec['decision']['action']}  (file: {fp})", flush=True)

                    seen.add(key)
                    json.dump(list(seen), open(STATE_FP, "w"))
                except Exception as e:
                    print("ERROR processing", fp, ":", repr(e))
        time.sleep(1.0)

if __name__ == "__main__":
    main()