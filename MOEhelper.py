# MOEhelper.py
import torch, timm
from PIL import Image
from torchvision import transforms as T
from typing import Dict, Any, Tuple
import helper  # uses helper._device, helper.IM_SIZE, helper.MEAN, helper.STD, rexnet_predict_prob
from typing import Callable, Tuple

_device = helper._device if hasattr(helper, "_device") else ("cuda" if torch.cuda.is_available() else "cpu")
_TFM = T.Compose([
    T.Resize((helper.IM_SIZE, helper.IM_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=helper.MEAN, std=helper.STD),
])

# 3 committee members besides ReXNet
_COMMITTEE_SPECS: Tuple[Tuple[str, str], ...] = (
    ("resnet50d",           "cv_models/resnet50d_fold5_best.pth"),
    ("regnety_016",         "cv_models/regnety_016_fold5_best.pth"),
    ("tf_efficientnetv2_b3","cv_models/tf_efficientnetv2_b3_fold5_best.pth"),
)

_cache: Dict[Tuple[str, str], torch.nn.Module] = {}

def _load_state_dict_safely(model, ckpt_path, strict=True):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=strict)
    return model

def _get_model(arch: str, ckpt: str) -> torch.nn.Module:
    key = (arch, ckpt)
    if key not in _cache:
        m = timm.create_model(arch, pretrained=False, num_classes=2)
        m = _load_state_dict_safely(m, ckpt, strict=True)
        m = m.to(_device).eval()
        _cache[key] = m
    return _cache[key]

@torch.no_grad()
def _prob_from_model(m: torch.nn.Module, img_pil: Image.Image) -> float:
    x = _TFM(img_pil).unsqueeze(0).to(_device)
    logits = m(x)                       # [1,2]
    return float(torch.softmax(logits, dim=1)[0, 1].item())

@torch.no_grad()
def run_committee(image_path: str, vote_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Returns a dict with per-model probs, hard votes, and majority result.
    Majority rule: 3/4 agreement (either PE_yes or PE_no) → passed=True.
    vote_threshold: prob cutoff for a 'PE_yes' vote (default 0.5).
    """
    # Ensure ReXNet ready (use your existing init if needed)
    pass_num=3
    if getattr(helper, "_model", None) is None:
        helper.init_rexnet("cv_models/rexnet_fold5_best.pth")

    img = Image.open(image_path).convert("RGB")

    # ReXNet via helper
    p_rex = helper.rexnet_predict_prob(image_path)
    probs = {"rexnet_150": p_rex}
    votes = {"rexnet_150": 1 if p_rex >= vote_threshold else 0}

    # Other three models
    for arch, ckpt in _COMMITTEE_SPECS:
        m = _get_model(arch, ckpt)
        p = _prob_from_model(m, img)
        # print(p)
        probs[arch] = p
        votes[arch] = 1 if p >= vote_threshold else 0

    n_yes = sum(votes.values())
    n_models = len(votes)
    n_no = n_models - n_yes

    passed = (n_yes >= pass_num) or (n_no >= pass_num)     # 3-of-4 either way
    label = "PE_yes" if n_yes >= pass_num else ("PE_no" if n_no >= pass_num else "undecided")

    return {
        "probs": probs,              # per-model probabilities for class-1 (Edema)
        "votes": votes,              # 1=yes, 0=no
        "n_yes": n_yes,
        "n_no": n_no,
        "n_models": n_models,
        "passed": passed,            # True if >=3 agree
        "label": label,              # majority label or 'undecided'
        "vote_threshold": vote_threshold,
    }


def moe_decisive(moe: Dict[str, Any],
                 min_support: int | None = None,
                 min_total: int | None = None,
                 min_conf: float | None = None) -> bool:
    """
    Decide if a committee result is *final* (no need to escalate).
    Works even if run_committee() doesn't return 'ok'.
    Heuristics (in order):
      1) If 'ok' is present -> use it.
      2) If 'support'/'total' present -> apply threshold if given.
      3) If 'confidence' present -> apply threshold if given.
      4) Else -> treat label in {'PE_yes','PE_no'} as decisive.
    """
    if "ok" in moe:
        return bool(moe["ok"])

    # vote-based check
    if min_support is not None and "support" in moe:
        total = moe.get("total", min_total or 0)
        if total:  # avoid divide-by-zero; only check if we have a total
            if moe["support"] < min_support:
                return False

    # confidence check
    if min_conf is not None and ("confidence" in moe):
        if moe["confidence"] < float(min_conf):
            return False

    # plain label check
    return moe.get("label") in ("PE_yes", "PE_no")


def cascade_moe_vlm(
    image_path: str,
    run_committee_fn: Callable[[str], Dict[str, Any]],
    vlm_fn: Callable[..., Dict[str, Any]],
    *,
    vlm_model: str = "gpt-4.1-mini",
    vote_threshold_support: int | None = None,  # e.g., 3 (of 4)
    vote_threshold_total: int | None = None,    # e.g., 4
    confidence_threshold: float | None = None,  # e.g., 0.65
    do_prints: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """
    Run MoE committee; if not decisive -> escalate to VLM.
    Returns (result, stage), where stage is "moe" or "vlm".
    The 'result' always contains at least {'label': ...}.
    """
    moe = run_committee_fn(image_path)
    if do_prints:
        print("Committee decision:", moe.get("label", "undetermined"), flush=True)

    decisive = moe_decisive(
        moe,
        min_support=vote_threshold_support,
        min_total=vote_threshold_total,
        min_conf=confidence_threshold,
    )

    if decisive:
        return {"label": moe.get("label"), "moe": moe}, "moe"

    # escalate to VLM
    vlm = vlm_fn(image_path, model=vlm_model)
    if vlm.get("ok"):
        pe_yes = (vlm["label"] == 1)
        result = {
            "label": ("PE_yes" if pe_yes else "PE_no"),
            "pe_prob": vlm.get("confidence"),
            "explanation": vlm.get("explanation"),
            "moe": moe,
            "vlm": vlm,
            "fallback": "vlm",
        }
        if do_prints:
            print("\nVLM's suggestion:", result["label"], flush=True)
            if result.get("explanation"):
                print("Explanation:", result["explanation"], flush=True)
        return result, "vlm"

    # VLM also inconclusive
    return {"label": "undetermined", "moe": moe, "vlm": vlm, "fallback": "vlm_failed"}, "vlm"