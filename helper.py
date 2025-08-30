# rexnet_infer.py
import torch, timm
import numpy as np
from PIL import Image
from torchvision import transforms as T
import numpy as np
import joblib
import SimpleITK as sitk
from skimage.io import imread
from radiomics import featureextractor
import logging
from scipy.stats import chi2
from dataclasses import dataclass
from typing import Optional, Dict, Any,Callable, Tuple

# Get PyRadiomics logger
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)  # Show only errors, hide warnings/info

IM_SIZE = 1024
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_tfm = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model  = None

@dataclass
class Policy:
    mode: str = "balanced"        # "conservative" | "balanced" | "sensitive"
    thr_maha: float = 1.0         # your 95th percentile threshold (distance, not squared)
    p_accept: float = 0.60        # accept PE_yes if p>=p_accept, PE_no if p<=1-p_accept
    std_ok: float = 0.05          # ensemble/TTA agreement threshold (prob std)
    max_auto_accept_frd_multiple: float = 2.0  # never auto-accept if FRD > multiple*thr
    use_pvalue_gate: bool = True
    pvalue_min: float = 1e-3      # if χ² p-value < this → treat as “very OOD”

def _load_state_dict_safely(model, ckpt_path, strict=True):
    sd = torch.load(ckpt_path, map_location=_device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # strip 'module.' if saved from DataParallel
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.','',1): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=strict)
    return model

def init_rexnet(model_path="cv_models/rexnet_fold5_best.pth"):
    global _model
    _model = timm.create_model("rexnet_150", pretrained=False, num_classes=2)
    _model = _load_state_dict_safely(_model, model_path, strict=True)
    _model.eval().to(_device)

# optional: temperature scaling (set T=1.0 if you don’t have calibration yet)
_T = 1.0
def set_temperature(T=1.0):
    global _T
    _T = float(T)

@torch.no_grad()
def rexnet_predict_prob(image_path: str) -> float:
    """
    Returns calibrated probability for class-1 (Edema).
    Assumes you trained with 2 classes and softmax at inference.
    """
    assert _model is not None, "Call init_rexnet() once before predicting."
    img = Image.open(image_path).convert("RGB")
    x = _tfm(img).unsqueeze(0).to(_device)

    logits = _model(x)  # shape [1,2]
    logits = logits / _T
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)

# ================ FRD ================
# --- must match reference exactly ---
BIN_WIDTH = 10  # same as reference build
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(
    imageType={'Original': {}},
    setting={'binWidth': BIN_WIDTH, 'force2D': True}
)
EXTRACTOR.enableAllFeatures()  # only if you used this in reference!


def load_image_as_sitk(path):
    arr = imread(path)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32)
    return sitk.GetImageFromArray(arr)

def extract_radiomics_feature_vector(img_sitk):
    arr = sitk.GetArrayFromImage(img_sitk)
    H, W = arr.shape
    mask_arr = np.ones((H, W), dtype=np.uint8)
    mask_arr[0,:]=mask_arr[-1,:]=mask_arr[:,0]=mask_arr[:,-1]=0  # thin 0 border
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(img_sitk)

    res = EXTRACTOR.execute(img_sitk, mask, label=1)
    keys = sorted(k for k in res.keys() if k.startswith('original_'))
    vec = np.array([float(res[k]) for k in keys], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec

# ---- load reference once (do this at module import or in init) ----
_ref = joblib.load("radiomics_ref_pack.joblib")
FEATURE_IDX = np.array(_ref["feature_keys"], dtype=int)
SCALER = _ref["scaler"]
EMP = _ref["emp"]
THR_MAHA = _ref.get("thr_maha", None)  
# print("THR_MAHA = ", THR_MAHA)

def single_image_score(image_path: str):
    """
    Returns Mahalanobis distance in standardized radiomics space.
    Requires that EMP was fit on SCALER-transformed reference features.
    """
    img = load_image_as_sitk(image_path)
    v_all = extract_radiomics_feature_vector(img)    # full sorted 'original_*'
    v_ref = v_all[FEATURE_IDX]                       # select same columns as reference
    v_std = SCALER.transform(v_ref.reshape(1, -1))   # standardize
    # sklearn's EmpiricalCovariance.mahalanobis returns squared distance(s)
    maha_sq = float(EMP.mahalanobis(v_std)[0])
    maha = float(np.sqrt(maha_sq))
    out = {"Mahalanobis": maha}
    if THR_MAHA is not None:
        out["thr_maha"] = float(THR_MAHA)
        out["in_distribution"] = bool(maha <= THR_MAHA)
    return out


# ================ ROUTER ================
def maha_pvalue(maha: float, df: int) -> float:
    """Two-sided tail isn’t needed; χ² is one-sided on distance^2."""
    return float(1.0 - chi2.cdf(maha*maha, df=df))

def frd_bin(maha: float, thr: float) -> str:
    """Just for logging/labels."""
    if maha <= thr: return "low"
    if maha <= 1.5 * thr: return "mid"
    return "high"

def route_case(
    frd_maha: float,                   # Mahalanobis distance of the case
    thr_maha: float,                   # FRD threshold (e.g., 95th percentile of ref)
    p: float,                          # ReXNet PE probability (calibrated)
    p_accept: float = 0.50,            # accept PE_yes if p>=p_accept; PE_no if p<=1-p_accept
    mode: str = "balanced",
    ensemble_std: Optional[float] = None,  # TTA/ensemble std of prob (None if not used)
    std_ok: float = 0.05,
    df_features: Optional[int] = None, # len(FEATURE_IDX) if you want p-values
    use_pvalue_gate: bool = True,
    pvalue_min: float = 1e-3,
    max_auto_accept_frd_multiple: float = 2.0,
    may_request_tta: bool = True,
) -> Dict[str, Any]:
    """
    Deterministic router.
    Returns: {action, reason, tags, audit}
      action ∈ {"ACCEPT","RUN_MOE","ESCALATE_VLM","ESCALATE_HUMAN"}
    """
    # ---- derived flags ----
    # p_reject = 1.0 - p_accept
    # high_conf = (p >= p_accept) or (p <= p_reject)
    in_domain = frd_maha <= thr_maha
    agree = (ensemble_std is not None) and (ensemble_std <= std_ok)
    ood_mult = frd_maha / max(thr_maha, 1e-8)
    ood_bin = frd_bin(frd_maha, thr_maha)
    # Small pv == large OOD
    pv = maha_pvalue(frd_maha, df_features) if (use_pvalue_gate and df_features is not None) else None
    very_ood = (pv is not None and pv < pvalue_min) or (ood_mult > max_auto_accept_frd_multiple)

    # ---- mode tweaks ----
    if mode == "conservative":
        p_accept_adj = min(0.98, p_accept + 0.2)
        std_ok_adj = max(0.03, std_ok - 0.01)
    elif mode == "sensitive":
        p_accept_adj = max(0.5, p_accept + 0.05)
        std_ok_adj = std_ok + 0.02
    else:
        p_accept_adj = p_accept+0.1
        std_ok_adj = std_ok
    
    high_conf_adj = (p >= p_accept_adj) or (p <= (1.0 - p_accept_adj))
    agree_adj = (ensemble_std is not None) and (ensemble_std <= std_ok_adj)

    audit = {
        "frd_maha": frd_maha,
        "thr_maha": thr_maha,
        "ood_multiple": ood_mult,
        "ood_bin": ood_bin,
        "p": p,
        "ensemble_std": ensemble_std,
        "mode": mode,
        "p_accept_adj": p_accept_adj,
        "std_ok_adj": std_ok_adj,
        "pvalue": pv,
        "very_ood": very_ood,
    }

    # ---- Case 1: in-domain + high confidence ----
    if in_domain and high_conf_adj:
        return {
            "action": "ACCEPT",
            "reason": "High confidence with in-domain data",
            "tags": ["accept","in-domain", ood_bin],
            "audit": audit,
        }

    # ---- Case 2: OOD look + high confidence ----
    if (not in_domain) and high_conf_adj:
        if (ensemble_std is None) and may_request_tta:
            return {
                "action": "TTA",
                "reason": "High confidence while out of domain data",
                "tags": ["ood","high-conf", ood_bin],
                "audit": audit,
            }
        # after TTA (or if not allowed), decide:
        if agree_adj:
            return {
                "action": "ACCEPT",
                "reason": "TTA test past",
                "tags": ["accept","ood","rescued", ood_bin],
                "audit": audit,
            }
        return {
            "action": "RUN_MOE",
            "reason": "TTA Test failed, suggesting to try MoE",
            "tags": ["ood","high-conf", ood_bin],
            "audit": audit,
        }


    # ---- Case 3: OOD + low confidence ----
    if (not in_domain) and (not high_conf_adj):
        return {
            "action": "ESCALATE_VLM",
            "reason": "Out of domain case + Low confidence, I cant Help",
            "tags": ["ood","low-conf", ood_bin],
            "audit": audit,
        }

    # ---- Case 4: in-domain + low confidence ----
    if in_domain and (not high_conf_adj):
        if (ensemble_std is None) and may_request_tta:
            return {
                "action": "TTA",
                "reason": "Low confidence while in domain, suggesting TTA test",
                "tags": ["in-domain","low-conf", ood_bin],
                "audit": audit,
            }
        # after TTA (or if not allowed), decide:
        if agree_adj:
            return {
                "action": "ACCEPT",
                "reason": "TTA test past",
                "tags": ["accept","in-domain","rescued", ood_bin],
                "audit": audit,
            }
        return {
            "action": "ESCALATE_VLM",
            "reason": "Soryy I cant help this case as it's beyond my current knowledge",
            "tags": ["in-domain","low-conf", ood_bin],
            "audit": audit,
        }

    # ---- guardrails: never auto-accept extreme OOD ----
    # if very_ood:
    #     # still allow MoE to try; else human
    #     return {
    #         "action": "RUN_MOE",
    #         "reason": "guardrail_very_ood_run_moe",
    #         "tags": ["guardrail","very-ood", ood_bin],
    #         "audit": audit,
    #     }

    # # ---- Case 1: in-domain + high confidence ----
    # if in_domain and high_conf_adj:
    #     return {
    #         "action": "ACCEPT",
    #         "reason": "case1_highconf_lowfrd",
    #         "tags": ["accept","in-domain", ood_bin],
    #         "audit": audit,
    #     }

    # # ---- Case 2: OOD look + high confidence ----
    # if (not in_domain) and high_conf_adj:
    #     if agree_adj:
    #         return {
    #             "action": "ACCEPT",
    #             "reason": "case2_rescued_agree",
    #             "tags": ["accept","ood","rescued", ood_bin],
    #             "audit": audit,
    #         }
    #     return {
    #         "action": "RUN_MOE",
    #         "reason": "case2_highconf_highfrd",
    #         "tags": ["ood","high-conf", ood_bin],
    #         "audit": audit,
    #     }

    # # ---- Case 3: OOD + low confidence ----
    # if (not in_domain) and (not high_conf_adj):
    #     return {
    #         "action": "ESCALATE_VLM",
    #         "reason": "case3_lowconf_highfrd",
    #         "tags": ["ood","low-conf", ood_bin],
    #         "audit": audit,
    #     }

    # # ---- Case 4: in-domain + low confidence ----
    # if in_domain and (not high_conf_adj):
    #     if agree_adj:
    #         return {
    #             "action": "ACCEPT",
    #             "reason": "case4_rescued_agree",
    #             "tags": ["accept","in-domain","rescued", ood_bin],
    #             "audit": audit,
    #         }
    #     return {
    #         "action": "ESCALATE_VLM",
    #         "reason": "case4_lowconf_lowfrd",
    #         "tags": ["in-domain","low-conf", ood_bin],
    #         "audit": audit,
    #     }