# CAMhelper.py
import os, numpy as np
from typing import Optional
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.cm as cm

import helper  # uses helper._model, helper._device, helper.IM_SIZE, helper.MEAN, helper.STD

# Build the same normalization used by your classifier
_TFM = T.Compose([
    T.Resize((helper.IM_SIZE, helper.IM_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=helper.MEAN, std=helper.STD),
])

def _find_last_conv_layer(m: nn.Module) -> nn.Module:
    """Find the last Conv2d in model; works well for timm RexNet."""
    last = None
    for _, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d layer found for CAM.")
    return last

def _to_numpy_img(pil: Image.Image) -> np.ndarray:
    """Return float32 HxWx3 in [0,1] from PIL."""
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return arr

def _overlay_heatmap_on_img(base_rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """base_rgb01: HxWx3 in [0,1]; heat01: HxW in [0,1]. Returns HxWx3 uint8."""
    heat_rgba = cm.jet(heat01)                # HxWx4 in [0,1]
    heat_rgb  = heat_rgba[..., :3]
    out = (1 - alpha) * base_rgb01 + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

def save_rexnet_cam_overlay(image_path: str, out_path: str, alpha: float = 0.35, edema_class_idx: int = 1):
    """
    Save CAM overlay for ReXNet (class=edema_class_idx) onto original image.
    """
    if getattr(helper, "_model", None) is None:
        raise RuntimeError("ReXNet not initialized. Call init_rexnet(...) first.")

    model = helper._model.to(helper._device).eval()
    target_layer = _find_last_conv_layer(model)

    feats = []
    grads = []

    def fwd_hook(_, __, output):
        feats.append(output.detach())

    def bwd_hook(_, grad_in, grad_out):
        grads.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    pil_img = Image.open(image_path).convert("RGB")
    x = _TFM(pil_img).unsqueeze(0).to(helper._device)

    # forward + backward on desired class (edema=1)
    model.zero_grad(set_to_none=True)
    logits = model(x)                       # [1, 2]
    score = logits[0, edema_class_idx]
    score.backward()

    # collect tensors
    act = feats[-1]                         # [B, C, H, W]
    grad = grads[-1]                        # [B, C, H, W]
    h1.remove(); h2.remove()

    # Grad-CAM: weights = GAP over spatial dims on gradients
    weights = grad.mean(dim=(2, 3), keepdim=True)        # [B, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=False)      # [B, H, W]
    cam = torch.relu(cam)[0]                              # [H, W]

    # normalize to [0,1] and resize to original image size
    cam -= cam.min()
    if float(cam.max()) > 1e-8:
        cam /= cam.max()
    cam_np = cam.cpu().numpy()

    # upscale to original image size
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    cam01 = np.asarray(cam_pil, dtype=np.float32) / 255.0

    base01 = _to_numpy_img(pil_img)
    overlay = _overlay_heatmap_on_img(base01, cam01, alpha=alpha)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(overlay).save(out_path)
