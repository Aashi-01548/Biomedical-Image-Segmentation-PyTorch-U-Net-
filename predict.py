import numpy as np
import cv2
import torch

from model import UNet, UNetBrain


def load_cell_model(path="weights/unet_cells.pt"):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_brain_model(path="weights/unet_brain.pt"):
    model = UNetBrain(num_classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ---------- Part 1: cells ----------
def cell_background(gray):
    """The 388x388 centre of the 512x512 image: the area the model predicts."""
    img = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    return img[62:450, 62:450]


def predict_cells(model, gray):
    img = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_REFLECT_101)   # 572 x 572
    x = torch.from_numpy(img)[None, None]
    with torch.no_grad():
        prob = model(x)[0, 0].numpy()                                        # 388 x 388
    return prob


# ---------- Part 2: brain ----------
def predict_brain(model, gray):
    img = cv2.resize(gray, (240, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = img / max(float(img.max()), 1e-8)
    x = torch.from_numpy(img)[None, None]
    with torch.inference_mode():
        prob = torch.sigmoid(model(x)[0, 0]).numpy()    # raw scores -> probabilities
    return (img * 255).astype(np.uint8), prob


# ---------- Shared helpers ----------
def make_overlay(background, mask, color=(255, 0, 0), alpha=0.5):
    rgb = np.stack([background.astype(np.float32)] * 3, axis=-1)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
    return rgb.astype(np.uint8)


def dice_score(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = pred.sum() + truth.sum()
    return 1.0 if total == 0 else 2.0 * (pred & truth).sum() / total
