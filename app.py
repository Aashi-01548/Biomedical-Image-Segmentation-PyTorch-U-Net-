from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from predict import (
    load_cell_model, load_brain_model,
    cell_background, predict_cells,
    predict_brain, make_overlay, dice_score,
)

st.set_page_config(page_title="U-Net Segmentation", layout="wide")


@st.cache_resource          # load each model once, not on every click
def get_cell_model():
    return load_cell_model()


@st.cache_resource
def get_brain_model():
    return load_brain_model()


def pick_image(folder, key):
    """Choose a sample or upload an image. Returns (grayscale image, truth mask or None)."""
    choice = st.radio("Image source", ["Sample image", "Upload your own"],
                      horizontal=True, key=f"{key}_src")
    if choice == "Upload your own":
        f = st.file_uploader("PNG, JPG or TIF", type=["png", "jpg", "jpeg", "tif", "tiff"],
                             key=f"{key}_up")
        if f is None:
            return None, None
        return np.array(Image.open(f).convert("L")), None

    samples = sorted(p for p in Path(folder).glob("*.png") if not p.stem.endswith("_mask"))
    if not samples:
        st.warning(f"No samples found in '{folder}'. Run the sample-saving cell in the notebook first.")
        return None, None
    name = st.selectbox("Sample", [p.name for p in samples], key=f"{key}_sel")
    path = Path(folder) / name
    gray = np.array(Image.open(path).convert("L"))
    mask_path = path.with_name(path.stem + "_mask.png")
    truth = (np.array(Image.open(mask_path).convert("L")) > 127) if mask_path.exists() else None
    return gray, truth


def show_results(background, prob, truth, label, threshold):
    mask = prob > threshold
    overlay = make_overlay(background, mask)

    cols = st.columns(4 if truth is not None else 3)
    cols[0].image(background, caption="Input", clamp=True)
    cols[1].image((mask * 255).astype(np.uint8), caption=f"Predicted {label}")
    cols[2].image(overlay, caption="Overlay on input")
    if truth is not None:
        cols[3].image((truth * 255).astype(np.uint8), caption="Ground truth")

    m1, m2 = st.columns(2)
    m1.metric(f"Predicted {label} area", f"{mask.mean() * 100:.1f}% of pixels")
    if truth is not None:
        m2.metric("Dice score (1.0 = perfect overlap)", f"{dice_score(mask, truth):.3f}")


st.title("U-Net image segmentation")
st.caption("A learning project. Not a medical device and not for diagnosis.")

tab_cells, tab_brain, tab_how, tab_about = st.tabs(
    ["Cells (Part 1)", "Brain MRI (Part 2)", "How it works", "About"]
)

with tab_cells:
    st.subheader("Neuron segmentation in electron microscope images")
    gray, truth = pick_image("samples/cells", "cell")
    threshold = st.slider("Threshold: how confident the model must be to mark a pixel",
                          0.05, 0.95, 0.5, 0.05, key="cell_thr")
    if gray is not None:
        with st.spinner("Running the model (a few seconds on CPU)..."):
            prob = predict_cells(get_cell_model(), gray)
        show_results(cell_background(gray), prob, truth, "neurons", threshold)
        st.caption("Sample images come from the training set, so the Dice score is optimistic.")

with tab_brain:
    st.subheader("Tumour segmentation in a brain MRI slice (T2)")
    gray, truth = pick_image("samples/brain", "brain")
    threshold = st.slider("Threshold: how confident the model must be to mark a pixel",
                          0.05, 0.95, 0.5, 0.05, key="brain_thr")
    if gray is not None:
        with st.spinner("Running the model..."):
            background, prob = predict_brain(get_brain_model(), gray)
        show_results(background, prob, truth, "tumour", threshold)
        st.caption("Uploads should be a single T2 MRI slice. Results on other image types are not meaningful.")

with tab_how:
    st.subheader("How the U-Net works")
    st.markdown("""
- **Down path:** convolutions and pooling shrink the image while learning *what* is in it.
- **Up path:** transposed convolutions grow it back, working out *where* things are.
- **Skip connections:** copy fine detail from the down path to the up path.
- **Output:** a probability for every pixel. The threshold slider turns it into a yes/no mask.

**Part 1** uses no padding: a 572×572 input gives a 388×388 output.
**Part 2** uses padding and batch normalisation, so the output size matches the input (240×240).
""")
    # Later: st.image("assets/unet_diagram.png", caption="U-Net architecture")

with tab_about:
    st.subheader("About this project")
    st.markdown("""
- **Task:** semantic segmentation, which labels every pixel of an image.
- **Data:** a small electron-microscopy set (Part 1) and brain MRI scans from BraTS 2020 (Part 2).
- **Models:** pre-trained weights from an IBM Skills Network lab, rebuilt in PyTorch.
- **Limits:** small data, no clinical validation, a single dataset. Do not use it to make medical decisions.
""")
    