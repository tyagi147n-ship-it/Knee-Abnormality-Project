"""Local browser demo for the knee abnormality capstone project.

Run with:
    streamlit run app.py

This app accepts PNG/JPG/JPEG and common DICOM files. The supplied project
does not include a trained image classifier, so the result is an image-quality
and signal-analysis prototype, not a medical diagnosis.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Knee MRI Analysis", page_icon="🦵", layout="wide")


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Robustly normalize a medical image to displayable 8-bit pixels."""
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim > 2:
        # For a volume/multi-frame file, display the middle slice/frame.
        array = array[array.shape[0] // 2]
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - low) / (high - low)
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def read_uploaded_file(filename: str, content: bytes):
    """Return a display image and file metadata for an image or DICOM."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in {"dcm", "dicom"}:
        import pydicom

        dataset = pydicom.dcmread(BytesIO(content), force=True)
        if not hasattr(dataset, "PixelData"):
            raise ValueError("This DICOM file has no pixel data.")
        pixels = normalize_to_uint8(dataset.pixel_array)
        image = Image.fromarray(pixels).convert("L")
        metadata = {
            "Format": "DICOM",
            "Rows": getattr(dataset, "Rows", image.height),
            "Columns": getattr(dataset, "Columns", image.width),
            "Modality": str(getattr(dataset, "Modality", "Unknown")),
            "Photometric interpretation": str(
                getattr(dataset, "PhotometricInterpretation", "Unknown")
            ),
        }
    else:
        image = Image.open(BytesIO(content)).convert("L")
        metadata = {
            "Format": image.format or suffix.upper() or "Image",
            "Rows": image.height,
            "Columns": image.width,
            "Modality": "Not available",
            "Photometric interpretation": "Grayscale conversion",
        }
    return image, metadata


def analyze_image(image: Image.Image) -> dict:
    """Compute transparent, reproducible prototype image metrics."""
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = float(pixels.mean())
    std = float(pixels.std())
    p01, p99 = [float(x) for x in np.percentile(pixels, [1, 99])]
    contrast = p99 - p01
    dark_fraction = float((pixels < 0.03).mean())
    bright_fraction = float((pixels > 0.97).mean())

    warnings = []
    if contrast < 0.15:
        warnings.append("Very low contrast; image may be difficult to interpret.")
    if dark_fraction > 0.98 or bright_fraction > 0.98:
        warnings.append("Image is almost entirely blank or saturated.")
    if min(image.size) < 128:
        warnings.append("Image resolution is low for an MRI-style image.")

    quality = "Good for prototype display" if not warnings else "Needs review"
    return {
        "quality": quality,
        "mean_intensity": mean,
        "contrast": contrast,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "warnings": warnings,
    }


st.title("🦵 Knee MRI Abnormality Analysis")
st.caption("Local capstone demonstration • Upload one image or DICOM file")

with st.sidebar:
    st.header("About this demo")
    st.write(
        "This interface analyzes the uploaded pixels and reports image-quality "
        "signals. It is not a clinical diagnosis."
    )
    st.info(
        "The current dataset does not include paired DICOM images and labels, "
        "so a validated image abnormality classifier is not yet available."
    )

uploaded = st.file_uploader(
    "Upload one knee image or DICOM file",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    help="Supported: PNG, JPG, JPEG, and DICOM (.dcm/.dicom).",
)

if uploaded is None:
    st.markdown(
        "### Start here\n"
        "Choose an MRI screenshot/image or a DICOM file above. "
        "The analysis will appear automatically."
    )
else:
    try:
        image, metadata = read_uploaded_file(uploaded.name, uploaded.getvalue())
        result = analyze_image(image)

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True, caption=uploaded.name)

        with right:
            st.subheader("Analysis result")
            if result["quality"] == "Good for prototype display":
                st.success(result["quality"])
            else:
                st.warning(result["quality"])

            st.metric("Image dimensions", f"{image.width} × {image.height} px")
            st.metric("Contrast range", f"{result['contrast']:.1%}")
            st.metric("Mean intensity", f"{result['mean_intensity']:.1%}")

        st.divider()
        st.subheader("File information")
        st.json(metadata)

        if result["warnings"]:
            st.subheader("Quality warnings")
            for warning in result["warnings"]:
                st.write(f"⚠️ {warning}")
        else:
            st.write("✅ No basic image-quality warnings detected.")

        st.subheader("Capstone model status")
        st.warning(
            "This upload is being analyzed by the prototype image-processing "
            "pipeline. It does not yet predict ACL tear, meniscus tear, OA, "
            "or other abnormalities because the project has no trained image "
            "weights. Do not use this output for medical decisions."
        )

    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.info("Try a standard PNG/JPG image or an uncompressed DICOM file.")
