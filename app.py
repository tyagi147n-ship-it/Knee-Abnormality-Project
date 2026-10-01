"""Local browser demo for the knee abnormality capstone project.

Run with:
    streamlit run app.py

WHAT THIS APP SHOWS, HONESTLY
------------------------------
Your project actually has two separate pieces:

1. An image-quality checker for an uploaded PNG/JPG/DICOM (this was already
   in app.py). It reports pixel stats (contrast, brightness, etc.) — it does
   NOT diagnose anything, because the project has no trained image classifier.

2. A rule-based NLP pipeline, built in knee_abnormality_project.ipynb, that
   reads the TEXT of a radiology report ("FINDINGS: ... IMPRESSION: ...") and
   labels each finding (Medial OA, Medial/Lateral Meniscus, Effusion,
   Synovitis, Baker's cyst, MCL, Lateral OA, PF OA, Contusion, Fracture) as
   POSITIVE / NEGATIVE / UNCERTAIN / NOT_MENTIONED, with the matched sentence
   and pattern shown — this is exactly what produced the terminal output you
   were looking at (precision/recall/F1, PASS/FAIL lines, etc.).

These two pieces are independent: the classifier reads TEXT, not image
pixels. So to see the same kind of structured result you saw in your
terminal, paste the report text for a study into the box below (you can
still upload the matching image above it just to display it side by side).
This app does not, and cannot, generate diagnostic labels from the image
pixels alone — being upfront about that with your evaluator is safer than
implying otherwise.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from nlp_findings import analyze_report, UNAVAILABLE_TARGETS

st.set_page_config(page_title="Knee MRI Analysis", page_icon="🦵", layout="wide")


# ============================================================
# Image-quality section (unchanged behaviour from the original app)
# ============================================================

def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Robustly normalize a medical image to displayable 8-bit pixels."""
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim > 2:
        array = array[array.shape[0] // 2]
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - low) / (high - low)
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def read_uploaded_file(filename: str, content: bytes):
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


# ============================================================
# NLP findings display helpers
# ============================================================

STATUS_STYLE = {
    "POSITIVE": ("🔴", "#5c1a1a"),
    "NEGATIVE": ("🟢", "#173d17"),
    "UNCERTAIN": ("🟡", "#4d3c05"),
    "NOT_MENTIONED": ("⚪", "#2b2b2b"),
    "NO_DETECTOR": ("⚫", "#2b2b2b"),
}


def render_findings(results: dict) -> None:
    order = [
        "Medial OA", "Lateral OA", "PF OA",
        "Medial Meniscus", "Lateral Meniscus",
        "MCL", "ACL",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]
    for finding in order:
        r = results.get(finding)
        if r is None:
            continue
        icon, color = STATUS_STYLE.get(r["status"], ("⚪", "#2b2b2b"))
        with st.container(border=True):
            top = st.columns([3, 2, 2])
            top[0].markdown(f"**{icon} {finding}**")
            top[1].markdown(f"`{r['status']}`")
            if r["confidence"]:
                top[2].markdown(f"confidence: {r['confidence']:.2f}")
            if r["matched_sentence"]:
                st.caption(f"Matched sentence: “{r['matched_sentence']}”")
            if r.get("reason"):
                st.caption(f"Reason: {r['reason']}")


# ============================================================
# Layout
# ============================================================

st.title("🦵 Knee MRI Abnormality Analysis")
st.caption("Local capstone demonstration")

with st.sidebar:
    st.header("About this demo")
    st.write(
        "This project has two independent parts, shown as two sections below: "
        "an image-quality check for an uploaded scan, and an NLP model that "
        "extracts structured findings from radiology **report text**."
    )
    st.info(
        "There is no trained image classifier in this project, so the "
        "uploaded scan alone cannot produce a diagnosis. The structured "
        "findings (Medial OA, effusion, etc.) come from the report-text "
        "NLP pipeline in the notebook — paste a report below to see it run."
    )
    if UNAVAILABLE_TARGETS:
        st.caption(
            f"Not yet implemented in the notebook: {', '.join(UNAVAILABLE_TARGETS)}."
        )

st.header("1. Uploaded scan (image quality check)")

uploaded = st.file_uploader(
    "Upload one knee image or DICOM file",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    help="Supported: PNG, JPG, JPEG, and DICOM (.dcm/.dicom).",
)

if uploaded is None:
    st.markdown("Choose an MRI screenshot/image or a DICOM file above.")
else:
    try:
        image, metadata = read_uploaded_file(uploaded.name, uploaded.getvalue())
        result = analyze_image(image)

        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True, caption=uploaded.name)

        with right:
            st.subheader("Image-quality result")
            if result["quality"] == "Good for prototype display":
                st.success(result["quality"])
            else:
                st.warning(result["quality"])
            st.metric("Image dimensions", f"{image.width} × {image.height} px")
            st.metric("Contrast range", f"{result['contrast']:.1%}")
            st.metric("Mean intensity", f"{result['mean_intensity']:.1%}")

        with st.expander("File information"):
            st.json(metadata)

        if result["warnings"]:
            for warning in result["warnings"]:
                st.write(f"⚠️ {warning}")
        else:
            st.write("✅ No basic image-quality warnings detected.")

    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.info("Try a standard PNG/JPG image or an uncompressed DICOM file.")

st.divider()
st.header("2. Structured findings (NLP model on report text)")
st.write(
    "Paste the radiology report text for this study (the Findings/Impression "
    "section) to run the notebook's rule-based classifiers — this is the "
    "part that produces POSITIVE/NEGATIVE/UNCERTAIN labels with matched "
    "sentences, the same logic behind the metrics printed in your terminal."
)

sample_report = (
    "MEDIAL COMPARTMENT: Medial meniscus: Normal in morphology and signal. "
    "Medial compartment cartilage: Cartilage appears intact.\n"
    "LATERAL COMPARTMENT: Lateral compartment cartilage: Mild cartilage "
    "thinning and fissuring is present.\n"
    "JOINT SPACE: Small joint effusion. Moderate-sized Baker cyst with "
    "slight surrounding edema.\n"
    "COLLATERAL LIGAMENTS: Medial collateral ligament (MCL): Normal."
)

report_text = st.text_area(
    "Radiology report text",
    height=220,
    placeholder=sample_report,
)

col_run, col_sample = st.columns([1, 1])
run_clicked = col_run.button("Run findings extraction", type="primary")
use_sample = col_sample.button("Load sample report")

if use_sample:
    report_text = sample_report
    st.session_state["_sample_loaded"] = sample_report

if "_sample_loaded" in st.session_state and not report_text:
    report_text = st.session_state["_sample_loaded"]

if run_clicked or use_sample:
    if not report_text or not report_text.strip():
        st.warning("Paste some report text first (or click “Load sample report”).")
    else:
        results = analyze_report(report_text)
        render_findings(results)
        st.caption(
            "This mirrors the notebook's own weak-labeling pipeline exactly — "
            "including its known limits (e.g. some findings are missed when "
            "the relevant words fall in different sentences than the anatomy "
            "they describe, which is reflected in the notebook's own "
            "coverage/precision/recall numbers)."
        )
