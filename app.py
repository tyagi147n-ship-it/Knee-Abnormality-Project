from io import BytesIO
import re

import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Knee Abnormality Detection",
    page_icon="🦵",
    layout="wide",
)


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


TERMS = {
    "ACL": ["acl", "anterior cruciate ligament"],
    "MCL": ["mcl", "medial collateral ligament"],
    "Medial Meniscus": ["medial meniscus", "medial meniscal"],
    "Lateral Meniscus": ["lateral meniscus", "lateral meniscal"],
    "Medial OA": [
        "medial osteoarthritis",
        "medial osteoarthrosis",
        "medial chondrosis",
        "medial compartment",
    ],
    "Lateral OA": [
        "lateral osteoarthritis",
        "lateral osteoarthrosis",
        "lateral chondrosis",
        "lateral compartment",
    ],
    "PF OA": [
        "patellofemoral osteoarthritis",
        "patellofemoral chondrosis",
        "patellofemoral compartment",
    ],
    "Effusion": ["effusion", "joint fluid", "derrame"],
    "Synovitis": ["synovitis", "synovial thickening"],
    "Baker's": ["baker cyst", "baker's cyst", "popliteal cyst"],
    "Contusion": ["contusion", "bone bruise", "bone marrow edema"],
    "Fracture": ["fracture", "fractured", "fraktur"],
}


NEGATIVE_WORDS = [
    "no ",
    "without",
    "normal",
    "intact",
    "unremarkable",
    "absent",
    "not seen",
]


UNCERTAIN_WORDS = [
    "possible",
    "possibly",
    "suspected",
    "cannot exclude",
    "suggestive",
    "likely",
]


DEFAULT_REPORT = """Findings:

ACL tear is present.
MCL tear is present.
Medial meniscus tear is present.
Lateral meniscus tear is present.
Medial compartment osteoarthritis is present.
Lateral compartment osteoarthritis is present.
Patellofemoral osteoarthritis is present.
Joint effusion is present.
Synovitis is present.
Baker's cyst is present.
Bone contusion is present.
Fracture is present.
"""


def normalize_image(array):
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array)

    if array.ndim > 2:
        array = array[array.shape[0] // 2]

    low, high = np.percentile(array, [1, 99])

    if high <= low:
        low = array.min()
        high = array.max()

    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    return np.clip(
        (array - low) / (high - low) * 255,
        0,
        255,
    ).astype(np.uint8)


def load_uploaded_file(filename, content):
    if filename.lower().endswith((".dcm", ".dicom")):
        import pydicom

        dataset = pydicom.dcmread(
            BytesIO(content),
            force=True,
        )

        if not hasattr(dataset, "PixelData"):
            raise ValueError("This DICOM file has no pixel data.")

        image = Image.fromarray(
            normalize_image(dataset.pixel_array)
        ).convert("L")

        metadata = {
            "Format": "DICOM",
            "Rows": getattr(dataset, "Rows", image.height),
            "Columns": getattr(dataset, "Columns", image.width),
            "Modality": str(
                getattr(dataset, "Modality", "Unknown")
            ),
        }

    else:
        image = Image.open(
            BytesIO(content)
        ).convert("L")

        metadata = {
            "Format": "Image",
            "Rows": image.height,
            "Columns": image.width,
            "Modality": "Not available",
        }

    return image, metadata


def analyze_report(report):
    text = re.sub(
        r"\s+",
        " ",
        report.lower()
    ).strip()

    results = []

    for abnormality in TARGETS:
        matches = [
            term
            for term in TERMS[abnormality]
            if term in text
        ]

        if not matches:
            status = "NOT_MENTIONED"
            probability = 0.00
            confidence = 0.00
            reason = (
                f"No {abnormality} evidence detected"
            )

        elif any(
            word in text
            for word in UNCERTAIN_WORDS
        ):
            status = "UNCERTAIN"
            probability = 0.50
            confidence = 0.50
            reason = (
                "Uncertain wording found in the report"
            )

        elif any(
            word in text
            for word in NEGATIVE_WORDS
        ):
            status = "NEGATIVE"
            probability = 0.20
            confidence = 0.80
            reason = (
                "Negative or normal wording found"
            )

        else:
            status = "POSITIVE"
            probability = 0.80
            confidence = 0.80
            reason = (
                "Abnormality term found in the report"
            )

        results.append({
            "abnormality": abnormality,
            "status": status,
            "probability": probability,
            "confidence": confidence,
            "accuracy": "N/A for one upload",
            "matched": (
                ", ".join(matches)
                if matches
                else "None"
            ),
            "reason": reason,
        })

    return results


st.title("🦵 Knee Abnormality Detection")
st.caption("DICOM upload with radiology-report analysis")


uploaded = st.file_uploader(
    "Upload one image or DICOM file",
    type=[
        "png",
        "jpg",
        "jpeg",
        "dcm",
        "dicom",
    ],
)


if uploaded is not None:
    try:
        image, metadata = load_uploaded_file(
            uploaded.name,
            uploaded.getvalue(),
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")

            st.image(
                image,
                use_container_width=True,
                caption=uploaded.name,
            )

        with right:
            st.subheader("Image information")

            st.json(metadata)

            pixels = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

            contrast = (
                np.percentile(pixels, 99)
                - np.percentile(pixels, 1)
            )

            st.metric(
                "Image quality",
                "Good for prototype display"
            )

            st.metric(
                "Contrast",
                f"{contrast:.1%}"
            )

            st.metric(
                "Mean intensity",
                f"{pixels.mean():.1%}"
            )

        st.divider()

        st.subheader("Radiology report analysis")

        report = st.text_area(
            "Radiology report",
            value=DEFAULT_REPORT,
            height=220,
        )

        results = analyze_report(report)

        st.divider()

        st.subheader("Abnormality results")

        for result in results:
            with st.container(border=True):

                title_col, status_col, metric_col = st.columns(
                    [2.5, 1.3, 1.5]
                )

                with title_col:
                    st.markdown(
                        f"### 🟢 "
                        f"{result['abnormality']}"
                    )

                with status_col:
                    st.markdown(
                        f"**{result['status']}**"
                    )

                with metric_col:
                    st.markdown(
                        f"**Confidence:** "
                        f"{result['confidence']:.0%}"
                    )

                    st.markdown(
                        f"**Probability:** "
                        f"{result['probability']:.0%}"
                    )

                    st.markdown(
                        f"**Accuracy:** "
                        f"{result['accuracy']}"
                    )

                st.markdown(
                    f"**Matched sentence:** "
                    f"`{result['matched']}`"
                )

                st.markdown(
                    f"**Reason:** "
                    f"{result['reason']}"
                )

        st.warning(
            "Probability and confidence are prototype "
            "report-text scores. Accuracy requires a "
            "labeled test dataset."
        )

    except Exception as error:
        st.error(
            f"Could not read this file: {error}"
        )

else:
    st.info(
        "Upload a DICOM or image file to begin."
    )