# %%
import re
import pandas as pd
import numpy as np

from pathlib import Path

print("Environment initialized successfully.")

# %%
# ============================================================
# CELL 2 — LOAD DATA
# ============================================================

from pathlib import Path
import pandas as pd
from IPython.display import display

DATA_PATH = Path(__file__).resolve().parent / "train.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# Actually load the CSV
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# %%
# ============================================================
# CELL 3 — BASIC INSPECTION
# ============================================================

print("FIRST 5 ROWS")
display(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nDUPLICATES")
print("Duplicate rows:", df.duplicated().sum())

# %%
# ============================================================
# CELL 6 — INSPECT GOLD LABEL AVAILABILITY
# ============================================================

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

print("Total studies:", len(df))

print("\nNumber of non-missing labels per target:")
print(df[TARGET_COLUMNS].notna().sum())

print("\nNumber of missing labels per target:")
print(df[TARGET_COLUMNS].isna().sum())

print("\nRows where ALL 12 gold labels are available:")
gold_mask = df[TARGET_COLUMNS].notna().all(axis=1)
print(gold_mask.sum())

print("\nRows where at least one gold label is available:")
partial_gold_mask = df[TARGET_COLUMNS].notna().any(axis=1)
print(partial_gold_mask.sum())

# %%
# ============================================================
# CELL 7 — CREATE GOLD DATAFRAME
# ============================================================

gold = df.loc[gold_mask].copy()

print("Gold dataframe shape:", gold.shape)

print("\nGold labels:")
display(gold[TARGET_COLUMNS].head())

print("\nGold label distributions:")
for col in TARGET_COLUMNS:
    print(f"\n{col}")
    print(gold[col].value_counts(dropna=False))

# %%
# ============================================================
# CELL 9 — INSPECT ALL GOLD REPORTS
# ============================================================

for i, (_, row) in enumerate(gold.iterrows(), start=1):

    print("\n" + "=" * 120)
    print(f"GOLD REPORT {i}/58")
    print("=" * 120)

    print("StudyInstanceUID:")
    print(row["StudyInstanceUID"])

    print("\nLABELS:")
    print(
        row[TARGET_COLUMNS]
        .astype(int)
        .to_dict()
    )

    print("\nREPORT:")
    print(row["Report"])

    print("=" * 120)

# %%
# ============================================================
# CELL 10 — REPORT LENGTH / BASIC TEXT ANALYSIS
# ============================================================

gold = gold.copy()

gold["Report_length_chars"] = gold["Report"].astype(str).str.len()
gold["Report_length_words"] = (
    gold["Report"]
    .astype(str)
    .str.split()
    .str.len()
)

print("REPORT LENGTH STATISTICS")
print("=" * 80)

print(gold[
    ["Report_length_chars", "Report_length_words"]
].describe())

# %%
# ============================================================
# CELL 11 — SAMPLE REPORTS BY ACL LABEL
# ============================================================

print("========== ACL POSITIVE REPORTS ==========")

for _, row in gold[gold["ACL"] == 1].head(10).iterrows():
    print("\n" + "-" * 100)
    print(row["Report"])


print("\n\n========== ACL NEGATIVE REPORTS ==========")

for _, row in gold[gold["ACL"] == 0].head(10).iterrows():
    print("\n" + "-" * 100)
    print(row["Report"])

# %% [markdown]
# # **ACL WEAK LABEL V1**
# 
# 

# %%
# ============================================================
# ACL NEGATIVE PATTERNS — V3
# ============================================================

ACL_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # ACL / ACL NAME followed by negative statement
    # --------------------------------------------------------

    r"\b(?:acl|anterior cruciate ligament)\b"
    r"[^.!?]{0,40}"
    r"\b(?:no|without)\b"
    r"[^.!?]{0,30}"
    r"\b(?:tear|rupture|injury|sprain|abnormality)\b",

    # Example:
    # "ACL: No tear"
    r"\b(?:acl|anterior cruciate ligament)\b"
    r"\s*[:\-]\s*"
    r"(?:no|without)\s+"
    r"(?:evidence of\s+)?"
    r"(?:tear|rupture|injury|abnormality)",

    # --------------------------------------------------------
    # Negative phrase BEFORE ACL
    # --------------------------------------------------------

    r"\b(?:no|without)\b"
    r"[^.!?]{0,40}"
    r"\b(?:acl|anterior cruciate ligament)\b"
    r"[^.!?]{0,30}"
    r"\b(?:tear|rupture|injury|abnormality)\b",

    # --------------------------------------------------------
    # ACL followed by intact / normal / preserved
    # --------------------------------------------------------

    r"\b(?:acl|anterior cruciate ligament)\b"
    r"[^.!?]{0,40}"
    r"\b(?:intact|normal|unremarkable|preserved)\b",

    # Negative adjective before ACL
    r"\b(?:normal|intact|unremarkable|preserved)\b"
    r"[^.!?]{0,30}"
    r"\b(?:acl|anterior cruciate ligament)\b",

    # --------------------------------------------------------
    # Explicit "no abnormality" forms
    # --------------------------------------------------------

    r"\bno\b"
    r"[^.!?]{0,30}"
    r"\b(?:abnormality|abnormalities)\b"
    r"[^.!?]{0,20}"
    r"\b(?:acl|anterior cruciate ligament)\b",

    # --------------------------------------------------------
    # "no evidence of" forms
    # --------------------------------------------------------

    r"\bno evidence of\b"
    r"[^.!?]{0,40}"
    r"\b(?:acl|anterior cruciate ligament)\b",

    # ========================================================
    # SPANISH
    # ========================================================

    r"\b(?:lca|ligamento cruzado anterior)\b"
    r"[^.!?]{0,40}"
    r"\b(?:sin|no)\b"
    r"[^.!?]{0,30}"
    r"\b(?:rotura|ruptura|lesión|esguince)\b",

    r"\b(?:lca|ligamento cruzado anterior)\b"
    r"[^.!?]{0,30}"
    r"\b(?:normal|íntegro|integro|conservado)\b",

    # ========================================================
    # TURKISH
    # ========================================================

    r"\b(?:ön çapraz bağ|ön çapraz bağı)\b"
    r"[^.!?]{0,40}"
    r"\b(?:normal|sağlam|intakt)\b",

    # ========================================================
    # DUTCH
    # ========================================================

    r"\bvoorste kruisband\b"
    r"[^.!?]{0,40}"
    r"\b(?:normaal|intact|onopvallend)\b",

    # ========================================================
    # BULGARIAN
    # ========================================================

    r"\bпредна кръстна връзка\b"
    r"[^.!?]{0,40}"
    r"\b(?:нормална|запазена|интактна)\b",
]

print("ACL V3 negative patterns loaded.")

# %%
# ============================================================
# ACL UNIT TESTS
# ============================================================

def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text))
        if sentence.strip()
    ]


def first_match(sentence, patterns):
    for pattern in patterns:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return pattern
    return None


ACL_POSITIVE_PATTERNS = [
    r"\bacl\b.{0,60}\b(?:tear|rupture|ruptured|injury|sprain)\b",
    r"\b(?:tear|rupture|ruptured|injury|sprain)\b.{0,60}\bacl\b",
    r"\banterior cruciate ligament\b.{0,60}\b(?:tear|rupture|injury)\b",
    r"\b(?:tear|rupture|injury)\b.{0,60}\banterior cruciate ligament\b",
]


def acl_label_v3(report):
    if pd.isna(report) or not str(report).strip():
        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report",
        }

    text = str(report)
    sentences = split_sentences(text)

    # Special cases
    if re.search(
        r"\b(?:history of|prior|previously|old)\b.{0,50}\b(?:acl|anterior cruciate ligament)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return {
            "label": np.nan,
            "status": "HISTORICAL_OR_COMPARISON",
            "confidence": 0.50,
            "matched_sentence": text,
            "matched_pattern": "",
            "reason": "Historical ACL statement",
        }

    if re.search(
        r"\b(?:acl reconstruction|status post acl|post[- ]surgical acl)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return {
            "label": np.nan,
            "status": "POST_SURGICAL",
            "confidence": 0.50,
            "matched_sentence": text,
            "matched_pattern": "",
            "reason": "Post-surgical ACL statement",
        }

    for sentence in sentences:
        if re.search(
            r"\b(?:possible|possibly|suspected|cannot exclude|suggestive of)\b"
            r".{0,60}\b(?:acl|anterior cruciate ligament)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            return {
                "label": np.nan,
                "status": "UNCERTAIN",
                "confidence": 0.50,
                "matched_sentence": sentence,
                "matched_pattern": "",
                "reason": "Uncertain ACL statement",
            }

        negative_pattern = first_match(
            sentence,
            ACL_NEGATIVE_PATTERNS,
        )

        if negative_pattern:
            return {
                "label": 0,
                "status": "NEGATIVE",
                "confidence": 0.98,
                "matched_sentence": sentence,
                "matched_pattern": negative_pattern,
                "reason": "Explicit negative ACL statement",
            }

        positive_pattern = first_match(
            sentence,
            ACL_POSITIVE_PATTERNS,
        )

        if positive_pattern:
            return {
                "label": 1,
                "status": "POSITIVE",
                "confidence": 0.95,
                "matched_sentence": sentence,
                "matched_pattern": positive_pattern,
                "reason": "Positive ACL evidence detected",
            }

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No ACL evidence detected",
    }


def analyze_report(report):
    return {
        "ACL": {
            "state": acl_label_v3(report)["status"]
        }
    }

# %%
# ============================================================
# CELL — ACL V3 GOLD EVALUATION
# ============================================================

# APPLY ACL V3 TO ALL REPORTS

acl_results_v3 = df["Report"].apply(acl_label_v3)

acl_results_v3_df = pd.DataFrame(
    acl_results_v3.tolist(),
    index=df.index
)

acl_results_v3_df = acl_results_v3_df.rename(columns={
    "label": "ACL_weak_v3",
    "status": "ACL_status_v3",
    "confidence": "ACL_confidence_v3",
    "matched_sentence": "ACL_sentence_v3",
    "matched_pattern": "ACL_match_v3",
    "reason": "ACL_reason_v3",
})

df_acl_v3 = pd.concat(
    [df, acl_results_v3_df],
    axis=1
)

print("ACL V3 dataframe created.")

gold_acl_v3 = df_acl_v3.loc[gold.index].copy()

gold_acl_v3["ACL_gold"] = gold_acl_v3["ACL"].astype(int)

comparison_v3 = gold_acl_v3[
    [
        "StudyInstanceUID",
        "Report",
        "ACL_gold",
        "ACL_weak_v3",
        "ACL_status_v3",
        "ACL_confidence_v3",
        "ACL_sentence_v3",
        "ACL_match_v3",
        "ACL_reason_v3"
    ]
].copy()

evaluated_v3 = comparison_v3[
    comparison_v3["ACL_weak_v3"].isin([0, 1])
].copy()

TP = (
    (evaluated_v3["ACL_gold"] == 1) &
    (evaluated_v3["ACL_weak_v3"] == 1)
).sum()

TN = (
    (evaluated_v3["ACL_gold"] == 0) &
    (evaluated_v3["ACL_weak_v3"] == 0)
).sum()

FP = (
    (evaluated_v3["ACL_gold"] == 0) &
    (evaluated_v3["ACL_weak_v3"] == 1)
).sum()

FN = (
    (evaluated_v3["ACL_gold"] == 1) &
    (evaluated_v3["ACL_weak_v3"] == 0)
).sum()

coverage = len(evaluated_v3) / len(comparison_v3)

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) else 0
)

print("=" * 70)
print("ACL V3 RESULTS")
print("=" * 70)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))

print("\nCoverage:", round(coverage * 100, 2), "%")
print(
    "Uncertain / Not Mentioned:",
    len(comparison_v3) - len(evaluated_v3)
)

# %%
# ============================================================
# CELL 15 — ACL CONFUSION CATEGORIES
# ============================================================
# Compatibility dataframe for the older ACL V1 analysis

gold_acl = gold_acl_v3.copy()

gold_acl["ACL_weak_v1"] = gold_acl["ACL_weak_v3"]
gold_acl["ACL_status_v1"] = gold_acl["ACL_status_v3"]
gold_acl["ACL_confidence_v1"] = gold_acl["ACL_confidence_v3"]
gold_acl["ACL_sentence_v1"] = gold_acl["ACL_sentence_v3"]
gold_acl["ACL_match_v1"] = gold_acl["ACL_match_v3"]
gold_acl["ACL_reason_v1"] = gold_acl["ACL_reason_v3"]

comparison = gold_acl[
    ["StudyInstanceUID",
     "Report",
     "ACL_gold",
     "ACL_weak_v1",
     "ACL_status_v1",
     "ACL_confidence_v1",
     "ACL_sentence_v1",
     "ACL_match_v1",
     "ACL_reason_v1"]
].copy()

# Only compare rows where NLP produced a binary label
evaluated = comparison[
    comparison["ACL_weak_v1"].isin([0, 1])
].copy()

TP = (
    (evaluated["ACL_gold"] == 1) &
    (evaluated["ACL_weak_v1"] == 1)
).sum()

TN = (
    (evaluated["ACL_gold"] == 0) &
    (evaluated["ACL_weak_v1"] == 0)
).sum()

FP = (
    (evaluated["ACL_gold"] == 0) &
    (evaluated["ACL_weak_v1"] == 1)
).sum()

FN = (
    (evaluated["ACL_gold"] == 1) &
    (evaluated["ACL_weak_v1"] == 0)
).sum()

print("=" * 70)
print("ACL V1 CONFUSION MATRIX")
print("=" * 70)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nNot evaluated / uncertain / not mentioned:",
      len(comparison) - len(evaluated))

# %%
# ============================================================
# ACL V3 — FALSE POSITIVES
# ============================================================

acl_fp_v3 = comparison_v3[
    (comparison_v3["ACL_gold"] == 0) &
    (comparison_v3["ACL_weak_v3"] == 1)
]

print("=" * 100)
print("ACL V3 FALSE POSITIVES:", len(acl_fp_v3))
print("=" * 100)

for _, row in acl_fp_v3.iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["ACL_gold"])
    print("Weak:", row["ACL_weak_v3"])
    print("Status:", row["ACL_status_v3"])
    print("Match:", row["ACL_match_v3"])

    print("\nMatched sentence:")
    print(row["ACL_sentence_v3"])

    print("\nReason:")
    print(row["ACL_reason_v3"])

    print("\nFull report:")
    print(row["Report"])

# %%
# ============================================================
# ACL V3 — FALSE NEGATIVES
# ============================================================

acl_fn_v3 = comparison_v3[
    (comparison_v3["ACL_gold"] == 1) &
    (comparison_v3["ACL_weak_v3"] == 0)
]

print("=" * 100)
print("ACL V3 FALSE NEGATIVES:", len(acl_fn_v3))
print("=" * 100)

for _, row in acl_fn_v3.iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["ACL_gold"])
    print("Weak:", row["ACL_weak_v3"])
    print("Status:", row["ACL_status_v3"])
    print("Match:", row["ACL_match_v3"])

    print("\nMatched sentence:")
    print(row["ACL_sentence_v3"])

    print("\nReason:")
    print(row["ACL_reason_v3"])

    print("\nFull report:")
    print(row["Report"])

# %%
# ============================================================
# CELL — ACL V3 GOLD STATUS BREAKDOWN
# ============================================================

print("=" * 80)
print("ACL V3 — GOLD STATUS BREAKDOWN")
print("=" * 80)

status_table = (
    comparison_v3
    .groupby(["ACL_gold", "ACL_status_v3"])
    .size()
    .reset_index(name="count")
)

display(status_table)

# %%
# ============================================================
# CELL — ACL GOLD POSITIVES MISSED BY NLP
# ============================================================

acl_gold_positive_missed = comparison_v3[
    (comparison_v3["ACL_gold"] == 1) &
    (comparison_v3["ACL_weak_v3"].isna())
]

print("=" * 100)
print(
    "GOLD POSITIVE BUT NLP DID NOT PRODUCE BINARY LABEL:",
    len(acl_gold_positive_missed)
)
print("=" * 100)

for _, row in acl_gold_positive_missed.iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["ACL_gold"])
    print("Weak:", row["ACL_weak_v3"])
    print("Status:", row["ACL_status_v3"])

    print("\nReport:")
    print(row["Report"])

# %%
# ============================================================
# CELL — ACL GOLD NEGATIVES MISSED BY NLP
# ============================================================

acl_gold_negative_missed = comparison_v3[
    (comparison_v3["ACL_gold"] == 0) &
    (comparison_v3["ACL_weak_v3"].isna())
]

print("=" * 100)
print(
    "GOLD NEGATIVE BUT NLP DID NOT PRODUCE BINARY LABEL:",
    len(acl_gold_negative_missed)
)
print("=" * 100)

for _, row in acl_gold_negative_missed.iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["ACL_gold"])
    print("Weak:", row["ACL_weak_v3"])
    print("Status:", row["ACL_status_v3"])

    print("\nReport:")
    print(row["Report"])

# %%
# ============================================================
# CELL 18 — TARGET CONFIGURATION
# ============================================================

TARGET_CONFIG = {

    "MCL": {

        "positive_patterns": [
            r"\bmcl\b.{0,60}\b(?:tear|rupture|ruptured|injury|sprain)\b",
            r"\b(?:tear|rupture|ruptured|injury|sprain)\b.{0,60}\bmcl\b",
            r"\bmedial collateral ligament\b.{0,60}\b(?:tear|rupture|injury|sprain)\b",
            r"\b(?:tear|rupture|injury|sprain)\b.{0,60}\bmedial collateral ligament\b",
        ],

        "negative_patterns": [
            r"\bmcl\b.{0,40}\b(?:intact|normal|unremarkable|preserved)\b",
            r"\bmedial collateral ligament\b.{0,40}\b(?:intact|normal|unremarkable|preserved)\b",
            r"\b(?:normal|intact|unremarkable|preserved)\b.{0,40}\bmcl\b",
            r"\bno\b.{0,40}\bmcl\b.{0,40}\b(?:tear|rupture|injury)\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot exclude)\b.{0,60}\bmcl\b",
            r"\b(?:mcl|medial collateral ligament)\b.{0,60}\b(?:possible|suspected|suggestive|likely)\b",
        ],
    },


    "Medial Meniscus": {

        "positive_patterns": [
            r"\bmedial meniscus\b.{0,80}\b(?:tear|torn|rupture|maceration|macerated|fragment)\b",
            r"\bmedial meniscal\b.{0,80}\b(?:tear|torn|rupture)\b",
            r"\bmeniscus\b.{0,50}\b(?:medial|medial compartment)\b.{0,50}\b(?:tear|rupture)\b",
        ],

        "negative_patterns": [
            r"\bmedial meniscus\b.{0,50}\b(?:intact|normal|unremarkable|preserved|not torn)\b",
            r"\b(?:medial meniscus)\b.{0,50}\bwithout\b.{0,40}\btear\b",
            r"\bno\b.{0,40}\b(?:tear|rupture)\b.{0,40}\bmedial meniscus\b",
            r"\b(?:medial meniscus)\b.{0,40}\bno\b.{0,30}\btear\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot exclude)\b.{0,60}\bmedial meniscus\b",
            r"\bmedial meniscus\b.{0,60}\b(?:possible|suspected|suggestive|likely)\b",
        ],
    },


    "Lateral Meniscus": {

        "positive_patterns": [
            r"\blateral meniscus\b.{0,80}\b(?:tear|torn|rupture|maceration|macerated|fragment)\b",
            r"\blateral meniscal\b.{0,80}\b(?:tear|torn|rupture)\b",
            r"\bmeniscus\b.{0,50}\b(?:lateral|lateral compartment)\b.{0,50}\b(?:tear|rupture)\b",
        ],

        "negative_patterns": [
            r"\blateral meniscus\b.{0,50}\b(?:intact|normal|unremarkable|preserved|not torn)\b",
            r"\blateral meniscus\b.{0,50}\bwithout\b.{0,40}\btear\b",
            r"\bno\b.{0,40}\b(?:tear|rupture)\b.{0,40}\blateral meniscus\b",
            r"\blateral meniscus\b.{0,40}\bno\b.{0,30}\btear\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot exclude)\b.{0,60}\blateral meniscus\b",
            r"\blateral meniscus\b.{0,60}\b(?:possible|suspected|suggestive|likely)\b",
        ],
    },


    "Medial OA": {

        "positive_patterns": [
            r"\bmedial compartment\b.{0,100}\b(?:osteoarthritis|oa|chondrosis|chondral loss|cartilage loss|degenerative changes|osteophytes)\b",
            r"\bmedial\b.{0,50}\b(?:osteoarthritis|oa|chondrosis)\b",
            r"\b(?:osteoarthritis|oa|chondrosis)\b.{0,50}\bmedial compartment\b",
            r"\bmedial femorotibial\b.{0,80}\b(?:osteoarthritis|oa|chondrosis|cartilage loss)\b",
        ],

        "negative_patterns": [
            r"\bmedial compartment\b.{0,60}\b(?:normal|no abnormality|without abnormality)\b",
            r"\bno\b.{0,50}\bmedial\b.{0,50}\b(?:osteoarthritis|oa)\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|possible early|suspected|suggestive)\b.{0,60}\b(?:medial oa|medial osteoarthritis|medial compartment)\b",
        ],
    },


    "Lateral OA": {

        "positive_patterns": [
            r"\blateral compartment\b.{0,100}\b(?:osteoarthritis|oa|chondrosis|chondral loss|cartilage loss|degenerative changes|osteophytes)\b",
            r"\blateral\b.{0,50}\b(?:osteoarthritis|oa|chondrosis)\b",
            r"\b(?:osteoarthritis|oa|chondrosis)\b.{0,50}\blateral compartment\b",
            r"\blateral femorotibial\b.{0,80}\b(?:osteoarthritis|oa|chondrosis|cartilage loss)\b",
        ],

        "negative_patterns": [
            r"\blateral compartment\b.{0,60}\b(?:normal|no abnormality|without abnormality)\b",
            r"\bno\b.{0,50}\blateral\b.{0,50}\b(?:osteoarthritis|oa)\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suggestive)\b.{0,60}\b(?:lateral oa|lateral osteoarthritis|lateral compartment)\b",
        ],
    },


    "PF OA": {

        "positive_patterns": [
            r"\bpatellofemoral\b.{0,100}\b(?:osteoarthritis|oa|chondrosis|chondral loss|cartilage loss|degenerative changes|osteophytes)\b",
            r"\bpatellofemoral\b.{0,80}\b(?:arthrosis|arthritis)\b",
            r"\b(?:patellar|trochlear)\b.{0,70}\b(?:chondrosis|cartilage loss|osteoarthritis|oa)\b",
            r"\bpatellar\b.{0,50}\b(?:chondromalacia|chondropathy)\b",
        ],

        "negative_patterns": [
            r"\bpatellofemoral\b.{0,60}\b(?:normal|intact|no abnormality)\b",
            r"\bno\b.{0,50}\bpatellofemoral\b.{0,50}\b(?:oa|osteoarthritis|chondrosis)\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suggestive)\b.{0,60}\bpatellofemoral\b",
        ],
    },


    "Effusion": {

        "positive_patterns": [
            r"\b(?:joint|knee|intra[- ]articular)\b.{0,50}\b(?:effusion|fluid|collection)\b",
            r"\b(?:effusion|joint fluid|fluid accumulation)\b",
            r"\b(?:mild|moderate|small|large|trace|massive)\b.{0,20}\b(?:joint effusion|effusion)\b",
            r"\b(?:derrame|derrame articular)\b",
            r"\b(?:výpotek|izljev)\b",
        ],

        "negative_patterns": [
            r"\bno\b.{0,30}\b(?:joint|knee)?\s*effusion\b",
            r"\bno evidence of\b.{0,30}\beffusion\b",
            r"\bwithout\b.{0,30}\beffusion\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suggestive)\b.{0,40}\beffusion\b",
        ],
    },


    "Synovitis": {

        "positive_patterns": [
            r"\bsynovitis\b",
            r"\bsynovial\b.{0,60}\b(?:thickening|hypertrophy|proliferation)\b",
            r"\bthickened synovium\b",
            r"\bsynovial membrane thickening\b",
            r"\b(?:sinovitis|synovitis)\b",
        ],

        "negative_patterns": [
            r"\bno\b.{0,30}\bsynovitis\b",
            r"\bwithout\b.{0,30}\bsynovitis\b",
            r"\bno\b.{0,30}\bsynovial\b.{0,30}\bthickening\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suggestive)\b.{0,40}\bsynovitis\b",
        ],
    },


    "Baker's": {

        "positive_patterns": [
            r"\bbaker['’]s?\s+cyst\b",
            r"\bpopliteal cyst\b",
            r"\bpopliteal fluid collection\b",
            r"\bcyst\b.{0,50}\bpopliteal\b",
        ],

        "negative_patterns": [
            r"\bno\b.{0,30}\bbaker['’]s?\s+cyst\b",
            r"\bno\b.{0,30}\bpopliteal cyst\b",
            r"\bwithout\b.{0,30}\bbaker['’]s?\s+cyst\b",
            r"\bbaker['’]s?\s+cyst\b.{0,30}\bnone\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suggestive)\b.{0,40}\b(?:baker['’]s?\s+cyst|popliteal cyst)\b",
        ],
    },


    "Contusion": {

        "positive_patterns": [
            r"\bbone contusion\b",
            r"\bbone bruise\b",
            r"\bbone marrow edema\b.{0,80}\b(?:contusion|bruise)\b",
            r"\bcontusion\b.{0,50}\b(?:bone|bony|osseous|marrow)\b",
            r"\boste?e?dema\b",
        ],

        "negative_patterns": [
            r"\bno\b.{0,30}\b(?:bone contusion|bone bruise)\b",
            r"\bwithout\b.{0,30}\b(?:bone contusion|bone bruise)\b",
            r"\bno\b.{0,40}\bbone\b.{0,30}\bcontusion\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|suspected|suspect|suggestive)\b.{0,50}\b(?:bone contusion|bone bruise)\b",
        ],
    },


    "Fracture": {

        "positive_patterns": [
            r"\bfracture\b",
            r"\bfractured\b",
            r"\bfracture line\b",
            r"\bcortical break\b",
            r"\binsufficiency fracture\b",
            r"\bimpaction fracture\b",
            r"\bosteochondral fracture\b",
            r"\bavulsion fracture\b",
            r"\bfractura\b",
            r"\bfraktura\b",
            r"\bfraktur\b",
        ],

        "negative_patterns": [
            r"\bno\b.{0,30}\bfracture\b",
            r"\bwithout\b.{0,30}\bfracture\b",
            r"\bno evidence of\b.{0,30}\bfracture\b",
            r"\bfracture\b.{0,30}\bnone\b",
        ],

        "uncertain_patterns": [
            r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot exclude)\b.{0,50}\bfracture\b",
        ],
    },
}


print("Target configuration created.")
print("Targets:", list(TARGET_CONFIG.keys()))

# %%
# ============================================================
# CELL 19 — GENERIC WEAK-LABEL EXTRACTOR
# ============================================================

def detect_target(report, target):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    if target not in TARGET_CONFIG:
        raise ValueError(
            f"Target '{target}' is not configured."
        )

    config = TARGET_CONFIG[target]

    sentences = split_sentences(str(report))

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        positive_match = first_match(
            sentence,
            config["positive_patterns"]
        )

        negative_match = first_match(
            sentence,
            config["negative_patterns"]
        )

        uncertain_match = first_match(
            sentence,
            config["uncertain_patterns"]
        )

        if positive_match:
            positive_hits.append(
                (sentence, positive_match)
            )

        if negative_match:
            negative_hits.append(
                (sentence, negative_match)
            )

        if uncertain_match:
            uncertain_hits.append(
                (sentence, uncertain_match)
            )

    # --------------------------------------------------------
    # Explicit negative
    # --------------------------------------------------------

    if negative_hits and not positive_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": f"Explicit negative {target} statement"
        }

    # --------------------------------------------------------
    # Uncertain
    # --------------------------------------------------------

    if uncertain_hits and not positive_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": f"Uncertain {target} statement"
        }

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": f"Positive {target} evidence detected"
        }

    # --------------------------------------------------------
    # Nothing detected
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": f"No {target} evidence detected"
    }


print("Generic weak-label extractor created.")

# %%
# ============================================================
# CELL 20 — GENERATE 11 WEAK-LABEL SETS
# ============================================================

TARGETS_V2 = list(TARGET_CONFIG.keys())

weak_results = {}

for target in TARGETS_V2:

    print(f"Processing: {target}")

    results = df["Report"].apply(
        lambda text: detect_target(text, target)
    )

    result_df = pd.DataFrame(
        results.tolist(),
        index=df.index
    )

    safe_name = (
        target
        .replace(" ", "_")
        .replace("'", "")
    )

    result_df = result_df.rename(columns={
        "label": f"{safe_name}_weak",
        "status": f"{safe_name}_status",
        "confidence": f"{safe_name}_confidence",
        "matched_sentence": f"{safe_name}_sentence",
        "matched_pattern": f"{safe_name}_match",
        "reason": f"{safe_name}_reason"
    })

    weak_results[target] = result_df


print("\nAll 11 targets processed.")

# %%
# ============================================================
# CELL 21 — BUILD MASTER NLP DATAFRAME
# ============================================================

master_nlp = df.copy()

# ------------------------------------------------------------
# Add locked ACL V3
# ------------------------------------------------------------

master_nlp = pd.concat(
    [
        master_nlp,
        acl_results_v3_df
    ],
    axis=1
)

# ------------------------------------------------------------
# Add remaining 11 targets
# ------------------------------------------------------------

for target, result_df in weak_results.items():

    master_nlp = pd.concat(
        [
            master_nlp,
            result_df
        ],
        axis=1
    )


print("Master NLP dataframe created.")
print("Shape:", master_nlp.shape)

# %%
# ============================================================
# CELL 22 — EVALUATE ALL 12 TARGETS
# ============================================================

from sklearn.metrics import precision_score, recall_score, f1_score

# Locked ACL V3
weak_columns = {
    "ACL": "ACL_weak_v3",
    "MCL": "MCL_weak",
    "Medial Meniscus": "Medial_Meniscus_weak",
    "Lateral Meniscus": "Lateral_Meniscus_weak",
    "Medial OA": "Medial_OA_weak",
    "Lateral OA": "Lateral_OA_weak",
    "PF OA": "PF_OA_weak",
    "Effusion": "Effusion_weak",
    "Synovitis": "Synovitis_weak",
    "Baker's": "Bakers_weak",
    "Contusion": "Contusion_weak",
    "Fracture": "Fracture_weak"
}

results = []

for target, weak_col in weak_columns.items():

    gold_values = master_nlp.loc[gold.index, target]
    weak_values = master_nlp.loc[gold.index, weak_col]

    # Only evaluate binary predictions
    mask = weak_values.isin([0, 1])

    y_true = gold_values[mask].astype(int)
    y_pred = weak_values[mask].astype(int)

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    precision = (
        TP / (TP + FP)
        if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    coverage = mask.mean()

    results.append({
        "Target": target,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Coverage": round(coverage, 4),
        "Binary_Cases": mask.sum(),
        "Uncertain_or_NotMentioned": (~mask).sum()
    })


metrics_df = pd.DataFrame(results)

print("=" * 110)
print("ALL 12 TARGETS — INITIAL NLP EVALUATION")
print("=" * 110)

display(metrics_df)

# %%
# ============================================================
# CELL 23 — RANK TARGETS
# ============================================================

display(
    metrics_df
    .sort_values("F1", ascending=False)
    .reset_index(drop=True)
)

# %%
# ============================================================
# CELL 24 — COVERAGE ANALYSIS
# ============================================================

display(
    metrics_df[
        [
            "Target",
            "Coverage",
            "Binary_Cases",
            "Uncertain_or_NotMentioned"
        ]
    ]
    .sort_values("Coverage")
    .reset_index(drop=True)
)

# %%
# ============================================================
# MEDIAL MENISCUS POSITIVE PATTERNS — V3
# ============================================================

MEDIAL_MENISCUS_POSITIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit medial meniscus + abnormality
    # --------------------------------------------------------

    r"\bmedial meniscus\b.{0,100}\b"
    r"(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b",

    r"\bmedial meniscal\b.{0,100}\b"
    r"(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b",

    # Abnormality before medial meniscus
    r"\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b"
    r".{0,80}\bmedial meniscus\b",

    r"\b(?:tear|tearing|rupture)\b"
    r".{0,60}\bmedial meniscal\b",

    # --------------------------------------------------------
    # Specific tear types
    # --------------------------------------------------------

    r"\bmedial meniscus\b.{0,100}\b"
    r"(?:radial tear|horizontal tear|complex tear|root tear|"
    r"bucket[- ]handle tear|longitudinal tear|vertical tear|"
    r"oblique tear|degenerative tear)\b",

    r"\b(?:radial|horizontal|complex|root|bucket[- ]handle|"
    r"longitudinal|vertical|oblique|degenerative)\s+tear\b"
    r".{0,80}\bmedial meniscus\b",

    # --------------------------------------------------------
    # Horn / root / body
    # --------------------------------------------------------

    r"\b(?:posterior|anterior)\s+(?:horn|root)\b"
    r".{0,80}\bmedial meniscus\b"
    r".{0,50}\b(?:tear|tearing|rupture)\b",

    r"\bmedial meniscus\b"
    r".{0,100}\b(?:posterior horn|anterior horn|body|root)\b"
    r".{0,80}\b(?:tear|tearing|rupture)\b",

    # --------------------------------------------------------
    # Extrusion + definite tear/pathology
    # --------------------------------------------------------

    r"\bmedial meniscus\b"
    r".{0,100}\bextrusion\b"
    r".{0,100}\b(?:tear|tearing|rupture|maceration)\b",

    r"\b(?:tear|tearing|rupture|maceration)\b"
    r".{0,100}\bmedial meniscus\b"
    r".{0,100}\bextrusion\b",

    # --------------------------------------------------------
    # Simple phrases
    # --------------------------------------------------------

    r"\btear of (?:the )?medial meniscus\b",

    r"\btearing of (?:the )?medial meniscus\b",

    r"\bmedial meniscus tear\b",

    r"\bmedial meniscal tear\b",

    # ========================================================
    # SPANISH
    # ========================================================

    r"\bmenisco medial\b"
    r".{0,100}\b(?:rotura|ruptura|desgarro)\b",

    r"\b(?:rotura|ruptura|desgarro)\b"
    r".{0,80}\bmenisco medial\b",

    r"\bmenisco interno\b"
    r".{0,100}\b(?:rotura|ruptura|desgarro)\b",

    r"\b(?:rotura|ruptura)\b"
    r".{0,80}\bmenisco interno\b",

    # ========================================================
    # TURKISH
    # ========================================================

    r"\bmedial menisküs\b"
    r".{0,100}\b(?:yırtık|yırtığı|rüptür|kopma)\b",

    r"\b(?:yırtık|yırtığı|rüptür|kopma)\b"
    r".{0,100}\bmedial menisküs\b",

    # ========================================================
    # GERMAN
    # ========================================================

    r"\b(?:Innenmeniskus|Innenmenisk)\b"
    r".{0,100}\b(?:Riss|Ruptur)\b",

    r"\b(?:Riss|Ruptur)\b"
    r".{0,100}\b(?:Innenmeniskus|Innenmenisk)\b",

    # ========================================================
    # CROATIAN / BOSNIAN / SERBIAN
    # ========================================================

    r"\bmedijalni menisk\b"
    r".{0,100}\b(?:ruptura|rutura|puknuće|puknuća)\b",

    r"\b(?:ruptura|rutura|puknuće|puknuća)\b"
    r".{0,100}\bmedijalni menisk\b",

    # ========================================================
    # BULGARIAN
    # ========================================================

    r"\b(?:медиален|медиалния)\s+менискус\b"
    r".{0,100}\b(?:руптура|скъсване|разкъсване)\b",
]

print("Medial Meniscus V3 positive patterns loaded.")

# %%
# ============================================================
# CELL 27 — MEDIAL MENISCUS V2 DETECTOR
# ============================================================
MEDIAL_MENISCUS_UNCERTAIN_PATTERNS = [
    r"\bpossible\b.{0,80}\bmedial meniscus\b",
    r"\bsuspected\b.{0,80}\bmedial meniscus\b",
    r"\bcannot exclude\b.{0,80}\bmedial meniscus\b",
    r"\bmedial meniscus\b.{0,80}\bpossible\b",
    r"\bmedial meniscus\b.{0,80}\bsuspected\b",
    r"\bmay represent\b.{0,80}\bmedial meniscus\b",
]

def medial_meniscus_label_v2(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)
    sentences = split_sentences(report)

    uncertain_patterns = [
        r"\bpossible\b.{0,80}\bmedial meniscus\b",
        r"\bsuspected\b.{0,80}\bmedial meniscus\b",
        r"\bcannot exclude\b.{0,80}\bmedial meniscus\b",
        r"\bmedial meniscus\b.{0,80}\bpossible\b",
        r"\bmedial meniscus\b.{0,80}\bsuspected\b",
        r"\bmay represent\b.{0,80}\bmedial meniscus\b",
    ]

    for sentence in sentences:

        negative_match = first_match(
            sentence,
            MEDIAL_MENISCUS_NEGATIVE_PATTERNS
        )

        uncertain_match = first_match(
            sentence,
            uncertain_patterns
        )

        positive_match = first_match(
            sentence,
            MEDIAL_MENISCUS_POSITIVE_PATTERNS
        )

        if negative_match:

            return {
                "label": 0,
                "status": "NEGATIVE",
                "confidence": 0.98,
                "matched_sentence": sentence,
                "matched_pattern": negative_match,
                "reason": "Explicit medial meniscus negative statement"
            }

        if uncertain_match:

            return {
                "label": np.nan,
                "status": "UNCERTAIN",
                "confidence": 0.50,
                "matched_sentence": sentence,
                "matched_pattern": uncertain_match,
                "reason": "Uncertain medial meniscus statement"
            }

        if positive_match:

            return {
                "label": 1,
                "status": "POSITIVE",
                "confidence": 0.95,
                "matched_sentence": sentence,
                "matched_pattern": positive_match,
                "reason": "Positive medial meniscus abnormality detected"
            }

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial meniscus evidence detected"
    }


print("Medial Meniscus V2 detector created.")

# %%
# ============================================================
# MEDIAL MENISCUS NEGATIVE PATTERNS
# ============================================================

MEDIAL_MENISCUS_NEGATIVE_PATTERNS = [

    # Exact "not torn"
    r"\bmedial meniscus\b\s+(?:is|appears to be|seems to be)\s+not\s+torn\b",

    r"\bmedial meniscus\b[^.!?]{0,40}\bnot\s+torn\b",

    # Normal / intact
    r"\bmedial meniscus\b.{0,60}\b"
    r"(?:intact|normal|unremarkable|preserved)\b",

    r"\b(?:normal|intact|unremarkable|preserved)\b"
    r".{0,50}\bmedial meniscus\b",

    # Without / no tear
    r"\bmedial meniscus\b.{0,60}\bwithout\b.{0,40}\btear\b",

    r"\bmedial meniscus\b.{0,60}\bno\b.{0,40}\btear\b",

    r"\bno\b.{0,50}\btear\b.{0,50}\bmedial meniscus\b",

    r"\bwithout\b.{0,50}\btear\b.{0,50}\bmedial meniscus\b",

    # No frank tear
    r"\bmedial meniscus\b.{0,100}\bno\s+frank\s+tear\b",

    # Spanish
  # Spanish negative forms
r"\bmenisco medial\b.{0,100}\b"
r"(?:sin signos de desgarro|sin signos de rotura|"
r"sin signos de ruptura|sin criterios categóricos de rotura)\b",

r"\bmenisco interno\b.{0,100}\b"
r"(?:sin signos de desgarro|sin signos de rotura|"
r"sin signos de ruptura|sin criterios categóricos de rotura)\b",

    r"\bmenisco interno\b.{0,80}\b"
    r"(?:sin signos de rotura|sin signos de ruptura|"
    r"sin criterios.*rotura|normal|conservado)\b",

    # Turkish
    r"\bmedial menisküs\b.{0,60}\b"
    r"(?:normal|sağlam|intakt|yırtık yok)\b",

    # German
    r"\b(?:Innenmeniskus|Innenmenisk)\b.{0,60}\b"
    r"(?:intakt|normal|unauffällig)\b",

    # Croatian
    r"\bmedijalni menisk\b.{0,60}\b"
    r"(?:uredan|normalan|bez znakova rupture)\b",
]

print("Medial Meniscus negative patterns loaded.")

# %%
# ============================================================
# MEDIAL MENISCUS V4 — UNCERTAINTY TEST
# ============================================================

tests = [
    "Possible tear of the medial meniscus.",
    "Suspected medial meniscus tear.",
    "Possible medial meniscus tear.",
    "The finding may represent a tear of the medial meniscus.",
    "Cannot exclude tear of the medial meniscus."
]

for text in tests:

    result = medial_meniscus_label_v2(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected: UNCERTAIN")
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])
    print("Reason:", result["reason"])

    assert result["status"] == "UNCERTAIN"

print("\n" + "=" * 100)
print("✅ ALL UNCERTAINTY TESTS PASSED")
print("=" * 100)

# %%
# ============================================================
# MEDIAL MENISCUS LABEL FUNCTION — V5
# ============================================================

def medial_meniscus_label_v5(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)
    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        negative_match = first_match(
            sentence,
            MEDIAL_MENISCUS_NEGATIVE_PATTERNS
        )

        uncertain_match = first_match(
            sentence,
            MEDIAL_MENISCUS_UNCERTAIN_PATTERNS
        )

        positive_match = first_match(
            sentence,
            MEDIAL_MENISCUS_POSITIVE_PATTERNS
        )

        if negative_match:
            negative_hits.append(
                (sentence, negative_match)
            )

        elif uncertain_match:
            uncertain_hits.append(
                (sentence, uncertain_match)
            )

        elif positive_match:
            positive_hits.append(
                (sentence, positive_match)
            )

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit medial meniscus negative statement"
        }

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain medial meniscus statement"
        }

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Positive medial meniscus abnormality detected"
        }

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial meniscus evidence detected"
    }


print("Medial Meniscus V5 detector created successfully.")

    
print("Medial Meniscus V5 detector created successfully.")

# %%
# ============================================================
# MEDIAL MENISCUS V3 UNIT TESTS
# ============================================================

medial_meniscus_tests_v3 = {

    # POSITIVE
    "Medial meniscus tear.": "POSITIVE",
    "Complex tear of the medial meniscus.": "POSITIVE",
    "Radial tear of the posterior root of the medial meniscus.": "POSITIVE",
    "Medial meniscus: Complete radial tear.": "POSITIVE",
    "The medial meniscus has a horizontal tear.": "POSITIVE",
    "Medial meniscal tear with extrusion.": "POSITIVE",
    "Complete tearing of the body and posterior horn of the medial meniscus.": "POSITIVE",
    "Tear of the posterior horn of medial meniscus.": "POSITIVE",

    # NEGATIVE
    "Medial meniscus is intact.": "NEGATIVE",
    "Medial meniscus is normal.": "NEGATIVE",
    "Medial meniscus is not torn.": "NEGATIVE",
    "No tear of the medial meniscus.": "NEGATIVE",
    "There is no tear of the medial meniscus.": "NEGATIVE",
    "Medial meniscus: No frank tear.": "NEGATIVE",
    "Normal medial meniscus.": "NEGATIVE",

    # UNCERTAIN
    "Possible tear of the medial meniscus.": "UNCERTAIN",
    "Suspected medial meniscus tear.": "UNCERTAIN",
    "Possible medial meniscus tear.": "UNCERTAIN",
    "The finding may represent a tear of the medial meniscus.": "UNCERTAIN",
    "Cannot exclude tear of the medial meniscus.": "UNCERTAIN",
}


passed = 0
failed = 0

for text, expected in medial_meniscus_tests_v3.items():

    result = medial_meniscus_label_v2(text)
    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:
        passed += 1
        print("✅ PASS")
    else:
        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL MENISCUS V3 UNIT TEST SUMMARY")
print("=" * 100)

print("Total tests :", len(medial_meniscus_tests_v3))
print("Passed      :", passed)
print("Failed      :", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL MENISCUS V3 TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED")

# %%
# ============================================================
# MEDIAL MENISCUS V4 — FULL UNIT TEST
# ============================================================

medial_meniscus_tests_v4 = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Medial meniscus tear.": "POSITIVE",

    "Complex tear of the medial meniscus.": "POSITIVE",

    "Radial tear of the posterior root of the medial meniscus.": "POSITIVE",

    "Medial meniscus: Complete radial tear.": "POSITIVE",

    "The medial meniscus has a horizontal tear.": "POSITIVE",

    "Medial meniscal tear with extrusion.": "POSITIVE",

    "Complete tearing of the body and posterior horn of the medial meniscus.": "POSITIVE",

    "Tear of the posterior horn of medial meniscus.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "Medial meniscus is intact.": "NEGATIVE",

    "Medial meniscus is normal.": "NEGATIVE",

    "Medial meniscus is not torn.": "NEGATIVE",

    "No tear of the medial meniscus.": "NEGATIVE",

    "There is no tear of the medial meniscus.": "NEGATIVE",

    "Medial meniscus: No frank tear.": "NEGATIVE",

    "Normal medial meniscus.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible tear of the medial meniscus.": "UNCERTAIN",

    "Suspected medial meniscus tear.": "UNCERTAIN",

    "Possible medial meniscus tear.": "UNCERTAIN",

    "The finding may represent a tear of the medial meniscus.": "UNCERTAIN",

    "Cannot exclude tear of the medial meniscus.": "UNCERTAIN",
}


passed = 0
failed = 0

for text, expected in medial_meniscus_tests_v4.items():

    result = medial_meniscus_label_v2(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL MENISCUS V4 UNIT TEST SUMMARY")
print("=" * 100)

print("Total tests :", len(medial_meniscus_tests_v4))
print("Passed      :", passed)
print("Failed      :", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL MENISCUS V4 TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION YET")

# %%
# ============================================================
# CELL 32 — APPLY MEDIAL MENISCUS V4 TO GOLD SET
# ============================================================

mm_results = gold["Report"].apply(
    medial_meniscus_label_v2
)

mm_results_df = pd.DataFrame(
    mm_results.tolist(),
    index=gold.index
)

mm_results_df = mm_results_df.rename(columns={
    "label": "Medial_Meniscus_weak_v4",
    "status": "Medial_Meniscus_status_v4",
    "confidence": "Medial_Meniscus_confidence_v4",
    "matched_sentence": "Medial_Meniscus_sentence_v4",
    "matched_pattern": "Medial_Meniscus_match_v4",
    "reason": "Medial_Meniscus_reason_v4"
})

gold_mm_v4 = pd.concat(
    [
        gold,
        mm_results_df
    ],
    axis=1
)

print("Medial Meniscus V4 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_mm_v4["Medial_Meniscus_weak_v4"]
    .value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_mm_v4["Medial_Meniscus_status_v4"]
    .value_counts(dropna=False)
)


# %%
# ============================================================
# CELL 33 — MEDIAL MENISCUS V4 METRICS
# ============================================================

evaluated = gold_mm_v4[
    gold_mm_v4["Medial_Meniscus_weak_v4"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial Meniscus"] == 1) &
    (evaluated["Medial_Meniscus_weak_v4"] == 1)
).sum()

TN = (
    (evaluated["Medial Meniscus"] == 0) &
    (evaluated["Medial_Meniscus_weak_v4"] == 0)
).sum()

FP = (
    (evaluated["Medial Meniscus"] == 0) &
    (evaluated["Medial_Meniscus_weak_v4"] == 1)
).sum()

FN = (
    (evaluated["Medial Meniscus"] == 1) &
    (evaluated["Medial_Meniscus_weak_v4"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_mm_v4)

print("=" * 80)
print("MEDIAL MENISCUS V4 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_mm_v4) - len(evaluated)
)

# %%
# ============================================================
# CELL 34 — MEDIAL MENISCUS FP/FN
# ============================================================

fp = gold_mm_v4[
    (gold_mm_v4["Medial Meniscus"] == 0) &
    (gold_mm_v4["Medial_Meniscus_weak_v4"] == 1)
]

fn = gold_mm_v4[
    (gold_mm_v4["Medial Meniscus"] == 1) &
    (gold_mm_v4["Medial_Meniscus_weak_v4"] == 0)
]

print("=" * 100)
print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))
print("=" * 100)

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Medial Meniscus"])
    print("Weak:", row["Medial_Meniscus_weak_v4"])
    print("Status:", row["Medial_Meniscus_status_v4"])
    print("Matched:", row["Medial_Meniscus_match_v4"])

    print("\nMatched sentence:")
    print(row["Medial_Meniscus_sentence_v4"])

    print("\nReason:")
    print(row["Medial_Meniscus_reason_v4"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# Rule-out / differential diagnosis
r"\br/o\b.{0,120}\b(?:medial meniscus|medial meniscal)\b",

r"\brule out\b.{0,120}\b(?:medial meniscus|medial meniscal)\b",

r"\b(?:r/o|rule out)\b.{0,120}\b"
r"(?:tear|rupture)\b.{0,100}\b"
r"(?:medial meniscus|medial meniscal)\b",

r"\b(?:r/o|rule out)\b.{0,120}\b"
r"(?:medial meniscus|medial meniscal)\b.{0,80}\b"
r"(?:tear|rupture)\b",

# %%
test = (
    "R/O Ramp lesion or peripheral longitudinal tear "
    "of posterior horn of medial meniscus."
)

result = medial_meniscus_label_v2(test)

print(result)

# %%
# ============================================================
# DEBUG R/O UNCERTAINTY MATCH
# ============================================================

text = (
    "R/O Ramp lesion or peripheral longitudinal tear "
    "of posterior horn of medial meniscus."
)

print("TEXT:")
print(text)

print("\nUNCERTAINTY PATTERN RESULTS:")
for i, pattern in enumerate(MEDIAL_MENISCUS_UNCERTAIN_PATTERNS):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.UNICODE
    )

    if match:
        print(f"\n✅ MATCH FOUND — Pattern {i}")
        print("Pattern:", pattern)
        print("Matched:", match.group(0))

print("\nPOSITIVE PATTERN RESULTS:")
for i, pattern in enumerate(MEDIAL_MENISCUS_POSITIVE_PATTERNS):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.UNICODE
    )

    if match:
        print(f"\n✅ MATCH FOUND — Pattern {i}")
        print("Pattern:", pattern)
        print("Matched:", match.group(0))

# %%
# ============================================================
# MEDIAL MENISCUS LABEL FUNCTION — V6
# ============================================================

def medial_meniscus_label_v6(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ====================================================
        # 1. NEGATIVE
        # ====================================================

        negative_match = first_match(
            sentence,
            MEDIAL_MENISCUS_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            # Explicit negative means don't treat other
            # positive wording in this sentence as positive.
            continue

        # ====================================================
        # 2. EXPLICIT UNCERTAINTY
        # ====================================================

        # First use the configured uncertainty patterns.
        uncertain_match = first_match(
            sentence,
            MEDIAL_MENISCUS_UNCERTAIN_PATTERNS
        )

        # Then explicitly catch R/O style wording.
        ro_match = re.search(
            r"\b(?:r/o|rule\s+out)\b",
            sentence,
            flags=re.IGNORECASE
        )

        # ====================================================
        # If R/O is present AND the sentence refers to the
        # medial meniscus, classify the sentence as uncertain.
        # ====================================================

        if ro_match and re.search(
            r"\b(?:medial meniscus|medial meniscal)\b",
            sentence,
            flags=re.IGNORECASE
        ):

            uncertain_hits.append(
                (
                    sentence,
                    ro_match.group(0)
                )
            )

            continue

        # Normal uncertainty handling
        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ====================================================
        # 3. POSITIVE
        # ====================================================

        positive_match = first_match(
            sentence,
            MEDIAL_MENISCUS_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # ========================================================
    # REPORT-LEVEL DECISION
    # ========================================================

    # A definitive positive sentence wins over uncertainty
    # elsewhere in the report.

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive medial meniscus abnormality detected"
        }

    # Explicit negative

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit medial meniscus negative statement"
        }

    # Uncertain

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain medial meniscus statement"
        }

    # Nothing detected

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial meniscus evidence detected"
    }


print("Medial Meniscus V6 detector created successfully.")

# %%
# ============================================================
# TEST R/O
# ============================================================

text = (
    "R/O Ramp lesion or peripheral longitudinal tear "
    "of posterior horn of medial meniscus."
)

result = medial_meniscus_label_v6(text)

print(result)

assert result["status"] == "UNCERTAIN"
assert pd.isna(result["label"])

print("\n✅ R/O TEST PASSED")

# %%
# ============================================================
# MEDIAL MENISCUS V6 CRITICAL TESTS
# ============================================================

critical_tests = {

    "R/O Ramp lesion or peripheral longitudinal tear of posterior horn of medial meniscus.":
        "UNCERTAIN",

    "Possible tear of the medial meniscus.":
        "UNCERTAIN",

    "Suspected medial meniscus tear.":
        "UNCERTAIN",

    "Medial meniscus is not torn.":
        "NEGATIVE",

    "Medial meniscus is intact.":
        "NEGATIVE",

    "Complex tear of the medial meniscus.":
        "POSITIVE",

    "Complete tearing of the posterior horn of the medial meniscus.":
        "POSITIVE",
}


for text, expected in critical_tests.items():

    result = medial_meniscus_label_v6(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])

    assert result["status"] == expected


print("\n" + "=" * 100)
print("✅ ALL MEDIAL MENISCUS V6 CRITICAL TESTS PASSED")
print("=" * 100)

# %%
# ============================================================
# MEDIAL MENISCUS V6 — GOLD EVALUATION
# ============================================================

mm_v6_results = gold["Report"].apply(
    medial_meniscus_label_v6
)

mm_v6_results_df = pd.DataFrame(
    mm_v6_results.tolist(),
    index=gold.index
)

mm_v6_results_df = mm_v6_results_df.rename(columns={
    "label": "Medial_Meniscus_weak_v6",
    "status": "Medial_Meniscus_status_v6",
    "confidence": "Medial_Meniscus_confidence_v6",
    "matched_sentence": "Medial_Meniscus_sentence_v6",
    "matched_pattern": "Medial_Meniscus_match_v6",
    "reason": "Medial_Meniscus_reason_v6"
})

gold_mm_v6 = pd.concat(
    [gold, mm_v6_results_df],
    axis=1
)

evaluated = gold_mm_v6[
    gold_mm_v6["Medial_Meniscus_weak_v6"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial Meniscus"] == 1) &
    (evaluated["Medial_Meniscus_weak_v6"] == 1)
).sum()

TN = (
    (evaluated["Medial Meniscus"] == 0) &
    (evaluated["Medial_Meniscus_weak_v6"] == 0)
).sum()

FP = (
    (evaluated["Medial Meniscus"] == 0) &
    (evaluated["Medial_Meniscus_weak_v6"] == 1)
).sum()

FN = (
    (evaluated["Medial Meniscus"] == 1) &
    (evaluated["Medial_Meniscus_weak_v6"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_mm_v6)

print("=" * 80)
print("MEDIAL MENISCUS V6 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_mm_v6) - len(evaluated)
)

# %%
# ============================================================
# LATERAL MENISCUS UNCERTAINTY PATTERNS — V2
# ============================================================

LATERAL_MENISCUS_UNCERTAIN_PATTERNS = [

    # --------------------------------------------------------
    # English uncertainty
    # --------------------------------------------------------

    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"suspicious|questionable)\b"
    r".{0,100}\blateral meniscus\b",

    r"\blateral meniscus\b.{0,100}\b"
    r"(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"suspicious|questionable)\b",

    r"\blateral meniscus\b.{0,100}\b"
    r"(?:could represent|may represent)\b",

    # "R/O"
    r"\br/o\b.{0,120}\b(?:lateral meniscus|lateral meniscal)\b",

    r"\brule\s+out\b.{0,120}\b(?:lateral meniscus|lateral meniscal)\b",

    r"\b(?:r/o|rule\s+out)\b.{0,120}\b"
    r"(?:tear|rupture)\b.{0,100}\b"
    r"(?:lateral meniscus|lateral meniscal)\b",

    r"\b(?:r/o|rule\s+out)\b.{0,120}\b"
    r"(?:lateral meniscus|lateral meniscal)\b.{0,80}\b"
    r"(?:tear|rupture)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,100}\bmenisco lateral\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\b(?:olası|şüpheli|muhtemel|şüpheli)\b"
    r".{0,100}\blateral menisküs\b",

    # --------------------------------------------------------
    # Croatian
    # --------------------------------------------------------

    r"\b(?:moguće|vjerojatno|sumnja)\b"
    r".{0,100}\blateralni menisk\b",
]

print("Lateral Meniscus uncertainty patterns updated.")

# %%
# ============================================================
# LATERAL MENISCUS NEGATIVE PATTERNS — V2
# ============================================================

LATERAL_MENISCUS_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit "not torn"
    # --------------------------------------------------------

    r"\blateral meniscus\b\s+"
    r"(?:is|appears to be|seems to be)\s+not\s+torn\b",

    r"\blateral meniscus\b[^.!?]{0,50}\bnot\s+torn\b",

    # --------------------------------------------------------
    # Explicit "no tear"
    # --------------------------------------------------------

    r"\blateral meniscus\b\s*[:\-]\s*"
    r"(?:no|without)\s+"
    r"(?:evidence of\s+)?"
    r"(?:tear|rupture|injury|abnormality)\b",

    r"\blateral meniscal\b\s*[:\-]\s*"
    r"(?:no|without)\s+"
    r"(?:evidence of\s+)?"
    r"(?:tear|rupture|injury|abnormality)\b",

    r"\bno\b.{0,50}\b"
    r"(?:lateral meniscus|lateral meniscal)\b"
    r".{0,50}\b(?:tear|rupture|injury)\b",

    r"\b(?:lateral meniscus|lateral meniscal)\b"
    r".{0,50}\bno\b.{0,40}\b"
    r"(?:tear|rupture|injury)\b",

    # --------------------------------------------------------
    # Explicit "without tear"
    # --------------------------------------------------------

    r"\b(?:lateral meniscus|lateral meniscal)\b"
    r".{0,60}\bwithout\b.{0,40}\btear\b",

    r"\bwithout\b.{0,50}\btear\b.{0,50}\b"
    r"(?:lateral meniscus|lateral meniscal)\b",

    # --------------------------------------------------------
    # "No frank tear"
    # --------------------------------------------------------

    r"\blateral meniscus\b.{0,100}\bno\s+frank\s+tear\b",

    # --------------------------------------------------------
    # Normal / intact
    # --------------------------------------------------------

    r"\blateral meniscus\b.{0,60}\b"
    r"(?:intact|normal|unremarkable|preserved)\b",

    r"\b(?:intact|normal|unremarkable|preserved)\b"
    r".{0,60}\blateral meniscus\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\bmenisco lateral\b.{0,100}\b"
    r"(?:sin signos de desgarro|sin signos de rotura|"
    r"sin signos de ruptura|sin criterios.*rotura|"
    r"normal|conservado)\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\blateral menisküs\b.{0,80}\b"
    r"(?:normal|sağlam|intakt|yırtık yok)\b",

    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    r"\b(?:Außenmeniskus|Außenmenisk)\b.{0,80}\b"
    r"(?:intakt|normal|unauffällig)\b",

    # --------------------------------------------------------
    # Croatian
    # --------------------------------------------------------

    r"\blateralni menisk\b.{0,80}\b"
    r"(?:uredan|normalan|bez znakova rupture)\b",
]

print("Lateral Meniscus negative patterns updated.")

# %%
# ============================================================
# LATERAL MENISCUS V2 — ADDITIONAL NEGATIVE PATTERNS
# ============================================================

LATERAL_MENISCUS_NEGATIVE_PATTERNS += [

    # "No evidence of tear noted in either medial or lateral meniscus"
    r"\bno evidence of\b"
    r".{0,50}\btear\b"
    r".{0,80}\b(?:lateral meniscus|lateral meniscal)\b",

    # "No evidence of tear ... lateral meniscus"
    r"\bno evidence of tear\b"
    r".{0,100}\b(?:lateral meniscus|lateral meniscal)\b",

    # "No tear ... lateral meniscus"
    r"\bno tear\b"
    r".{0,80}\b(?:lateral meniscus|lateral meniscal)\b",

    # "No evidence ... either medial or lateral meniscus"
    r"\bno evidence\b"
    r".{0,100}\b(?:medial|lateral)\s+meniscus\b",

    # Generic negative pattern
    r"\bno\b"
    r".{0,50}\b(?:tear|abnormality|injury|rupture)\b"
    r".{0,100}\b(?:lateral meniscus|lateral meniscal)\b",
]

print("Additional Lateral Meniscus negative patterns added.")

# %%
# ============================================================
# LATERAL MENISCUS V2 — TEST NEGATION
# ============================================================

# text = (
#     "No evidence of tear noted in either medial or lateral "
#     "meniscus in this study."
# )

# # result = lateral_meniscus_label_v2(text)

# print("TEXT:")
# print(text)

# print("\nRESULT:")
# print(result)

# assert result["status"] == "NEGATIVE"
# assert result["label"] == 0

# print("\n✅ NEGATION TEST PASSED")

# %%
# ============================================================
# LATERAL MENISCUS V2 — TEST NORMAL CASE
# ============================================================

# text = "Bulgular: Lateral menisküs normal."

# # result = lateral_meniscus_label_v2(text)

# print("TEXT:")
# print(text)

# print("\nRESULT:")
# print(result)

# assert result["status"] == "NEGATIVE"
# assert result["label"] == 0

# print("\n✅ NORMAL CASE TEST PASSED")

# %%
# ============================================================
#
LATERAL_MENISCUS_POSITIVE_PATTERNS = [
    r"\blateral meniscus\b.{0,80}\b(?:tear|torn|rupture|ruptured|maceration|macerated|fragment)\b",
    r"\blateral meniscal\b.{0,80}\b(?:tear|torn|rupture|ruptured)\b",
    r"\b(?:tear|torn|rupture|ruptured)\b.{0,80}\blateral meniscus\b",
    r"\b(?:complex|radial|horizontal|vertical|root)\s+tear\b.{0,80}\blateral meniscus\b",
    r"\blateral meniscus\b.{0,80}\b(?:injury|abnormality)\b",
]

def lateral_meniscus_label_v2(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # Check NEGATIVE first
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            LATERAL_MENISCUS_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # Check UNCERTAINTY before POSITIVE
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            LATERAL_MENISCUS_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # Finally check POSITIVE
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            LATERAL_MENISCUS_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # Definitive positive
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive lateral meniscus abnormality detected"
        }

    # --------------------------------------------------------
    # Explicit negative
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit lateral meniscus negative statement"
        }

    # --------------------------------------------------------
    # Uncertain
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain lateral meniscus statement"
        }

    # --------------------------------------------------------
    # Not mentioned
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No lateral meniscus evidence detected"
    }


print("Lateral Meniscus V2 detector created.")

# %%
# ============================================================
# LATERAL MENISCUS V2 — CRITICAL TESTS
# ============================================================

critical_tests = {

    # Negation bug from gold data
    "No lateral meniscal tear.": "NEGATIVE",

    "Lateral meniscus: No tear.": "NEGATIVE",

    "Lateral meniscus is not torn.": "NEGATIVE",

    # Uncertainty bug from gold data
    "The lateral compartment, there is suspicious small "
    "longitudinal vertical tear at the periphery posterior "
    "horn of the lateral meniscus.": "UNCERTAIN",

    "Possible lateral meniscus tear.": "UNCERTAIN",

    "Suspected lateral meniscus tear.": "UNCERTAIN",

    "R/O tear of the lateral meniscus.": "UNCERTAIN",

    # Clear positives
    "Lateral meniscus tear.": "POSITIVE",

    "Complex tear of the lateral meniscus.": "POSITIVE",

    "Tear of the anterior horn of lateral meniscus.": "POSITIVE",

    # Anatomical separation
    "Medial meniscus is torn. Lateral meniscus is intact.": "NEGATIVE",

    "Medial meniscus is intact. Lateral meniscus is torn.": "POSITIVE",
}


passed = 0
failed = 0

for text, expected in critical_tests.items():

    # result = lateral_meniscus_label_v2(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:
        passed += 1
        print("✅ PASS")
    else:
        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("LATERAL MENISCUS V2 CRITICAL TEST SUMMARY")
print("=" * 100)

print("Total:", len(critical_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ ALL CRITICAL TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED")

# %%
# ============================================================
# CELL 37 — LATERAL MENISCUS V1 UNIT TESTS
# ============================================================

lateral_meniscus_tests = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Lateral meniscus tear.": "POSITIVE",

    "Complex tear of the lateral meniscus.": "POSITIVE",

    "Radial tear of the posterior root of the lateral meniscus.": "POSITIVE",

    "Lateral meniscus: Complete radial tear.": "POSITIVE",

    "The lateral meniscus has a horizontal tear.": "POSITIVE",

    "Lateral meniscal tear with extrusion.": "POSITIVE",

    "Complete tearing of the posterior horn of the lateral meniscus.": "POSITIVE",

    "Tear of the anterior horn of lateral meniscus.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "Lateral meniscus is intact.": "NEGATIVE",

    "Lateral meniscus is normal.": "NEGATIVE",

    "Lateral meniscus is not torn.": "NEGATIVE",

    "No tear of the lateral meniscus.": "NEGATIVE",

    "There is no tear of the lateral meniscus.": "NEGATIVE",

    "Lateral meniscus: No frank tear.": "NEGATIVE",

    "Normal lateral meniscus.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible tear of the lateral meniscus.": "UNCERTAIN",

    "Suspected lateral meniscus tear.": "UNCERTAIN",

    "Possible lateral meniscus tear.": "UNCERTAIN",

    "The finding may represent a tear of the lateral meniscus.": "UNCERTAIN",

    "Cannot exclude tear of the lateral meniscus.": "UNCERTAIN",

    "R/O tear of the lateral meniscus.": "UNCERTAIN",

    # --------------------------------------------------------
    # ANATOMICAL CONTEXT
    # --------------------------------------------------------

    "Medial meniscus is intact. Lateral meniscus is torn.": "POSITIVE",

    "Lateral meniscus is intact. Medial meniscus is torn.": "NEGATIVE",
}


passed = 0
failed = 0

for text, expected in lateral_meniscus_tests.items():

    result = lateral_meniscus_label_v2(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("LATERAL MENISCUS V1 UNIT TEST SUMMARY")
print("=" * 100)

print("Total tests :", len(lateral_meniscus_tests))
print("Passed      :", passed)
print("Failed      :", failed)

if failed == 0:
    print("\n✅ ALL LATERAL MENISCUS V1 TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION")

# %%
# ============================================================
# CELL 41 — APPLY LATERAL MENISCUS V2 TO GOLD SET
# ============================================================

lateral_v2_results = gold["Report"].apply(
    lateral_meniscus_label_v2
)

lateral_v2_results_df = pd.DataFrame(
    lateral_v2_results.tolist(),
    index=gold.index
)

lateral_v2_results_df = lateral_v2_results_df.rename(columns={
    "label": "Lateral_Meniscus_weak_v2",
    "status": "Lateral_Meniscus_status_v2",
    "confidence": "Lateral_Meniscus_confidence_v2",
    "matched_sentence": "Lateral_Meniscus_sentence_v2",
    "matched_pattern": "Lateral_Meniscus_match_v2",
    "reason": "Lateral_Meniscus_reason_v2"
})

gold_lateral_v2 = pd.concat(
    [gold, lateral_v2_results_df],
    axis=1
)

print("Lateral Meniscus V2 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_lateral_v2[
        "Lateral_Meniscus_weak_v2"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_lateral_v2[
        "Lateral_Meniscus_status_v2"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 42 — LATERAL MENISCUS V2 GOLD METRICS
# ============================================================

evaluated = gold_lateral_v2[
    gold_lateral_v2["Lateral_Meniscus_weak_v2"].isin([0, 1])
].copy()

TP = (
    (evaluated["Lateral Meniscus"] == 1) &
    (evaluated["Lateral_Meniscus_weak_v2"] == 1)
).sum()

TN = (
    (evaluated["Lateral Meniscus"] == 0) &
    (evaluated["Lateral_Meniscus_weak_v2"] == 0)
).sum()

FP = (
    (evaluated["Lateral Meniscus"] == 0) &
    (evaluated["Lateral_Meniscus_weak_v2"] == 1)
).sum()

FN = (
    (evaluated["Lateral Meniscus"] == 1) &
    (evaluated["Lateral_Meniscus_weak_v2"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_lateral_v2)

print("=" * 80)
print("LATERAL MENISCUS V2 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_lateral_v2) - len(evaluated)
)

# %%
# ============================================================
# CELL 43 — LATERAL MENISCUS V2 ERRORS
# ============================================================

fp = gold_lateral_v2[
    (gold_lateral_v2["Lateral Meniscus"] == 0) &
    (gold_lateral_v2["Lateral_Meniscus_weak_v2"] == 1)
]

fn = gold_lateral_v2[
    (gold_lateral_v2["Lateral Meniscus"] == 1) &
    (gold_lateral_v2["Lateral_Meniscus_weak_v2"] == 0)
]

print("=" * 100)
print("LATERAL MENISCUS V2 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Lateral Meniscus"])
    print("Weak:", row["Lateral_Meniscus_weak_v2"])
    print("Status:", row["Lateral_Meniscus_status_v2"])

    print("\nMatched sentence:")
    print(row["Lateral_Meniscus_sentence_v2"])

    print("\nMatched pattern:")
    print(row["Lateral_Meniscus_match_v2"])

    print("\nReason:")
    print(row["Lateral_Meniscus_reason_v2"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 44 — EFFUSION V1 PATTERNS
# ============================================================

EFFUSION_POSITIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit joint/knee effusion
    # --------------------------------------------------------

    r"\bjoint effusion\b",

    r"\bknee effusion\b",

    r"\bintra[- ]articular effusion\b",

    r"\beffusion of the knee\b",

    # --------------------------------------------------------
    # Effusion with size/severity
    # --------------------------------------------------------

    r"\b(?:small|mild|moderate|large|massive|trace|minimal|"
    r"moderate[- ]to[- ]large|small[- ]to[- ]moderate)\b"
    r".{0,20}\beffusion\b",

    # --------------------------------------------------------
    # Common short wording
    # --------------------------------------------------------

    r"\bsome (?:amount of )?right knee effusion\b",

    r"\bsome (?:amount of )?left knee effusion\b",

    r"\bsome joint effusion\b",

    r"\bminimal amount of .*?effusion\b",

    # --------------------------------------------------------
    # Fluid wording when clearly referring to joint
    # --------------------------------------------------------

    r"\b(?:joint|knee)\b.{0,30}\bfluid\b",

    r"\bfluid\b.{0,30}\b(?:joint|knee)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\bderrame articular\b",

    r"\bderrame de la rodilla\b",

    r"\bderrame\b.{0,30}\b(?:articular|rodilla)\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\bdiz ekleminde\b.{0,50}\bsıvı artışı\b",

    r"\beklem aralığında\b.{0,50}\bsıvı\b",

    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    r"\bKniegelenkerguss\b",

    r"\bGelenkerguss\b",

    r"\bErguss\b",

    # --------------------------------------------------------
    # Croatian / Bosnian / Serbian
    # --------------------------------------------------------

    r"\bzglobni izljev\b",

    r"\bizljev\b",
]


EFFUSION_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit English negation
    # --------------------------------------------------------

    r"\bno joint effusion\b",

    r"\bno knee effusion\b",

    r"\bno evidence of effusion\b",

    r"\bwithout effusion\b",

    r"\bwithout joint effusion\b",

    r"\bno significant effusion\b",

    r"\bno significant joint effusion\b",

    # --------------------------------------------------------
    # Colon style
    # --------------------------------------------------------

    r"\beffusion\s*[:\-]\s*(?:none|absent)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\bno hay\b.{0,30}\bderrame\b",

    r"\bsin\b.{0,30}\bderrame\b",

    r"\bderrame\b.{0,30}\b(?:ausente|ausencia)\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\b(?:eklemde|dizde)\b.{0,50}\bsıvı artışı yok\b",

    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    r"\bkein\b.{0,30}\b(?:Gelenkerguss|Erguss)\b",

    r"\b(?:Gelenkerguss|Erguss)\b.{0,30}\bkein\b",
]


EFFUSION_UNCERTAIN_PATTERNS = [

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"questionable|may represent|could represent)\b"
    r".{0,50}\beffusion\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,50}\bderrame\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\b(?:olası|şüpheli|muhtemel)\b"
    r".{0,50}\bsıvı artışı\b",
]


print("Effusion V1 patterns loaded.")

# %%
# ============================================================
# CELL 45 — EFFUSION V1 DETECTOR
# ============================================================

def effusion_label_v1(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # NEGATIVE FIRST
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            EFFUSION_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # UNCERTAIN SECOND
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            EFFUSION_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # POSITIVE LAST
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            EFFUSION_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # DEFINITIVE POSITIVE
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive joint effusion detected"
        }

    # --------------------------------------------------------
    # EXPLICIT NEGATIVE
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of joint effusion"
        }

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain effusion statement"
        }

    # --------------------------------------------------------
    # NOT MENTIONED
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No effusion evidence detected"
    }


print("Effusion V1 detector created.")

# %%
# ============================================================
# CELL 46 — EFFUSION V1 UNIT TESTS
# ============================================================

effusion_tests = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Joint effusion.": "POSITIVE",

    "Small joint effusion.": "POSITIVE",

    "Moderate joint effusion.": "POSITIVE",

    "Large joint effusion.": "POSITIVE",

    "There is knee effusion.": "POSITIVE",

    "Some joint effusion is present.": "POSITIVE",

    "Moderate right knee effusion with hemarthrosis.": "POSITIVE",

    "Levele? derrame articular.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "No joint effusion.": "NEGATIVE",

    "No knee effusion.": "NEGATIVE",

    "There is no significant joint effusion.": "NEGATIVE",

    "Without joint effusion.": "NEGATIVE",

    "Effusion: None.": "NEGATIVE",

    "No evidence of effusion.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible joint effusion.": "UNCERTAIN",

    "Possible effusion.": "UNCERTAIN",

    "Suspected joint effusion.": "UNCERTAIN",

    # --------------------------------------------------------
    # MULTILINGUAL
    # --------------------------------------------------------

    "Derrame articular leve.": "POSITIVE",

    "Bulgular: Lateral menisküs normal.": "NOT_MENTIONED",
}


passed = 0
failed = 0

for text, expected in effusion_tests.items():

    result = effusion_label_v1(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("EFFUSION V1 UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(effusion_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL EFFUSION V1 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — FIX BEFORE GOLD EVALUATION")

# %%
# ============================================================
# CELL 47 — EFFUSION V1 GOLD EVALUATION
# ============================================================

effusion_results = gold["Report"].apply(
    effusion_label_v1
)

effusion_results_df = pd.DataFrame(
    effusion_results.tolist(),
    index=gold.index
)

effusion_results_df = effusion_results_df.rename(columns={
    "label": "Effusion_weak_v1",
    "status": "Effusion_status_v1",
    "confidence": "Effusion_confidence_v1",
    "matched_sentence": "Effusion_sentence_v1",
    "matched_pattern": "Effusion_match_v1",
    "reason": "Effusion_reason_v1"
})

gold_effusion_v1 = pd.concat(
    [gold, effusion_results_df],
    axis=1
)

print("Effusion V1 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_effusion_v1[
        "Effusion_weak_v1"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_effusion_v1[
        "Effusion_status_v1"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 48 — EFFUSION V1 METRICS
# ============================================================

evaluated = gold_effusion_v1[
    gold_effusion_v1["Effusion_weak_v1"].isin([0, 1])
].copy()

TP = (
    (evaluated["Effusion"] == 1) &
    (evaluated["Effusion_weak_v1"] == 1)
).sum()

TN = (
    (evaluated["Effusion"] == 0) &
    (evaluated["Effusion_weak_v1"] == 0)
).sum()

FP = (
    (evaluated["Effusion"] == 0) &
    (evaluated["Effusion_weak_v1"] == 1)
).sum()

FN = (
    (evaluated["Effusion"] == 1) &
    (evaluated["Effusion_weak_v1"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_effusion_v1)

print("=" * 80)
print("EFFUSION V1 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_effusion_v1) - len(evaluated)
)

# %%
# ============================================================
# CELL 49 — EFFUSION V1 FP / FN
# ============================================================

fp = gold_effusion_v1[
    (gold_effusion_v1["Effusion"] == 0) &
    (gold_effusion_v1["Effusion_weak_v1"] == 1)
]

fn = gold_effusion_v1[
    (gold_effusion_v1["Effusion"] == 1) &
    (gold_effusion_v1["Effusion_weak_v1"] == 0)
]

print("=" * 100)
print("EFFUSION V1 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Effusion"])
    print("Weak:", row["Effusion_weak_v1"])
    print("Status:", row["Effusion_status_v1"])

    print("\nMatched sentence:")
    print(row["Effusion_sentence_v1"])

    print("\nMatched pattern:")
    print(row["Effusion_match_v1"])

    print("\nReason:")
    print(row["Effusion_reason_v1"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# EFFUSION V2 — ADDITIONAL NEGATIVE PATTERNS
# ============================================================

EFFUSION_NEGATIVE_PATTERNS += [

    # "No evidence of knee effusion."
    r"\bno evidence of\b"
    r".{0,30}\bknee effusion\b",

    # "No significant knee effusion."
    r"\bno significant\b"
    r".{0,20}\bknee effusion\b",

    # General form
    r"\bno\b"
    r".{0,30}\b(?:knee|joint)\s+effusion\b",
]

print("Effusion V2 negative patterns added.")

# %%
# ============================================================
# EFFUSION V2 — NEGATION TESTS
# ============================================================

tests = {

    "No evidence of knee effusion.": "NEGATIVE",

    "No significant knee effusion.": "NEGATIVE",

    "No joint effusion.": "NEGATIVE",

    "No knee effusion.": "NEGATIVE",

    "Small joint effusion.": "POSITIVE",

    "Moderate joint effusion.": "POSITIVE",

    "Large joint effusion.": "POSITIVE",
}


for text, expected in tests.items():

    result = effusion_label_v1(text)

    print("\n" + "-" * 90)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])

    assert result["status"] == expected


print("\n" + "=" * 90)
print("✅ ALL EFFUSION V2 NEGATION TESTS PASSED")
print("=" * 90)

# %%
# ============================================================
# EFFUSION V2 — GOLD EVALUATION
# ============================================================

effusion_v2_results = gold["Report"].apply(
    effusion_label_v1
)

effusion_v2_df = pd.DataFrame(
    effusion_v2_results.tolist(),
    index=gold.index
)

effusion_v2_df = effusion_v2_df.rename(columns={
    "label": "Effusion_weak_v2",
    "status": "Effusion_status_v2",
    "confidence": "Effusion_confidence_v2",
    "matched_sentence": "Effusion_sentence_v2",
    "matched_pattern": "Effusion_match_v2",
    "reason": "Effusion_reason_v2"
})

gold_effusion_v2 = pd.concat(
    [gold, effusion_v2_df],
    axis=1
)

evaluated = gold_effusion_v2[
    gold_effusion_v2["Effusion_weak_v2"].isin([0, 1])
].copy()

TP = (
    (evaluated["Effusion"] == 1) &
    (evaluated["Effusion_weak_v2"] == 1)
).sum()

TN = (
    (evaluated["Effusion"] == 0) &
    (evaluated["Effusion_weak_v2"] == 0)
).sum()

FP = (
    (evaluated["Effusion"] == 0) &
    (evaluated["Effusion_weak_v2"] == 1)
).sum()

FN = (
    (evaluated["Effusion"] == 1) &
    (evaluated["Effusion_weak_v2"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_effusion_v2)

print("=" * 80)
print("EFFUSION V2 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_effusion_v2) - len(evaluated)
)

# %%
# ============================================================
# CELL 50 — SYNOVITIS V1 PATTERNS
# ============================================================

SYNOVITIS_POSITIVE_PATTERNS = [

    # --------------------------------------------------------
    # Direct terminology
    # --------------------------------------------------------

    r"\bsynovitis\b",

    r"\bsynovit(?:is|ic)\b",

    # --------------------------------------------------------
    # Synovial thickening / hypertrophy
    # --------------------------------------------------------

    r"\bsynovial\s+thickening\b",

    r"\bthickening\s+of\s+the\s+synovium\b",

    r"\bthickened\s+synovium\b",

    r"\bhypertrophy\s+of\s+the\s+synovium\b",

    r"\bhypertrophied\s+synovium\b",

    r"\bsynovial\s+hypertrophy\b",

    r"\bsynovial\s+proliferation\b",

    # --------------------------------------------------------
    # Indicative / compatible wording
    # --------------------------------------------------------

    r"\b(?:indicative|compatible|consistent|suggestive)\b"
    r".{0,50}\bsynovitis\b",

    r"\b(?:indicative|compatible|consistent|suggestive)\b"
    r".{0,50}\bsynovial\s+(?:thickening|hypertrophy|proliferation)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\bsinovitis\b",

    r"\bhipertrofia\s+sinovial\b",

    r"\bengrosamiento\s+sinovial\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\bsinovit\b",

    r"\bsinovyal\s+(?:kalınlaşma|hipertrofi)\b",

    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    r"\bsynovialitis\b",

    r"\bsynoviale\s+Verdickung\b",

    # --------------------------------------------------------
    # Croatian / Bosnian / Serbian
    # --------------------------------------------------------

    r"\bsinovitis\b",

    r"\bzadebljanje\s+sinovije\b",
]


SYNOVITIS_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # Direct negative
    # --------------------------------------------------------

    r"\bno\s+synovitis\b",

    r"\bwithout\s+synovitis\b",

    r"\bno\s+evidence\s+of\s+synovitis\b",

    # --------------------------------------------------------
    # No synovial thickening / hypertrophy
    # --------------------------------------------------------

    r"\bno\s+synovial\s+thickening\b",

    r"\bno\s+thickening\s+of\s+the\s+synovium\b",

    r"\bno\s+synovial\s+hypertrophy\b",

    # --------------------------------------------------------
    # Normal synovium
    # --------------------------------------------------------

    r"\bsynovium\b.{0,50}\b"
    r"(?:normal|unremarkable|preserved)\b",

    r"\b(?:normal|unremarkable|preserved)\b"
    r".{0,50}\bsynovium\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\bsinovitis\b.{0,50}\b(?:ausente|sin|no)\b",

    r"\bsin\b.{0,40}\bsinovitis\b",

    # --------------------------------------------------------
    # German
    # --------------------------------------------------------

    r"\bkeine\s+Synovialitis\b",
]


SYNOVITIS_UNCERTAIN_PATTERNS = [

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"questionable|may\s+represent|could\s+represent)\b"
    r".{0,60}\bsynovitis\b",

    r"\bsynovitis\b.{0,60}\b"
    r"(?:possible|possibly|suspected|suspect|suggestive|likely)\b",

    # Synovial thickening with uncertainty
    r"\b(?:possible|suspected|suggestive|likely)\b"
    r".{0,60}\bsynovial\s+(?:thickening|hypertrophy)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,60}\bsinovitis\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\b(?:olası|şüpheli|muhtemel)\b"
    r".{0,60}\bsinovit\b",
]


print("Synovitis V1 patterns loaded.")

# %%
# ============================================================
# CELL 51 — SYNOVITIS V1 DETECTOR
# ============================================================

def synovitis_label_v1(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # Negative first
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            SYNOVITIS_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # Uncertainty second
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            SYNOVITIS_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # Positive last
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            SYNOVITIS_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # Definitive positive
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive synovitis evidence detected"
        }

    # --------------------------------------------------------
    # Explicit negative
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of synovitis"
        }

    # --------------------------------------------------------
    # Uncertain
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain synovitis statement"
        }

    # --------------------------------------------------------
    # Not mentioned
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No synovitis evidence detected"
    }


print("Synovitis V1 detector created.")

# %%
# ============================================================
# CELL 52 — SYNOVITIS V1 UNIT TESTS
# ============================================================

synovitis_tests = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Synovitis.": "POSITIVE",

    "Mild synovitis of the knee joint.": "POSITIVE",

    "Moderate synovitis is present.": "POSITIVE",

    "Hypertrophy of the synovium, indicative of synovitis.": "POSITIVE",

    "Synovial thickening compatible with synovitis.": "POSITIVE",

    "Thickened synovium.": "POSITIVE",

    "Synovial hypertrophy.": "POSITIVE",

    # Spanish
    "Leve sinovitis.": "POSITIVE",

    # Croatian
    "Sinovitis.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "No synovitis.": "NEGATIVE",

    "Without synovitis.": "NEGATIVE",

    "No evidence of synovitis.": "NEGATIVE",

    "No synovial thickening.": "NEGATIVE",

    "Normal synovium.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible synovitis.": "UNCERTAIN",

    "Suspected synovitis.": "UNCERTAIN",

    "Suggestive of synovitis.": "UNCERTAIN",

    "Possible synovial thickening.": "UNCERTAIN",

    # --------------------------------------------------------
    # MIXED CONTEXT
    # --------------------------------------------------------

    "Joint effusion with synovitis.": "POSITIVE",

    "Joint effusion without synovitis.": "NEGATIVE",
}


passed = 0
failed = 0

for text, expected in synovitis_tests.items():

    result = synovitis_label_v1(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("SYNOVITIS V1 UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(synovitis_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL SYNOVITIS V1 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — FIX BEFORE GOLD EVALUATION")

# %%
# ============================================================
# CELL 53 — SYNOVITIS V1 GOLD EVALUATION
# ============================================================

synovitis_results = gold["Report"].apply(
    synovitis_label_v1
)

synovitis_results_df = pd.DataFrame(
    synovitis_results.tolist(),
    index=gold.index
)

synovitis_results_df = synovitis_results_df.rename(columns={
    "label": "Synovitis_weak_v1",
    "status": "Synovitis_status_v1",
    "confidence": "Synovitis_confidence_v1",
    "matched_sentence": "Synovitis_sentence_v1",
    "matched_pattern": "Synovitis_match_v1",
    "reason": "Synovitis_reason_v1"
})

gold_synovitis_v1 = pd.concat(
    [gold, synovitis_results_df],
    axis=1
)

print("Synovitis V1 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_synovitis_v1[
        "Synovitis_weak_v1"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_synovitis_v1[
        "Synovitis_status_v1"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 54 — SYNOVITIS V1 GOLD METRICS
# ============================================================

evaluated = gold_synovitis_v1[
    gold_synovitis_v1["Synovitis_weak_v1"].isin([0, 1])
].copy()

TP = (
    (evaluated["Synovitis"] == 1) &
    (evaluated["Synovitis_weak_v1"] == 1)
).sum()

TN = (
    (evaluated["Synovitis"] == 0) &
    (evaluated["Synovitis_weak_v1"] == 0)
).sum()

FP = (
    (evaluated["Synovitis"] == 0) &
    (evaluated["Synovitis_weak_v1"] == 1)
).sum()

FN = (
    (evaluated["Synovitis"] == 1) &
    (evaluated["Synovitis_weak_v1"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_synovitis_v1)

print("=" * 80)
print("SYNOVITIS V1 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_synovitis_v1) - len(evaluated)
)

# %%
# ============================================================
# CELL 55 — SYNOVITIS V1 ERRORS
# ============================================================

fp = gold_synovitis_v1[
    (gold_synovitis_v1["Synovitis"] == 0) &
    (gold_synovitis_v1["Synovitis_weak_v1"] == 1)
]

fn = gold_synovitis_v1[
    (gold_synovitis_v1["Synovitis"] == 1) &
    (gold_synovitis_v1["Synovitis_weak_v1"] == 0)
]

print("=" * 100)
print("SYNOVITIS V1 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Synovitis"])
    print("Weak:", row["Synovitis_weak_v1"])
    print("Status:", row["Synovitis_status_v1"])

    print("\nMatched sentence:")
    print(row["Synovitis_sentence_v1"])

    print("\nMatched pattern:")
    print(row["Synovitis_match_v1"])

    print("\nReason:")
    print(row["Synovitis_reason_v1"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 56 — BAKER'S CYST V1 PATTERNS
# ============================================================

BAKERS_POSITIVE_PATTERNS = [

    # English
    r"\bbaker['’]?s?\s+cyst\b",

    r"\bpopliteal cyst\b",

    r"\bpopliteal fluid collection\b",

    r"\bcyst\b.{0,60}\bpopliteal\b",

    # Spanish
    r"\bquiste poplíteo\b",

    r"\bquiste de baker\b",

    # Turkish
    r"\bbaker kisti\b",

    r"\bpopliteal kist\b",

    # German
    r"\bBaker[- ]Zyste\b",

    r"\bpopliteale Zyste\b",

    # Croatian
    r"\bpoplitealna cista\b",

    r"\bBakerova cista\b",
]


BAKERS_NEGATIVE_PATTERNS = [

    # English
    r"\bno baker['’]?s?\s+cyst\b",

    r"\bno popliteal cyst\b",

    r"\bwithout baker['’]?s?\s+cyst\b",

    r"\bwithout popliteal cyst\b",

    r"\bbaker['’]?s?\s+cyst\b.{0,30}\bnone\b",

    r"\bpopliteal cyst\b.{0,30}\bnone\b",

    # Spanish
    r"\bno hay\b.{0,40}\bquiste[s]?\s+poplíteos?\b",

    r"\bsin\b.{0,40}\bquiste[s]?\s+poplíteos?\b",

    r"\bno hay\b.{0,40}\bquiste de baker\b",

    # Turkish
    r"\bbaker kisti\b.{0,30}\b(?:yok|izlenmedi)\b",

    # German
    r"\bkeine\b.{0,40}\bBaker[- ]Zyste\b",

    # Croatian
    r"\bbe[zź]\b.{0,40}\bpoplitealna cista\b",
]


BAKERS_UNCERTAIN_PATTERNS = [

    # English
    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"suspicious|questionable)\b"
    r".{0,60}\b(?:baker['’]?s?\s+cyst|popliteal cyst)\b",

    r"\b(?:baker['’]?s?\s+cyst|popliteal cyst)\b"
    r".{0,60}\b(?:possible|possibly|suspected|suspect|suggestive|"
    r"likely|suspicious|questionable)\b",

    # Rule out
    r"\br/o\b.{0,100}\b(?:baker['’]?s?\s+cyst|popliteal cyst)\b",

    r"\brule\s+out\b.{0,100}\b(?:baker['’]?s?\s+cyst|popliteal cyst)\b",

    # Spanish
    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,60}\b(?:quiste poplíteo|quiste de baker)\b",

    # Turkish
    r"\b(?:olası|şüpheli|muhtemel)\b"
    r".{0,60}\b(?:baker kisti|popliteal kist)\b",
]


print("Baker's cyst V1 patterns loaded.")

# %%
# ============================================================
# CELL 57 — BAKER'S CYST V1 DETECTOR
# ============================================================

def bakers_label_v1(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # NEGATIVE FIRST
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            BAKERS_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # UNCERTAIN SECOND
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            BAKERS_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # POSITIVE LAST
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            BAKERS_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # DEFINITIVE POSITIVE
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive Baker's cyst evidence detected"
        }

    # --------------------------------------------------------
    # EXPLICIT NEGATIVE
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of Baker's cyst"
        }

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain Baker's cyst statement"
        }

    # --------------------------------------------------------
    # NOT MENTIONED
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No Baker's cyst evidence detected"
    }


print("Baker's cyst V1 detector created.")

# %%
# ============================================================
# CELL 58 — BAKER'S CYST V1 UNIT TESTS
# ============================================================

bakers_tests = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Baker's cyst.": "POSITIVE",

    "Bakers cyst is present.": "POSITIVE",

    "Popliteal cyst.": "POSITIVE",

    "Popliteal fluid collection.": "POSITIVE",

    "There is a Baker's cyst measuring 21 x 17 x 35 mm.": "POSITIVE",

    "Moderate Baker cyst with slight surrounding edema.": "POSITIVE",

    # Spanish
    "Quiste poplíteo.": "POSITIVE",

    "Quiste de Baker.": "POSITIVE",

    # German
    "Baker-Zyste.": "POSITIVE",

    # Croatian
    "Bakerova cista.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "No Baker's cyst.": "NEGATIVE",

    "No popliteal cyst.": "NEGATIVE",

    "Without Baker's cyst.": "NEGATIVE",

    "Baker's cyst: None.": "NEGATIVE",

    "Popliteal cyst: None.": "NEGATIVE",

    # Spanish
    "No hay quistes poplíteos.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible Baker's cyst.": "UNCERTAIN",

    "Suspected Baker's cyst.": "UNCERTAIN",

    "Possible popliteal cyst.": "UNCERTAIN",

    "R/O Baker's cyst.": "UNCERTAIN",

    # --------------------------------------------------------
    # MIXED
    # --------------------------------------------------------

    "No Baker's cyst. Mild joint effusion.": "NEGATIVE",

}


passed = 0
failed = 0

for text, expected in bakers_tests.items():

    result = bakers_label_v1(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("BAKER'S CYST V1 UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(bakers_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL BAKER'S CYST V1 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION")

# %%
# ============================================================
# CELL 59 — BAKER'S CYST V1 GOLD EVALUATION
# ============================================================

bakers_results = gold["Report"].apply(
    bakers_label_v1
)

bakers_results_df = pd.DataFrame(
    bakers_results.tolist(),
    index=gold.index
)

bakers_results_df = bakers_results_df.rename(columns={
    "label": "Bakers_weak_v1",
    "status": "Bakers_status_v1",
    "confidence": "Bakers_confidence_v1",
    "matched_sentence": "Bakers_sentence_v1",
    "matched_pattern": "Bakers_match_v1",
    "reason": "Bakers_reason_v1"
})

gold_bakers_v1 = pd.concat(
    [gold, bakers_results_df],
    axis=1
)

print("Baker's cyst V1 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_bakers_v1[
        "Bakers_weak_v1"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_bakers_v1[
        "Bakers_status_v1"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 60 — BAKER'S CYST V1 GOLD METRICS
# ============================================================

evaluated = gold_bakers_v1[
    gold_bakers_v1["Bakers_weak_v1"].isin([0, 1])
].copy()

TP = (
    (evaluated["Baker's"] == 1) &
    (evaluated["Bakers_weak_v1"] == 1)
).sum()

TN = (
    (evaluated["Baker's"] == 0) &
    (evaluated["Bakers_weak_v1"] == 0)
).sum()

FP = (
    (evaluated["Baker's"] == 0) &
    (evaluated["Bakers_weak_v1"] == 1)
).sum()

FN = (
    (evaluated["Baker's"] == 1) &
    (evaluated["Bakers_weak_v1"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_bakers_v1)

print("=" * 80)
print("BAKER'S CYST V1 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_bakers_v1) - len(evaluated)
)

# %%
# ============================================================
# CELL 61 — BAKER'S CYST V1 ERRORS
# ============================================================

fp = gold_bakers_v1[
    (gold_bakers_v1["Baker's"] == 0) &
    (gold_bakers_v1["Bakers_weak_v1"] == 1)
]

fn = gold_bakers_v1[
    (gold_bakers_v1["Baker's"] == 1) &
    (gold_bakers_v1["Bakers_weak_v1"] == 0)
]

print("=" * 100)
print("BAKER'S CYST V1 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Baker's"])
    print("Weak:", row["Bakers_weak_v1"])
    print("Status:", row["Bakers_status_v1"])

    print("\nMatched sentence:")
    print(row["Bakers_sentence_v1"])

    print("\nMatched pattern:")
    print(row["Bakers_match_v1"])

    print("\nReason:")
    print(row["Bakers_reason_v1"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 62 — BAKER'S CYST V1 FINAL GOLD METRICS
# ============================================================

evaluated = gold_bakers_v1[
    gold_bakers_v1["Bakers_weak_v1"].isin([0, 1])
].copy()

TP = (
    (evaluated["Baker's"] == 1) &
    (evaluated["Bakers_weak_v1"] == 1)
).sum()

TN = (
    (evaluated["Baker's"] == 0) &
    (evaluated["Bakers_weak_v1"] == 0)
).sum()

FP = (
    (evaluated["Baker's"] == 0) &
    (evaluated["Bakers_weak_v1"] == 1)
).sum()

FN = (
    (evaluated["Baker's"] == 1) &
    (evaluated["Bakers_weak_v1"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_bakers_v1)

print("=" * 80)
print("BAKER'S CYST V1 — FINAL GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_bakers_v1) - len(evaluated)
)

# %%
# ============================================================
# CELL 63 — MEDIAL OA V1 PATTERNS
# ============================================================

MEDIAL_OA_POSITIVE_PATTERNS = [

    # Explicit medial osteoarthritis
    r"\bmedial\s+(?:compartment|femorotibial compartment)\b"
    r".{0,100}\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b",

    r"\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b"
    r".{0,100}\bmedial\s+(?:compartment|femorotibial compartment)\b",

    # Medial compartment degenerative/cartilage findings
    r"\bmedial\s+compartment\b"
    r".{0,100}\b(?:chondrosis|chondropathy|cartilage loss|"
    r"cartilage thinning|cartilage defect|degenerative changes)\b",

    r"\bmedial\s+femorotibial\b"
    r".{0,100}\b(?:chondrosis|chondropathy|osteoarthritis|oa|"
    r"cartilage loss|cartilage thinning|degenerative changes)\b",

    # Medial femoral/tibial articular cartilage degeneration
    r"\bmedial\s+(?:femoral condyle|tibial plateau)\b"
    r".{0,100}\b(?:cartilage loss|cartilage thinning|"
    r"chondrosis|chondropathy|degenerative)\b",

    # Medial compartment osteophytes with degenerative context
    r"\bmedial\s+(?:joint line|compartment|femorotibial)\b"
    r".{0,100}\bosteophytes?\b",

    # Spanish
    r"\b(?:oa|artrosis|osteoartrosis)\b"
    r".{0,100}\b(?:compartimento medial|femorotibial medial)\b",

    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,100}\b(?:condropatía|artrosis|osteoartrosis)\b",

    # Turkish
    r"\bmedial\b.{0,80}\b(?:oa|osteoartrit|artroz|kondromalazi)\b",

    # Croatian
    r"\bmedijal(?:ni|nom)\b.{0,100}\b"
    r"(?:OA|artroza|degenerativne promjene|hondromalacija)\b",
]


MEDIAL_OA_NEGATIVE_PATTERNS = [

    # Explicit negative
    r"\bno\b.{0,50}\bmedial\b.{0,50}\b"
    r"(?:osteoarthritis|osteoarthrosis|oa|arthrosis|chondrosis)\b",

    r"\bmedial\s+compartment\b"
    r".{0,70}\b(?:normal|intact|no abnormality|without abnormality)\b",

    r"\bmedial\s+(?:femorotibial|compartment)\b"
    r".{0,70}\b(?:without degenerative changes|without chondrosis)\b",

    # Spanish
    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,70}\b(?:sin alteraciones|sin cambios degenerativos|normal)\b",

    # Turkish
    r"\bmedial\b.{0,70}\b(?:normal|intakt|degenerasyon yok)\b",

    # Croatian
    r"\bmedijal(?:ni|nom)\b.{0,70}\b"
    r"(?:uredan|normalan|bez degenerativnih promjena)\b",
]


MEDIAL_OA_UNCERTAIN_PATTERNS = [

    # English
    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"questionable|may represent|could represent)\b"
    r".{0,100}\bmedial\b"
    r".{0,100}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",

    r"\bmedial\b.{0,100}\b"
    r"(?:possible|possibly|suspected|suspect|suggestive|likely)\b"
    r".{0,60}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",

    # Spanish
    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,100}\b(?:compartimento medial|femorotibial medial)\b",

    # Rule-out
    r"\br/o\b.{0,120}\bmedial\b"
    r".{0,80}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",
]


print("Medial OA V1 patterns loaded.")

# %%
# ============================================================
# CELL 63 — MEDIAL OA V1 PATTERNS
# ============================================================

MEDIAL_OA_POSITIVE_PATTERNS = [

    # Explicit medial osteoarthritis
    r"\bmedial\s+(?:compartment|femorotibial compartment)\b"
    r".{0,100}\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b",

    r"\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b"
    r".{0,100}\bmedial\s+(?:compartment|femorotibial compartment)\b",

    # Medial compartment degenerative/cartilage findings
    r"\bmedial\s+compartment\b"
    r".{0,100}\b(?:chondrosis|chondropathy|cartilage loss|"
    r"cartilage thinning|cartilage defect|degenerative changes)\b",

    r"\bmedial\s+femorotibial\b"
    r".{0,100}\b(?:chondrosis|chondropathy|osteoarthritis|oa|"
    r"cartilage loss|cartilage thinning|degenerative changes)\b",

    # Medial femoral/tibial articular cartilage degeneration
    r"\bmedial\s+(?:femoral condyle|tibial plateau)\b"
    r".{0,100}\b(?:cartilage loss|cartilage thinning|"
    r"chondrosis|chondropathy|degenerative)\b",

    # Medial compartment osteophytes with degenerative context
    r"\bmedial\s+(?:joint line|compartment|femorotibial)\b"
    r".{0,100}\bosteophytes?\b",

    # Spanish
    r"\b(?:oa|artrosis|osteoartrosis)\b"
    r".{0,100}\b(?:compartimento medial|femorotibial medial)\b",

    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,100}\b(?:condropatía|artrosis|osteoartrosis)\b",

    # Turkish
    r"\bmedial\b.{0,80}\b(?:oa|osteoartrit|artroz|kondromalazi)\b",

    # Croatian
    r"\bmedijal(?:ni|nom)\b.{0,100}\b"
    r"(?:OA|artroza|degenerativne promjene|hondromalacija)\b",
]


MEDIAL_OA_NEGATIVE_PATTERNS = [

    # Explicit negative
    r"\bno\b.{0,50}\bmedial\b.{0,50}\b"
    r"(?:osteoarthritis|osteoarthrosis|oa|arthrosis|chondrosis)\b",

    r"\bmedial\s+compartment\b"
    r".{0,70}\b(?:normal|intact|no abnormality|without abnormality)\b",

    r"\bmedial\s+(?:femorotibial|compartment)\b"
    r".{0,70}\b(?:without degenerative changes|without chondrosis)\b",

    # Spanish
    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,70}\b(?:sin alteraciones|sin cambios degenerativos|normal)\b",

    # Turkish
    r"\bmedial\b.{0,70}\b(?:normal|intakt|degenerasyon yok)\b",

    # Croatian
    r"\bmedijal(?:ni|nom)\b.{0,70}\b"
    r"(?:uredan|normalan|bez degenerativnih promjena)\b",
]


MEDIAL_OA_UNCERTAIN_PATTERNS = [

    # English
    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|"
    r"questionable|may represent|could represent)\b"
    r".{0,100}\bmedial\b"
    r".{0,100}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",

    r"\bmedial\b.{0,100}\b"
    r"(?:possible|possibly|suspected|suspect|suggestive|likely)\b"
    r".{0,60}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",

    # Spanish
    r"\b(?:posible|probable|sospecha|sospechoso)\b"
    r".{0,100}\b(?:compartimento medial|femorotibial medial)\b",

    # Rule-out
    r"\br/o\b.{0,120}\bmedial\b"
    r".{0,80}\b(?:oa|osteoarthritis|arthrosis|chondrosis)\b",
]


print("Medial OA V1 patterns loaded.")

# %%
# ============================================================
# CELL 64 — MEDIAL OA V1 DETECTOR
# ============================================================

def medial_oa_label_v1(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # NEGATIVE FIRST
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            MEDIAL_OA_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # UNCERTAIN SECOND
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            MEDIAL_OA_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # POSITIVE LAST
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            MEDIAL_OA_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # DEFINITIVE POSITIVE
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive medial OA evidence detected"
        }

    # --------------------------------------------------------
    # EXPLICIT NEGATIVE
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of medial OA"
        }

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain medial OA statement"
        }

    # --------------------------------------------------------
    # NOT MENTIONED
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial OA evidence detected"
    }


print("Medial OA V1 detector created.")

# %%
# ============================================================
# CELL 65 — MEDIAL OA V1 UNIT TESTS
# ============================================================

medial_oa_tests = {

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    "Medial compartment osteoarthritis.": "POSITIVE",

    "Osteoarthritis of the medial compartment.": "POSITIVE",

    "Medial compartment chondrosis.": "POSITIVE",

    "Medial femorotibial chondrosis.": "POSITIVE",

    "Medial compartment cartilage loss.": "POSITIVE",

    "Degenerative changes in the medial compartment.": "POSITIVE",

    "Medial femoral condyle cartilage loss.": "POSITIVE",

    "Medial tibial plateau chondrosis.": "POSITIVE",

    "Medial compartment osteophytes.": "POSITIVE",

    # Spanish
    "Artrosis del compartimento medial.": "POSITIVE",

    "Condropatía femorotibial medial.": "POSITIVE",

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    "No medial osteoarthritis.": "NEGATIVE",

    "No osteoarthritis of the medial compartment.": "NEGATIVE",

    "Medial compartment is normal.": "NEGATIVE",

    "Medial compartment without degenerative changes.": "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.": "NEGATIVE",

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    "Possible medial osteoarthritis.": "UNCERTAIN",

    "Suspected medial compartment osteoarthritis.": "UNCERTAIN",

    "Suggestive of medial compartment chondrosis.": "UNCERTAIN",

    "Possible medial compartment degenerative changes.": "UNCERTAIN",

    # --------------------------------------------------------
    # ANATOMICAL SEPARATION
    # --------------------------------------------------------

    "Lateral compartment osteoarthritis. Medial compartment is normal.":
        "NEGATIVE",

    "Medial compartment osteoarthritis. Lateral compartment is normal.":
        "POSITIVE",
}


passed = 0
failed = 0

for text, expected in medial_oa_tests.items():

    result = medial_oa_label_v1(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL OA V1 UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(medial_oa_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL MEDIAL OA V1 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — FIX BEFORE GOLD EVALUATION")

# %%
# ============================================================
# CELL 66 — MEDIAL OA V1 POSITIVE PATTERNS — FIXED
# ============================================================

MEDIAL_OA_POSITIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit medial OA
    # --------------------------------------------------------

    r"\bmedial\s+(?:compartment|femorotibial compartment)\b"
    r".{0,100}\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b",

    r"\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b"
    r".{0,100}\bmedial\s+(?:compartment|femorotibial compartment)\b",

    # --------------------------------------------------------
    # Medial degenerative/cartilage findings
    # --------------------------------------------------------

    r"\bmedial\s+compartment\b"
    r".{0,100}\b(?:chondrosis|chondropathy|cartilage loss|"
    r"cartilage thinning|cartilage defect|degenerative changes)\b",

    r"\bdegenerative changes\b"
    r".{0,100}\bmedial\s+(?:compartment|femorotibial)\b",

    r"\bmedial\s+femorotibial\b"
    r".{0,100}\b(?:chondrosis|chondropathy|osteoarthritis|oa|"
    r"cartilage loss|cartilage thinning|degenerative changes)\b",

    # --------------------------------------------------------
    # Medial cartilage
    # --------------------------------------------------------

    r"\bmedial\s+(?:femoral condyle|tibial plateau)\b"
    r".{0,100}\b(?:cartilage loss|cartilage thinning|"
    r"chondrosis|chondropathy|degenerative)\b",

    # --------------------------------------------------------
    # Osteophytes
    # --------------------------------------------------------

    r"\bmedial\s+(?:joint line|compartment|femorotibial)\b"
    r".{0,100}\bosteophytes?\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:artrosis|osteoartrosis|oa)\b"
    r".{0,100}\b(?:compartimento medial|femorotibial medial)\b",

    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,100}\b(?:condropatía|artrosis|osteoartrosis)\b",

    r"\bcondropatía\s+femorotibial\s+medial\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------

    r"\bmedial\b.{0,80}\b"
    r"(?:oa|osteoartrit|artroz|kondromalazi)\b",

    # --------------------------------------------------------
    # Croatian
    # --------------------------------------------------------

    r"\bmedijal(?:ni|nom)\b.{0,100}\b"
    r"(?:OA|artroza|degenerativne promjene|hondromalacija)\b",
]


print("Medial OA positive patterns updated.")

# %%
# ============================================================
# CELL 67 — MEDIAL OA V2 DETECTOR
# ============================================================

def medial_oa_label_v2(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # ----------------------------------------------------
        # NEGATIVE FIRST
        # ----------------------------------------------------

        negative_match = first_match(
            sentence,
            MEDIAL_OA_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ----------------------------------------------------
        # UNCERTAINTY SECOND
        # ----------------------------------------------------

        uncertain_match = first_match(
            sentence,
            MEDIAL_OA_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ----------------------------------------------------
        # POSITIVE LAST
        # ----------------------------------------------------

        positive_match = first_match(
            sentence,
            MEDIAL_OA_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # --------------------------------------------------------
    # DEFINITIVE POSITIVE
    # --------------------------------------------------------

    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive medial OA evidence detected"
        }

    # --------------------------------------------------------
    # EXPLICIT NEGATIVE
    # --------------------------------------------------------

    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of medial OA"
        }

    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain medial OA statement"
        }

    # --------------------------------------------------------
    # NOT MENTIONED
    # --------------------------------------------------------

    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial OA evidence detected"
    }


print("Medial OA V2 detector created.")

# %%
# ============================================================
# CELL 68 — MEDIAL OA V2 CRITICAL TESTS
# ============================================================

critical_tests = {

    "Degenerative changes in the medial compartment.":
        "POSITIVE",

    "Condropatía femorotibial medial.":
        "POSITIVE",

    "No osteoarthritis of the medial compartment.":
        "NEGATIVE",

    "Possible medial compartment degenerative changes.":
        "UNCERTAIN",
}


passed = 0
failed = 0

for text, expected in critical_tests.items():

    result = medial_oa_label_v2(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL OA V2 CRITICAL TEST SUMMARY")
print("=" * 100)

print("Total :", len(critical_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL OA V2 CRITICAL TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED")

# %%
# ============================================================
# CELL 69 — MEDIAL OA V3 DETECTOR
# ============================================================

def medial_oa_label_v3(report):

    if pd.isna(report) or not str(report).strip():

        return {
            "label": np.nan,
            "status": "NOT_MENTIONED",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "Empty report"
        }

    report = str(report)

    sentences = split_sentences(report)

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:

        # Normalize whitespace for cleaner matching
        clean_sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        # ====================================================
        # 1. EXPLICIT NEGATION CHECK
        # ====================================================

        explicit_negative = re.search(
            r"\bno\s+(?:osteoarthritis|osteoarthrosis|oa|arthrosis)"
            r"\s+of\s+the\s+medial\s+compartment\b",
            clean_sentence,
            flags=re.IGNORECASE
        )

        if explicit_negative:

            negative_hits.append(
                (
                    sentence,
                    explicit_negative.group(0)
                )
            )

            continue

        # ====================================================
        # 2. EXPLICIT UNCERTAINTY CHECK
        # ====================================================

        explicit_uncertain = re.search(
            r"\b(?:possible|possibly|suspected|suspect|suggestive|"
            r"likely|questionable)\b"
            r".{0,100}\bmedial\s+compartment\b"
            r".{0,100}\b(?:oa|osteoarthritis|arthrosis|"
            r"chondrosis|degenerative changes)\b",
            clean_sentence,
            flags=re.IGNORECASE
        )

        if explicit_uncertain:

            uncertain_hits.append(
                (
                    sentence,
                    explicit_uncertain.group(0)
                )
            )

            continue

        # ====================================================
        # 3. GENERAL NEGATIVE PATTERNS
        # ====================================================

        negative_match = first_match(
            clean_sentence,
            MEDIAL_OA_NEGATIVE_PATTERNS
        )

        if negative_match:

            negative_hits.append(
                (sentence, negative_match)
            )

            continue

        # ====================================================
        # 4. GENERAL UNCERTAINTY PATTERNS
        # ====================================================

        uncertain_match = first_match(
            clean_sentence,
            MEDIAL_OA_UNCERTAIN_PATTERNS
        )

        if uncertain_match:

            uncertain_hits.append(
                (sentence, uncertain_match)
            )

            continue

        # ====================================================
        # 5. POSITIVE
        # ====================================================

        positive_match = first_match(
            clean_sentence,
            MEDIAL_OA_POSITIVE_PATTERNS
        )

        if positive_match:

            positive_hits.append(
                (sentence, positive_match)
            )

    # ========================================================
    # REPORT-LEVEL DECISION
    # ========================================================

    # Definitive positive
    if positive_hits:

        sentence, pattern = positive_hits[0]

        return {
            "label": 1,
            "status": "POSITIVE",
            "confidence": 0.95,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Definitive medial OA evidence detected"
        }

    # Explicit negative
    if negative_hits:

        sentence, pattern = negative_hits[0]

        return {
            "label": 0,
            "status": "NEGATIVE",
            "confidence": 0.98,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Explicit absence of medial OA"
        }

    # Uncertain
    if uncertain_hits:

        sentence, pattern = uncertain_hits[0]

        return {
            "label": np.nan,
            "status": "UNCERTAIN",
            "confidence": 0.50,
            "matched_sentence": sentence,
            "matched_pattern": pattern,
            "reason": "Uncertain medial OA statement"
        }

    # Not mentioned
    return {
        "label": np.nan,
        "status": "NOT_MENTIONED",
        "confidence": 0.0,
        "matched_sentence": "",
        "matched_pattern": "",
        "reason": "No medial OA evidence detected"
    }


print("Medial OA V3 detector created.")

# %%
# ============================================================
# CELL 70 — MEDIAL OA V3 FAILURE CHECK
# ============================================================

critical_tests = {

    "No osteoarthritis of the medial compartment.":
        "NEGATIVE",

    "Possible medial compartment degenerative changes.":
        "UNCERTAIN",

    "Medial compartment osteoarthritis.":
        "POSITIVE",

    "Degenerative changes in the medial compartment.":
        "POSITIVE",

    "Condropatía femorotibial medial.":
        "POSITIVE",
}


for text, expected in critical_tests.items():

    result = medial_oa_label_v3(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])

    assert result["status"] == expected, (
        f"\nFAILED\n"
        f"Text: {text}\n"
        f"Expected: {expected}\n"
        f"Actual: {result['status']}"
    )


print("\n" + "=" * 100)
print("✅ ALL MEDIAL OA V3 CRITICAL TESTS PASSED")
print("=" * 100)

# %%
# ============================================================
# CELL 71 — MEDIAL OA V3 FULL UNIT TESTS
# ============================================================

medial_oa_tests_v3 = {

    # ========================================================
    # POSITIVE
    # ========================================================

    "Medial compartment osteoarthritis.": "POSITIVE",

    "Osteoarthritis of the medial compartment.": "POSITIVE",

    "Medial compartment chondrosis.": "POSITIVE",

    "Medial femorotibial chondrosis.": "POSITIVE",

    "Medial compartment cartilage loss.": "POSITIVE",

    "Degenerative changes in the medial compartment.": "POSITIVE",

    "Medial femoral condyle cartilage loss.": "POSITIVE",

    "Medial tibial plateau chondrosis.": "POSITIVE",

    "Medial compartment osteophytes.": "POSITIVE",

    "Artrosis del compartimento medial.": "POSITIVE",

    "Condropatía femorotibial medial.": "POSITIVE",

    # ========================================================
    # NEGATIVE
    # ========================================================

    "No medial osteoarthritis.": "NEGATIVE",

    "No osteoarthritis of the medial compartment.": "NEGATIVE",

    "Medial compartment is normal.": "NEGATIVE",

    "Medial compartment without degenerative changes.": "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.": "NEGATIVE",

    # ========================================================
    # UNCERTAIN
    # ========================================================

    "Possible medial osteoarthritis.": "UNCERTAIN",

    "Suspected medial compartment osteoarthritis.": "UNCERTAIN",

    "Suggestive of medial compartment chondrosis.": "UNCERTAIN",

    "Possible medial compartment degenerative changes.": "UNCERTAIN",

    # ========================================================
    # ANATOMICAL SEPARATION
    # ========================================================

    "Lateral compartment osteoarthritis. Medial compartment is normal.":
        "NEGATIVE",

    "Medial compartment osteoarthritis. Lateral compartment is normal.":
        "POSITIVE",
}


passed = 0
failed = 0

for text, expected in medial_oa_tests_v3.items():

    result = medial_oa_label_v3(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL OA V3 FULL UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(medial_oa_tests_v3))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL MEDIAL OA V3 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION")

# %%
# ============================================================
# CELL 72 — MEDIAL OA V3 GOLD EVALUATION
# ============================================================

medial_oa_results = gold["Report"].apply(
    medial_oa_label_v3
)

medial_oa_results_df = pd.DataFrame(
    medial_oa_results.tolist(),
    index=gold.index
)

medial_oa_results_df = medial_oa_results_df.rename(columns={
    "label": "Medial_OA_weak_v3",
    "status": "Medial_OA_status_v3",
    "confidence": "Medial_OA_confidence_v3",
    "matched_sentence": "Medial_OA_sentence_v3",
    "matched_pattern": "Medial_OA_match_v3",
    "reason": "Medial_OA_reason_v3"
})

gold_medial_oa_v3 = pd.concat(
    [gold, medial_oa_results_df],
    axis=1
)

print("Medial OA V3 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_medial_oa_v3[
        "Medial_OA_weak_v3"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_medial_oa_v3[
        "Medial_OA_status_v3"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 73 — MEDIAL OA V3 GOLD METRICS
# ============================================================

evaluated = gold_medial_oa_v3[
    gold_medial_oa_v3["Medial_OA_weak_v3"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3"] == 1)
).sum()

TN = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3"] == 0)
).sum()

FP = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3"] == 1)
).sum()

FN = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_medial_oa_v3)

print("=" * 80)
print("MEDIAL OA V3 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_medial_oa_v3) - len(evaluated)
)

# %%
# ============================================================
# CELL 74 — MEDIAL OA V3 ERRORS
# ============================================================

fp = gold_medial_oa_v3[
    (gold_medial_oa_v3["Medial OA"] == 0) &
    (gold_medial_oa_v3["Medial_OA_weak_v3"] == 1)
]

fn = gold_medial_oa_v3[
    (gold_medial_oa_v3["Medial OA"] == 1) &
    (gold_medial_oa_v3["Medial_OA_weak_v3"] == 0)
]

print("=" * 100)
print("MEDIAL OA V3 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Medial OA"])
    print("Weak:", row["Medial_OA_weak_v3"])
    print("Status:", row["Medial_OA_status_v3"])

    print("\nMatched sentence:")
    print(row["Medial_OA_sentence_v3"])

    print("\nMatched pattern:")
    print(row["Medial_OA_match_v3"])

    print("\nReason:")
    print(row["Medial_OA_reason_v3"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 75 — MEDIAL OA NEGATION FIX
# ============================================================

MEDIAL_OA_NEGATIVE_PATTERNS += [

    # "no focal chondrosis"
    r"\bmedial\s+compartment\b"
    r".{0,100}\bno\s+(?:focal\s+)?chondrosis\b",

    # "without chondrosis"
    r"\bmedial\s+compartment\b"
    r".{0,100}\bwithout\b.{0,40}\bchondrosis\b",

    # "no cartilage abnormality"
    r"\bmedial\s+compartment\b"
    r".{0,100}\bno\b.{0,50}"
    r"(?:cartilage|chondral)\s+(?:abnormality|lesion|defect)\b",
]

print("Medial OA negation patterns updated.")

# %%
# ============================================================
# CELL 76 — TEST MEDIAL OA NEGATION
# ============================================================

text = (
    "In the medial compartment, the meniscus is not torn "
    "and there is no focal chondrosis or chondral injury."
)

result = medial_oa_label_v3(text)

print(result)

assert result["status"] == "NEGATIVE"
assert result["label"] == 0

print("\n✅ MEDIAL OA NEGATION TEST PASSED")

# %%
# ============================================================
# CELL 77 — TEST MEDIAL OA POSITIVE
# ============================================================

text = (
    "The medial compartment demonstrates focal chondrosis "
    "with cartilage loss."
)

result = medial_oa_label_v3(text)

print(result)

assert result["status"] == "POSITIVE"
assert result["label"] == 1

print("\n✅ MEDIAL OA POSITIVE TEST PASSED")

# %%
# ============================================================
# CELL 76 — TEST MEDIAL OA NEGATION
# ============================================================

text = (
    "In the medial compartment, the meniscus is not torn "
    "and there is no focal chondrosis or chondral injury."
)

result = medial_oa_label_v3(text)

print(result)

assert result["status"] == "NEGATIVE"
assert result["label"] == 0

print("\n✅ MEDIAL OA NEGATION TEST PASSED")

# %%
# ============================================================
# CELL 71 — MEDIAL OA V3 FULL UNIT TESTS
# ============================================================

medial_oa_tests_v3 = {

    # ========================================================
    # POSITIVE
    # ========================================================

    "Medial compartment osteoarthritis.": "POSITIVE",

    "Osteoarthritis of the medial compartment.": "POSITIVE",

    "Medial compartment chondrosis.": "POSITIVE",

    "Medial femorotibial chondrosis.": "POSITIVE",

    "Medial compartment cartilage loss.": "POSITIVE",

    "Degenerative changes in the medial compartment.": "POSITIVE",

    "Medial femoral condyle cartilage loss.": "POSITIVE",

    "Medial tibial plateau chondrosis.": "POSITIVE",

    "Medial compartment osteophytes.": "POSITIVE",

    "Artrosis del compartimento medial.": "POSITIVE",

    "Condropatía femorotibial medial.": "POSITIVE",

    # ========================================================
    # NEGATIVE
    # ========================================================

    "No medial osteoarthritis.": "NEGATIVE",

    "No osteoarthritis of the medial compartment.": "NEGATIVE",

    "Medial compartment is normal.": "NEGATIVE",

    "Medial compartment without degenerative changes.": "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.": "NEGATIVE",

    # ========================================================
    # UNCERTAIN
    # ========================================================

    "Possible medial osteoarthritis.": "UNCERTAIN",

    "Suspected medial compartment osteoarthritis.": "UNCERTAIN",

    "Suggestive of medial compartment chondrosis.": "UNCERTAIN",

    "Possible medial compartment degenerative changes.": "UNCERTAIN",

    # ========================================================
    # ANATOMICAL SEPARATION
    # ========================================================

    "Lateral compartment osteoarthritis. Medial compartment is normal.":
        "NEGATIVE",

    "Medial compartment osteoarthritis. Lateral compartment is normal.":
        "POSITIVE",
}


passed = 0
failed = 0

for text, expected in medial_oa_tests_v3.items():

    result = medial_oa_label_v3(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL OA V3 FULL UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(medial_oa_tests_v3))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:

    print("\n✅ ALL MEDIAL OA V3 UNIT TESTS PASSED")

else:

    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION")

# %%
# ============================================================
# CELL 72 — MEDIAL OA V3 GOLD EVALUATION
# ============================================================

medial_oa_results = gold["Report"].apply(
    medial_oa_label_v3
)

medial_oa_results_df = pd.DataFrame(
    medial_oa_results.tolist(),
    index=gold.index
)

medial_oa_results_df = medial_oa_results_df.rename(columns={
    "label": "Medial_OA_weak_v3",
    "status": "Medial_OA_status_v3",
    "confidence": "Medial_OA_confidence_v3",
    "matched_sentence": "Medial_OA_sentence_v3",
    "matched_pattern": "Medial_OA_match_v3",
    "reason": "Medial_OA_reason_v3"
})

gold_medial_oa_v3 = pd.concat(
    [gold, medial_oa_results_df],
    axis=1
)

print("Medial OA V3 applied successfully.")

print("\nWeak-label distribution:")
print(
    gold_medial_oa_v3[
        "Medial_OA_weak_v3"
    ].value_counts(dropna=False)
)

print("\nStatus distribution:")
print(
    gold_medial_oa_v3[
        "Medial_OA_status_v3"
    ].value_counts(dropna=False)
)

# %%
# ============================================================
# CELL 73 — MEDIAL OA V3 GOLD METRICS
# ============================================================

evaluated = gold_medial_oa_v3[
    gold_medial_oa_v3["Medial_OA_weak_v3"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3"] == 1)
).sum()

TN = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3"] == 0)
).sum()

FP = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3"] == 1)
).sum()

FN = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_medial_oa_v3)

print("=" * 80)
print("MEDIAL OA V3 — GOLD RESULTS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_medial_oa_v3) - len(evaluated)
)

# %%
# ============================================================
# CELL 74 — MEDIAL OA V3 ERRORS
# ============================================================

fp = gold_medial_oa_v3[
    (gold_medial_oa_v3["Medial OA"] == 0) &
    (gold_medial_oa_v3["Medial_OA_weak_v3"] == 1)
]

fn = gold_medial_oa_v3[
    (gold_medial_oa_v3["Medial OA"] == 1) &
    (gold_medial_oa_v3["Medial_OA_weak_v3"] == 0)
]

print("=" * 100)
print("MEDIAL OA V3 ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)

    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Medial OA"])
    print("Weak:", row["Medial_OA_weak_v3"])
    print("Status:", row["Medial_OA_status_v3"])

    print("\nMatched sentence:")
    print(row["Medial_OA_sentence_v3"])

    print("\nMatched pattern:")
    print(row["Medial_OA_match_v3"])

    print("\nReason:")
    print(row["Medial_OA_reason_v3"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 75 — MEDIAL OA NEGATIVE PATTERNS — CLEANED
# ============================================================

MEDIAL_OA_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit OA negation
    # --------------------------------------------------------

    r"\bno\s+(?:osteoarthritis|osteoarthrosis|oa|arthrosis)"
    r"\s+of\s+the\s+medial\s+compartment\b",

    r"\bno\s+(?:medial\s+)?(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\b",

    # --------------------------------------------------------
    # Explicit absence of degenerative changes
    # --------------------------------------------------------

    r"\bmedial\s+compartment\b"
    r".{0,80}\bwithout\s+degenerative\s+changes\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bwithout\s+chondrosis\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bno\s+(?:focal\s+)?chondrosis\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bno\s+(?:focal\s+)?chondral\s+(?:lesion|injury|defect)\b",

    # --------------------------------------------------------
    # Explicit normal cartilage
    # --------------------------------------------------------

    r"\bmedial\s+compartment\s+cartilage\b"
    r".{0,60}\b(?:normal|intact|preserved)\b",

    r"\bmedial\s+(?:femorotibial|articular)\s+cartilage\b"
    r".{0,60}\b(?:normal|intact|preserved)\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,80}\b(?:sin alteraciones|sin cambios degenerativos|"
    r"sin condropatía|normal)\b",

    # --------------------------------------------------------
    # Turkish
    # --------------------------------------------------------



    # --------------------------------------------------------
    # Croatian
    # --------------------------------------------------------

    r"\bmedijal(?:ni|nom)\b"
    r".{0,80}\b(?:uredan|normalan|bez degenerativnih promjena)\b",
]

MEDIAL_OA_NEGATIVE_PATTERNS += [
    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bwithout\s+chondrosis\b",
]

print("Medial OA negative patterns cleaned.")

# %%
# ============================================================
# CELL 76 — MEDIAL OA NEGATIVE TESTS
# ============================================================

negative_tests = {

    "No osteoarthritis of the medial compartment.":
        "NEGATIVE",

    "Medial compartment without degenerative changes.":
        "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.":
        "NEGATIVE",

    "In the medial compartment, the meniscus is not torn "
    "and there is no focal chondrosis or chondral injury.":
        "NEGATIVE",

    "Medial compartment cartilage is intact.":
        "NEGATIVE",

    "Medial collateral ligament (MCL): Normal.":
        "NOT_MENTIONED",

    "Medial meniscus: Normal in morphology and signal.":
        "NOT_MENTIONED",
}


for text, expected in negative_tests.items():

    result = medial_oa_label_v3(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])

    assert result["status"] == expected


print("\n" + "=" * 100)
print("✅ ALL MEDIAL OA NEGATIVE CONTEXT TESTS PASSED")
print("=" * 100)

# %%
# ============================================================
# CELL 77 — MEDIAL OA V3 NEGATIVE PATTERN FIX
# ============================================================

MEDIAL_OA_NEGATIVE_PATTERNS += [

    # Medial femorotibial compartment without chondrosis
    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bwithout\s+chondrosis\b",

    # Medial femorotibial compartment without degenerative changes
    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bwithout\s+degenerative\s+changes\b",

    # Medial femorotibial compartment with no chondrosis
    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bno\s+chondrosis\b",
]

print("Medial OA negative pattern fix added.")



# %%
# ============================================================
# CELL 78 — TEST THE FAILED CASE
# ============================================================

text = "Medial femorotibial compartment without chondrosis."

result = medial_oa_label_v3(text)

print(result)

assert result["status"] == "NEGATIVE"
assert result["label"] == 0

print("\n✅ MEDIAL FEMOROTIBIAL NEGATIVE TEST PASSED")

# %%
# ============================================================
# CELL 79 — MEDIAL OA NEGATIVE CONTEXT TESTS
# ============================================================

negative_tests = {

    "No osteoarthritis of the medial compartment.":
        "NEGATIVE",

    "Medial compartment without degenerative changes.":
        "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.":
        "NEGATIVE",

    "In the medial compartment, the meniscus is not torn "
    "and there is no focal chondrosis or chondral injury.":
        "NEGATIVE",

    "Medial compartment cartilage is intact.":
        "NEGATIVE",

    "Medial collateral ligament (MCL): Normal.":
        "NOT_MENTIONED",

    "Medial meniscus: Normal in morphology and signal.":
        "NOT_MENTIONED",
}


passed = 0
failed = 0

for text, expected in negative_tests.items():

    result = medial_oa_label_v3(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])

    if result["status"] == expected:
        passed += 1
        print("✅ PASS")
    else:
        failed += 1
        print("❌ FAIL")


print("\n" + "=" * 100)
print("MEDIAL OA NEGATIVE CONTEXT SUMMARY")
print("=" * 100)

print("Total :", len(negative_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL OA NEGATIVE CONTEXT TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED")

# %%
# ============================================================
# CELL 80 — CLEAN MEDIAL OA NEGATIVE PATTERNS
# ============================================================

MEDIAL_OA_NEGATIVE_PATTERNS = [

    # --------------------------------------------------------
    # Explicit medial OA negation
    # --------------------------------------------------------

    r"\bno\s+(?:osteoarthritis|osteoarthrosis|oa|arthrosis)"
    r"\s+of\s+the\s+medial\s+compartment\b",

    r"\bno\s+(?:medial\s+)?(?:osteoarthritis|osteoarthrosis|"
    r"oa|arthrosis)\b",

    # --------------------------------------------------------
    # Medial compartment explicitly without degeneration
    # --------------------------------------------------------

    r"\bmedial\s+compartment\b"
    r".{0,80}\bwithout\s+degenerative\s+changes\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bwithout\s+chondrosis\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bno\s+(?:focal\s+)?chondrosis\b",

    r"\bmedial\s+compartment\b"
    r".{0,80}\bno\s+(?:focal\s+)?chondral\s+"
    r"(?:lesion|injury|defect)\b",

    # --------------------------------------------------------
    # Medial cartilage explicitly normal/intact
    # --------------------------------------------------------

    r"\bmedial\s+compartment\s+cartilage\b"
    r".{0,60}\b(?:normal|intact|preserved)\b",

    r"\bmedial\s+(?:femorotibial|articular)\s+cartilage\b"
    r".{0,60}\b(?:normal|intact|preserved)\b",

    # --------------------------------------------------------
    # Medial femorotibial compartment
    # --------------------------------------------------------

    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bwithout\s+chondrosis\b",

    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bwithout\s+degenerative\s+changes\b",

    r"\bmedial\s+femorotibial\s+compartment\b"
    r".{0,60}\bno\s+chondrosis\b",

    # --------------------------------------------------------
    # Spanish
    # --------------------------------------------------------

    r"\b(?:compartimento medial|femorotibial medial)\b"
    r".{0,80}\b(?:sin alteraciones|sin cambios degenerativos|"
    r"sin condropatía|normal)\b",
]

print("Medial OA negative patterns CLEANED.")

# %%
# ============================================================
# CELL 81 — MEDIAL OA NEGATIVE CONTEXT TESTS
# ============================================================

negative_tests = {

    "No osteoarthritis of the medial compartment.":
        "NEGATIVE",

    "Medial compartment without degenerative changes.":
        "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.":
        "NEGATIVE",

    "In the medial compartment, the meniscus is not torn "
    "and there is no focal chondrosis or chondral injury.":
        "NEGATIVE",

    "Medial compartment cartilage is intact.":
        "NEGATIVE",

    # These are NOT about OA
    "Medial collateral ligament (MCL): Normal.":
        "NOT_MENTIONED",

    "Medial meniscus: Normal in morphology and signal.":
        "NOT_MENTIONED",
}


passed = 0
failed = 0

for text, expected in negative_tests.items():

    result = medial_oa_label_v3(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])

    if result["status"] == expected:
        passed += 1
        print("✅ PASS")
    else:
        failed += 1
        print("❌ FAIL")


print("\n" + "=" * 100)
print("MEDIAL OA NEGATIVE CONTEXT SUMMARY")
print("=" * 100)

print("Total :", len(negative_tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL OA NEGATIVE CONTEXT TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED")

# %%
# ============================================================
# CELL 82 — MEDIAL OA V3 FINAL UNIT TEST
# ============================================================

medial_oa_tests_final = {

    # ========================================================
    # POSITIVE
    # ========================================================

    "Medial compartment osteoarthritis.": "POSITIVE",

    "Osteoarthritis of the medial compartment.": "POSITIVE",

    "Medial compartment chondrosis.": "POSITIVE",

    "Medial femorotibial chondrosis.": "POSITIVE",

    "Medial compartment cartilage loss.": "POSITIVE",

    "Degenerative changes in the medial compartment.": "POSITIVE",

    "Medial femoral condyle cartilage loss.": "POSITIVE",

    "Medial tibial plateau chondrosis.": "POSITIVE",

    "Medial compartment osteophytes.": "POSITIVE",

    "Artrosis del compartimento medial.": "POSITIVE",

    "Condropatía femorotibial medial.": "POSITIVE",

    # ========================================================
    # NEGATIVE
    # ========================================================

    "No medial osteoarthritis.": "NEGATIVE",

    "No osteoarthritis of the medial compartment.": "NEGATIVE",

    "Medial compartment is normal.": "NEGATIVE",

    "Medial compartment without degenerative changes.": "NEGATIVE",

    "Medial femorotibial compartment without chondrosis.": "NEGATIVE",

    # ========================================================
    # UNCERTAIN
    # ========================================================

    "Possible medial osteoarthritis.": "UNCERTAIN",

    "Suspected medial compartment osteoarthritis.": "UNCERTAIN",

    "Suggestive of medial compartment chondrosis.": "UNCERTAIN",

    "Possible medial compartment degenerative changes.": "UNCERTAIN",

    # ========================================================
    # ANATOMICAL SEPARATION
    # ========================================================

    "Lateral compartment osteoarthritis. Medial compartment is normal.":
        "NEGATIVE",

    "Medial compartment osteoarthritis. Lateral compartment is normal.":
        "POSITIVE",
}


passed = 0
failed = 0

for text, expected in medial_oa_tests_final.items():

    result = medial_oa_label_v3(text)

    actual = result["status"]

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", actual)
    print("Label:", result["label"])

    if actual == expected:

        passed += 1
        print("✅ PASS")

    else:

        failed += 1
        print("❌ FAIL")
        print("Matched:", result["matched_pattern"])
        print("Reason:", result["reason"])


print("\n" + "=" * 100)
print("MEDIAL OA V3 FINAL UNIT TEST SUMMARY")
print("=" * 100)

print("Total :", len(medial_oa_tests_final))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ ALL MEDIAL OA V3 UNIT TESTS PASSED")
else:
    print("\n❌ SOME TESTS FAILED — DO NOT RUN GOLD EVALUATION")

# %%
# ============================================================
# CELL 83 — MEDIAL OA NORMAL-COMPARTMENT FIX
# ============================================================

MEDIAL_OA_NEGATIVE_PATTERNS += [

    # Exact medial compartment normal statement
    r"\bmedial\s+compartment\b"
    r"\s+(?:is|appears\s+to\s+be|looks\s+)\s*"
    r"(?:normal|unremarkable|preserved|intact)\b",

    # "The medial compartment is normal"
    r"\bthe\s+medial\s+compartment\b"
    r"\s+(?:is|appears\s+to\s+be|looks\s+)\s*"
    r"(?:normal|unremarkable|preserved|intact)\b",
]

print("Medial OA normal-compartment patterns added.")

# %%
# ============================================================
# CELL 84 — MEDIAL OA NORMAL-COMPARTMENT TEST
# ============================================================

tests = {

    "Medial compartment is normal.": "NEGATIVE",

    "Lateral compartment osteoarthritis. Medial compartment is normal.":
        "NEGATIVE",
}


passed = 0
failed = 0

for text, expected in tests.items():

    result = medial_oa_label_v3(text)

    print("\n" + "-" * 100)
    print("Text:", text)
    print("Expected:", expected)
    print("Actual:", result["status"])
    print("Label:", result["label"])
    print("Matched:", result["matched_pattern"])

    if result["status"] == expected:
        passed += 1
        print("✅ PASS")
    else:
        failed += 1
        print("❌ FAIL")


print("\n" + "=" * 100)
print("MEDIAL OA NORMAL-COMPARTMENT TEST")
print("=" * 100)

print("Total :", len(tests))
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\n✅ BOTH NORMAL-COMPARTMENT TESTS PASSED")
else:
    print("\n❌ TESTS STILL FAILING")

# %%
# ============================================================
# CELL 85 — MEDIAL OA V3 FINAL GOLD EVALUATION
# ============================================================

medial_oa_final_results = gold["Report"].apply(
    medial_oa_label_v3
)

medial_oa_final_df = pd.DataFrame(
    medial_oa_final_results.tolist(),
    index=gold.index
)

medial_oa_final_df = medial_oa_final_df.rename(columns={
    "label": "Medial_OA_weak_v3_final",
    "status": "Medial_OA_status_v3_final",
    "confidence": "Medial_OA_confidence_v3_final",
    "matched_sentence": "Medial_OA_sentence_v3_final",
    "matched_pattern": "Medial_OA_match_v3_final",
    "reason": "Medial_OA_reason_v3_final"
})

gold_medial_oa_final = pd.concat(
    [gold, medial_oa_final_df],
    axis=1
)

evaluated = gold_medial_oa_final[
    gold_medial_oa_final["Medial_OA_weak_v3_final"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3_final"] == 1)
).sum()

TN = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3_final"] == 0)
).sum()

FP = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3_final"] == 1)
).sum()

FN = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3_final"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_medial_oa_final)

print("=" * 90)
print("MEDIAL OA V3 — FINAL GOLD RESULTS")
print("=" * 90)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_medial_oa_final) - len(evaluated)
)

# %%
# ============================================================
# CELL 86 — MEDIAL OA V3 FINAL ERRORS
# ============================================================

fp = gold_medial_oa_final[
    (gold_medial_oa_final["Medial OA"] == 0) &
    (gold_medial_oa_final["Medial_OA_weak_v3_final"] == 1)
]

fn = gold_medial_oa_final[
    (gold_medial_oa_final["Medial OA"] == 1) &
    (gold_medial_oa_final["Medial_OA_weak_v3_final"] == 0)
]

print("=" * 100)
print("MEDIAL OA V3 FINAL ERRORS")
print("=" * 100)

print("FALSE POSITIVES:", len(fp))
print("FALSE NEGATIVES:", len(fn))

for _, row in pd.concat([fp, fn]).iterrows():

    print("\n" + "-" * 100)
    print("Study:", row["StudyInstanceUID"])
    print("Gold:", row["Medial OA"])
    print("Weak:", row["Medial_OA_weak_v3_final"])
    print("Status:", row["Medial_OA_status_v3_final"])

    print("\nMatched sentence:")
    print(row["Medial_OA_sentence_v3_final"])

    print("\nMatched pattern:")
    print(row["Medial_OA_match_v3_final"])

    print("\nReason:")
    print(row["Medial_OA_reason_v3_final"])

    print("\nREPORT:")
    print(row["Report"])

# %%
# ============================================================
# CELL 87 — MEDIAL OA V3 FINAL METRICS
# ============================================================

evaluated = gold_medial_oa_final[
    gold_medial_oa_final["Medial_OA_weak_v3_final"].isin([0, 1])
].copy()

TP = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3_final"] == 1)
).sum()

TN = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3_final"] == 0)
).sum()

FP = (
    (evaluated["Medial OA"] == 0) &
    (evaluated["Medial_OA_weak_v3_final"] == 1)
).sum()

FN = (
    (evaluated["Medial OA"] == 1) &
    (evaluated["Medial_OA_weak_v3_final"] == 0)
).sum()

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall else 0
)

coverage = len(evaluated) / len(gold_medial_oa_final)

print("=" * 80)
print("MEDIAL OA V3 — FINAL LOCKED METRICS")
print("=" * 80)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("Coverage:", round(coverage * 100, 2), "%")

print(
    "Uncertain / Not Mentioned:",
    len(gold_medial_oa_final) - len(evaluated)
)
#csv file output
results_summary = pd.DataFrame([{
    "Target": "Medial OA",
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Coverage": coverage
}])

results_summary.to_csv(
    "medial_oa_metrics.csv",
    index=False
)

print("Metrics saved to medial_oa_metrics.csv")
