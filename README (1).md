# Knee Abnormality Detection from MRI Reports

## Study Project / Capstone Project

This project explores the use of rule-based natural language processing (NLP) to identify knee abnormalities from radiology reports. The project focuses on extracting findings such as ligament injuries, meniscal tears, osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture.

This is a research prototype for academic study. It is not a clinical diagnostic system and must not be used to make medical decisions.

## Dataset

The project uses the RSNA Knee Abnormality Detection dataset from the Kaggle competition:

<https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data>

The complete dataset contains a very large collection of knee MRI DICOM images. It is not necessary to download the full image dataset for this report-text analysis. The current Python script uses the CSV metadata and the `Report` column.

The expected training file is:

```text
train.csv
```

It must be placed in the same folder as `knee_abnormality_project.py`.

The analysis uses the following target findings:

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's
- Contusion
- Fracture

The target labels use the following meaning:

- `1`: abnormality present
- `0`: abnormality absent
- blank/unknown: no reviewed label available

## Project Objective

The main objectives are to:

1. Load and inspect knee MRI radiology-report data.
2. Identify positive, negative, uncertain, and unmentioned findings.
3. Create weak labels using medical terminology and regular-expression patterns.
4. Compare weak NLP labels with available reviewed labels.
5. Measure performance using confusion-matrix statistics.
6. Analyse false positives, false negatives, and cases that could not be classified.

## Methodology

The project follows this workflow:

```text
CSV data
   ↓
Data inspection and missing-value analysis
   ↓
Sentence splitting and pattern matching
   ↓
Positive / negative / uncertain classification
   ↓
Comparison with reviewed labels
   ↓
Precision, recall, F1-score, coverage, and error analysis
```

The NLP rules include terminology for ligament injuries, meniscal tears, osteoarthritis, fluid, synovitis, cysts, contusions, and fractures. The rules also consider negation, uncertainty, historical statements, and post-surgical language where implemented.

## Project Files

```text
KNEE_ABNORMALITY_PROJECT/
├── knee_abnormality_project.py
├── knee_abnormality_project.ipynb
├── train.csv
├── medial_oa_metrics.csv
└── README.md
```

The full DICOM image folders are intentionally not included in this project folder because they require hundreds of gigabytes of storage. Image-based model training can be performed in a Kaggle Notebook, where the competition data is available in the notebook environment.

## Requirements

Install Python 3.10 or a compatible Python version. The project uses packages including:

```bash
pip install pandas numpy scikit-learn ipython
```

If a virtual environment is used on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn ipython
```

## How to Run

1. Place `train.csv` beside `knee_abnormality_project.py`.
2. Open the project folder in VS Code.
3. Activate the virtual environment if one is being used.
4. Run:

```bash
python knee_abnormality_project.py
```

The script performs data inspection, applies the NLP rules, evaluates the available gold labels, and prints the results in the terminal.

The script also creates metric files such as:

```text
medial_oa_metrics.csv
```

## Current Preliminary Result: Medial OA

The current preliminary Medial OA evaluation produced:

| Metric | Value |
|---|---:|
| True positives | 2 |
| True negatives | 5 |
| False positives | 1 |
| False negatives | 1 |
| Precision | 0.6667 |
| Recall | 0.6667 |
| F1-score | 0.6667 |
| Coverage | 15.52% |

The result is based only on cases for which the rule system generated a binary prediction. Forty-nine cases were uncertain or not mentioned. Therefore, this result should be considered preliminary and should not be interpreted as a final clinical performance score.

## Evaluation Metrics

For each abnormality, the project calculates:

- True positives (TP)
- True negatives (TN)
- False positives (FP)
- False negatives (FN)
- Precision
- Recall
- F1-score
- Coverage

Coverage is important because a rule system may achieve reasonable accuracy on the cases it classifies while leaving many reports as uncertain or unmentioned.

## Limitations

1. Only a small subset of studies has complete reviewed labels.
2. Rule-based NLP depends on the wording used in the report.
3. Reports may contain uncertainty, historical findings, comparisons, or post-surgical changes.
4. Medical terminology may vary across languages and institutions.
5. Low coverage limits the reliability of the performance results.
6. The current project does not replace expert radiologist review.
7. The results have not been externally validated on an independent hospital dataset.
8. The project currently focuses on report text and does not train a deep-learning model directly on all MRI images.

## Future Improvements

- Expand the manually reviewed gold-standard dataset.
- Have two or more radiologists independently annotate reports.
- Resolve disagreements using a third reviewer.
- Improve multilingual terminology and negation handling.
- Remove duplicated legacy rule versions from the script.
- Add automated unit tests for every target abnormality.
- Create one combined metrics table for all 12 findings.
- Compare rule-based NLP with a machine-learning or transformer-based classifier.
- Perform patient-level train/test splitting to avoid data leakage.
- Validate the method on an independent dataset.

## Ethical and Safety Statement

The data is used for academic analysis. Any use of radiology reports must follow the dataset license, privacy requirements, and institutional or course rules. This project is intended to support research and learning only. It does not provide medical advice, diagnosis, or treatment recommendations.

## Author

Study Project / Capstone Project
