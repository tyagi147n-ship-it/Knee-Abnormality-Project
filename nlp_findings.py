"""
nlp_findings.py — Rule-based NLP finding extraction for knee MRI radiology
REPORT TEXT (not image pixels).

This module is extracted directly from the capstone notebook
`knee_abnormality_project.ipynb`. It reproduces the exact regex-based
weak-labeling pipeline that was tested and evaluated there (the same
code path that produced the "Medial OA V3" pass/fail output and the
final Precision/Recall/F1/Coverage numbers you saw in the terminal).

IMPORTANT — what this actually does:
    These classifiers read a block of RADIOLOGY REPORT TEXT (the
    "Findings"/"Impression" section a radiologist writes) and look for
    sentence-level language patterns ("no osteoarthritis of the medial
    compartment", "moderate joint effusion", etc.) to label each
    finding as POSITIVE / NEGATIVE / UNCERTAIN / NOT_MENTIONED.

    They do NOT look at MRI pixels. The capstone project has no trained
    image classifier (see app.py's own sidebar note), so an uploaded
    DICOM/PNG cannot, by itself, produce these labels. To reproduce the
    terminal output inside the app, the user must also supply the
    report text for that study (paste it in, or load it from the
    dataset) — that text is what actually drives this module.

Two helper functions, `split_sentences` and `first_match`, are used by
every detector in the notebook but were never defined in the notebook
itself (they must have lived in a separate helper file that wasn't
included in the upload). Minimal, standard implementations are
provided below; they were verified to reproduce every pass/fail line
from the notebook's own terminal transcript exactly.
"""

import re
import pandas as pd
import numpy as np


def split_sentences(text):
    """Split a report into sentences on '.', '!', '?'."""
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p for p in parts if p.strip()]


def first_match(sentence, patterns):
    """Return the matched substring for the first pattern that hits, else None."""
    for pattern in patterns:
        m = re.search(pattern, sentence, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


# ============================================================
# Final pattern lists (resolved from the notebook's cell-execution
# order, including every later append/fix, so they match the
# "locked" version that produced the printed metrics)
# ============================================================

MEDIAL_MENISCUS_NEGATIVE_PATTERNS = ['\\bmedial meniscus\\b\\s+(?:is|appears to be|seems to be)\\s+not\\s+torn\\b',
 '\\bmedial meniscus\\b[^.!?]{0,40}\\bnot\\s+torn\\b',
 '\\bmedial meniscus\\b.{0,60}\\b(?:intact|normal|unremarkable|preserved)\\b',
 '\\b(?:normal|intact|unremarkable|preserved)\\b.{0,50}\\bmedial meniscus\\b',
 '\\bmedial meniscus\\b.{0,60}\\bwithout\\b.{0,40}\\btear\\b',
 '\\bmedial meniscus\\b.{0,60}\\bno\\b.{0,40}\\btear\\b',
 '\\bno\\b.{0,50}\\btear\\b.{0,50}\\bmedial meniscus\\b',
 '\\bwithout\\b.{0,50}\\btear\\b.{0,50}\\bmedial meniscus\\b',
 '\\bmedial meniscus\\b.{0,100}\\bno\\s+frank\\s+tear\\b',
 '\\bmenisco medial\\b.{0,100}\\b(?:sin signos de desgarro|sin signos de '
 'rotura|sin signos de ruptura|sin criterios categóricos de rotura)\\b',
 '\\bmenisco interno\\b.{0,100}\\b(?:sin signos de desgarro|sin signos de '
 'rotura|sin signos de ruptura|sin criterios categóricos de rotura)\\b',
 '\\bmenisco interno\\b.{0,80}\\b(?:sin signos de rotura|sin signos de '
 'ruptura|sin criterios.*rotura|normal|conservado)\\b',
 '\\bmedial menisküs\\b.{0,60}\\b(?:normal|sağlam|intakt|yırtık yok)\\b',
 '\\b(?:Innenmeniskus|Innenmenisk)\\b.{0,60}\\b(?:intakt|normal|unauffällig)\\b',
 '\\bmedijalni menisk\\b.{0,60}\\b(?:uredan|normalan|bez znakova rupture)\\b']

MEDIAL_MENISCUS_POSITIVE_PATTERNS = ['\\bmedial '
 'meniscus\\b.{0,100}\\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\\b',
 '\\bmedial '
 'meniscal\\b.{0,100}\\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\\b',
 '\\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\\b.{0,80}\\bmedial '
 'meniscus\\b',
 '\\b(?:tear|tearing|rupture)\\b.{0,60}\\bmedial meniscal\\b',
 '\\bmedial meniscus\\b.{0,100}\\b(?:radial tear|horizontal tear|complex '
 'tear|root tear|bucket[- ]handle tear|longitudinal tear|vertical tear|oblique '
 'tear|degenerative tear)\\b',
 '\\b(?:radial|horizontal|complex|root|bucket[- '
 ']handle|longitudinal|vertical|oblique|degenerative)\\s+tear\\b.{0,80}\\bmedial '
 'meniscus\\b',
 '\\b(?:posterior|anterior)\\s+(?:horn|root)\\b.{0,80}\\bmedial '
 'meniscus\\b.{0,50}\\b(?:tear|tearing|rupture)\\b',
 '\\bmedial meniscus\\b.{0,100}\\b(?:posterior horn|anterior '
 'horn|body|root)\\b.{0,80}\\b(?:tear|tearing|rupture)\\b',
 '\\bmedial '
 'meniscus\\b.{0,100}\\bextrusion\\b.{0,100}\\b(?:tear|tearing|rupture|maceration)\\b',
 '\\b(?:tear|tearing|rupture|maceration)\\b.{0,100}\\bmedial '
 'meniscus\\b.{0,100}\\bextrusion\\b',
 '\\btear of (?:the )?medial meniscus\\b',
 '\\btearing of (?:the )?medial meniscus\\b',
 '\\bmedial meniscus tear\\b',
 '\\bmedial meniscal tear\\b',
 '\\bmenisco medial\\b.{0,100}\\b(?:rotura|ruptura|desgarro)\\b',
 '\\b(?:rotura|ruptura|desgarro)\\b.{0,80}\\bmenisco medial\\b',
 '\\bmenisco interno\\b.{0,100}\\b(?:rotura|ruptura|desgarro)\\b',
 '\\b(?:rotura|ruptura)\\b.{0,80}\\bmenisco interno\\b',
 '\\bmedial menisküs\\b.{0,100}\\b(?:yırtık|yırtığı|rüptür|kopma)\\b',
 '\\b(?:yırtık|yırtığı|rüptür|kopma)\\b.{0,100}\\bmedial menisküs\\b',
 '\\b(?:Innenmeniskus|Innenmenisk)\\b.{0,100}\\b(?:Riss|Ruptur)\\b',
 '\\b(?:Riss|Ruptur)\\b.{0,100}\\b(?:Innenmeniskus|Innenmenisk)\\b',
 '\\bmedijalni menisk\\b.{0,100}\\b(?:ruptura|rutura|puknuće|puknuća)\\b',
 '\\b(?:ruptura|rutura|puknuće|puknuća)\\b.{0,100}\\bmedijalni menisk\\b',
 '\\b(?:медиален|медиалния)\\s+менискус\\b.{0,100}\\b(?:руптура|скъсване|разкъсване)\\b']


# NOTE: `MEDIAL_MENISCUS_UNCERTAIN_PATTERNS` is referenced by
# medial_meniscus_label_v6 in the notebook but is never assigned
# anywhere in the uploaded .ipynb (only the LATERAL meniscus
# equivalent was). This is reconstructed to mirror that list
# (swapping "lateral" for "medial") and checked against every
# medial-meniscus UNCERTAIN test case that appears in the notebook's
# own unit tests ("Possible tear of the medial meniscus.", "Suspected
# medial meniscus tear.", "The finding may represent a tear of the
# medial meniscus.", "Cannot exclude tear of the medial meniscus.",
# etc.) — all pass with this list.
MEDIAL_MENISCUS_UNCERTAIN_PATTERNS = [
    r"\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\b.{0,100}\bmedial meniscus\b",
    r"\bmedial meniscus\b.{0,100}\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\b",
    r"\bmedial meniscus\b.{0,100}\b(?:could represent|may represent)\b",
    r"\b(?:could represent|may represent)\b.{0,100}\bmedial meniscus\b",
    r"\b(?:cannot exclude|can not exclude)\b.{0,100}\bmedial meniscus\b",
    r"\br/o\b.{0,120}\b(?:medial meniscus|medial meniscal)\b",
    r"\brule\s+out\b.{0,120}\b(?:medial meniscus|medial meniscal)\b",
    r"\b(?:r/o|rule\s+out)\b.{0,120}\b(?:tear|rupture)\b.{0,100}\b(?:medial meniscus|medial meniscal)\b",
    r"\b(?:r/o|rule\s+out)\b.{0,120}\b(?:medial meniscus|medial meniscal)\b.{0,80}\b(?:tear|rupture)\b",
    r"\b(?:posible|probable|sospecha|sospechoso)\b.{0,100}\bmenisco medial\b",
]

LATERAL_MENISCUS_NEGATIVE_PATTERNS = ['\\blateral meniscus\\b\\s+(?:is|appears to be|seems to be)\\s+not\\s+torn\\b',
 '\\blateral meniscus\\b[^.!?]{0,50}\\bnot\\s+torn\\b',
 '\\blateral meniscus\\b\\s*[:\\-]\\s*(?:no|without)\\s+(?:evidence '
 'of\\s+)?(?:tear|rupture|injury|abnormality)\\b',
 '\\blateral meniscal\\b\\s*[:\\-]\\s*(?:no|without)\\s+(?:evidence '
 'of\\s+)?(?:tear|rupture|injury|abnormality)\\b',
 '\\bno\\b.{0,50}\\b(?:lateral meniscus|lateral '
 'meniscal)\\b.{0,50}\\b(?:tear|rupture|injury)\\b',
 '\\b(?:lateral meniscus|lateral '
 'meniscal)\\b.{0,50}\\bno\\b.{0,40}\\b(?:tear|rupture|injury)\\b',
 '\\b(?:lateral meniscus|lateral '
 'meniscal)\\b.{0,60}\\bwithout\\b.{0,40}\\btear\\b',
 '\\bwithout\\b.{0,50}\\btear\\b.{0,50}\\b(?:lateral meniscus|lateral '
 'meniscal)\\b',
 '\\blateral meniscus\\b.{0,100}\\bno\\s+frank\\s+tear\\b',
 '\\blateral meniscus\\b.{0,60}\\b(?:intact|normal|unremarkable|preserved)\\b',
 '\\b(?:intact|normal|unremarkable|preserved)\\b.{0,60}\\blateral meniscus\\b',
 '\\bmenisco lateral\\b.{0,100}\\b(?:sin signos de desgarro|sin signos de '
 'rotura|sin signos de ruptura|sin criterios.*rotura|normal|conservado)\\b',
 '\\blateral menisküs\\b.{0,80}\\b(?:normal|sağlam|intakt|yırtık yok)\\b',
 '\\b(?:Außenmeniskus|Außenmenisk)\\b.{0,80}\\b(?:intakt|normal|unauffällig)\\b',
 '\\blateralni menisk\\b.{0,80}\\b(?:uredan|normalan|bez znakova rupture)\\b',
 '\\bno evidence of\\b.{0,50}\\btear\\b.{0,80}\\b(?:lateral meniscus|lateral '
 'meniscal)\\b',
 '\\bno evidence of tear\\b.{0,100}\\b(?:lateral meniscus|lateral meniscal)\\b',
 '\\bno tear\\b.{0,80}\\b(?:lateral meniscus|lateral meniscal)\\b',
 '\\bno evidence\\b.{0,100}\\b(?:medial|lateral)\\s+meniscus\\b',
 '\\bno\\b.{0,50}\\b(?:tear|abnormality|injury|rupture)\\b.{0,100}\\b(?:lateral '
 'meniscus|lateral meniscal)\\b']


# NOTE: `LATERAL_MENISCUS_POSITIVE_PATTERNS` is referenced by
# lateral_meniscus_label_v2 in the notebook but, like the medial
# meniscus uncertain list above, is never assigned anywhere in the
# uploaded .ipynb. Reconstructed by mirroring the notebook's own
# MEDIAL_MENISCUS_POSITIVE_PATTERNS list (swap "medial" -> "lateral").
LATERAL_MENISCUS_POSITIVE_PATTERNS = [
    r"\blateral meniscus\b.{0,100}\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b",
    r"\blateral meniscal\b.{0,100}\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b",
    r"\b(?:tear|tearing|torn|rupture|ruptured|maceration|macerated)\b.{0,80}\blateral meniscus\b",
    r"\b(?:tear|tearing|rupture)\b.{0,60}\blateral meniscal\b",
    r"\blateral meniscus\b.{0,100}\b(?:radial tear|horizontal tear|complex tear|root tear|bucket[- ]handle tear|longitudinal tear|vertical tear|oblique tear|degenerative tear)\b",
    r"\b(?:radial|horizontal|complex|root|bucket[- ]handle|longitudinal|vertical|oblique|degenerative)\s+tear\b.{0,80}\blateral meniscus\b",
    r"\b(?:posterior|anterior)\s+(?:horn|root)\b.{0,80}\blateral meniscus\b.{0,50}\b(?:tear|tearing|rupture)\b",
    r"\blateral meniscus\b.{0,100}\b(?:posterior horn|anterior horn|body|root)\b.{0,80}\b(?:tear|tearing|rupture)\b",
    r"\blateral meniscus\b.{0,100}\bextrusion\b.{0,100}\b(?:tear|tearing|rupture|maceration)\b",
    r"\b(?:tear|tearing|rupture|maceration)\b.{0,100}\blateral meniscus\b.{0,100}\bextrusion\b",
]

LATERAL_MENISCUS_UNCERTAIN_PATTERNS = ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\\b.{0,100}\\blateral '
 'meniscus\\b',
 '\\blateral '
 'meniscus\\b.{0,100}\\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\\b',
 '\\blateral meniscus\\b.{0,100}\\b(?:could represent|may represent)\\b',
 '\\br/o\\b.{0,120}\\b(?:lateral meniscus|lateral meniscal)\\b',
 '\\brule\\s+out\\b.{0,120}\\b(?:lateral meniscus|lateral meniscal)\\b',
 '\\b(?:r/o|rule\\s+out)\\b.{0,120}\\b(?:tear|rupture)\\b.{0,100}\\b(?:lateral '
 'meniscus|lateral meniscal)\\b',
 '\\b(?:r/o|rule\\s+out)\\b.{0,120}\\b(?:lateral meniscus|lateral '
 'meniscal)\\b.{0,80}\\b(?:tear|rupture)\\b',
 '\\b(?:posible|probable|sospecha|sospechoso)\\b.{0,100}\\bmenisco lateral\\b',
 '\\b(?:olası|şüpheli|muhtemel|şüpheli)\\b.{0,100}\\blateral menisküs\\b',
 '\\b(?:moguće|vjerojatno|sumnja)\\b.{0,100}\\blateralni menisk\\b']

EFFUSION_NEGATIVE_PATTERNS = ['\\bno joint effusion\\b',
 '\\bno knee effusion\\b',
 '\\bno evidence of effusion\\b',
 '\\bwithout effusion\\b',
 '\\bwithout joint effusion\\b',
 '\\bno significant effusion\\b',
 '\\bno significant joint effusion\\b',
 '\\beffusion\\s*[:\\-]\\s*(?:none|absent)\\b',
 '\\bno hay\\b.{0,30}\\bderrame\\b',
 '\\bsin\\b.{0,30}\\bderrame\\b',
 '\\bderrame\\b.{0,30}\\b(?:ausente|ausencia)\\b',
 '\\b(?:eklemde|dizde)\\b.{0,50}\\bsıvı artışı yok\\b',
 '\\bkein\\b.{0,30}\\b(?:Gelenkerguss|Erguss)\\b',
 '\\b(?:Gelenkerguss|Erguss)\\b.{0,30}\\bkein\\b',
 '\\bno evidence of\\b.{0,30}\\bknee effusion\\b',
 '\\bno significant\\b.{0,20}\\bknee effusion\\b',
 '\\bno\\b.{0,30}\\b(?:knee|joint)\\s+effusion\\b']

EFFUSION_POSITIVE_PATTERNS = ['\\bjoint effusion\\b',
 '\\bknee effusion\\b',
 '\\bintra[- ]articular effusion\\b',
 '\\beffusion of the knee\\b',
 '\\b(?:small|mild|moderate|large|massive|trace|minimal|moderate[- ]to[- '
 ']large|small[- ]to[- ]moderate)\\b.{0,20}\\beffusion\\b',
 '\\bsome (?:amount of )?right knee effusion\\b',
 '\\bsome (?:amount of )?left knee effusion\\b',
 '\\bsome joint effusion\\b',
 '\\bminimal amount of .*?effusion\\b',
 '\\b(?:joint|knee)\\b.{0,30}\\bfluid\\b',
 '\\bfluid\\b.{0,30}\\b(?:joint|knee)\\b',
 '\\bderrame articular\\b',
 '\\bderrame de la rodilla\\b',
 '\\bderrame\\b.{0,30}\\b(?:articular|rodilla)\\b',
 '\\bdiz ekleminde\\b.{0,50}\\bsıvı artışı\\b',
 '\\beklem aralığında\\b.{0,50}\\bsıvı\\b',
 '\\bKniegelenkerguss\\b',
 '\\bGelenkerguss\\b',
 '\\bErguss\\b',
 '\\bzglobni izljev\\b',
 '\\bizljev\\b']

EFFUSION_UNCERTAIN_PATTERNS = ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|questionable|may '
 'represent|could represent)\\b.{0,50}\\beffusion\\b',
 '\\b(?:posible|probable|sospecha|sospechoso)\\b.{0,50}\\bderrame\\b',
 '\\b(?:olası|şüpheli|muhtemel)\\b.{0,50}\\bsıvı artışı\\b']

SYNOVITIS_NEGATIVE_PATTERNS = ['\\bno\\s+synovitis\\b',
 '\\bwithout\\s+synovitis\\b',
 '\\bno\\s+evidence\\s+of\\s+synovitis\\b',
 '\\bno\\s+synovial\\s+thickening\\b',
 '\\bno\\s+thickening\\s+of\\s+the\\s+synovium\\b',
 '\\bno\\s+synovial\\s+hypertrophy\\b',
 '\\bsynovium\\b.{0,50}\\b(?:normal|unremarkable|preserved)\\b',
 '\\b(?:normal|unremarkable|preserved)\\b.{0,50}\\bsynovium\\b',
 '\\bsinovitis\\b.{0,50}\\b(?:ausente|sin|no)\\b',
 '\\bsin\\b.{0,40}\\bsinovitis\\b',
 '\\bkeine\\s+Synovialitis\\b']

SYNOVITIS_POSITIVE_PATTERNS = ['\\bsynovitis\\b',
 '\\bsynovit(?:is|ic)\\b',
 '\\bsynovial\\s+thickening\\b',
 '\\bthickening\\s+of\\s+the\\s+synovium\\b',
 '\\bthickened\\s+synovium\\b',
 '\\bhypertrophy\\s+of\\s+the\\s+synovium\\b',
 '\\bhypertrophied\\s+synovium\\b',
 '\\bsynovial\\s+hypertrophy\\b',
 '\\bsynovial\\s+proliferation\\b',
 '\\b(?:indicative|compatible|consistent|suggestive)\\b.{0,50}\\bsynovitis\\b',
 '\\b(?:indicative|compatible|consistent|suggestive)\\b.{0,50}\\bsynovial\\s+(?:thickening|hypertrophy|proliferation)\\b',
 '\\bsinovitis\\b',
 '\\bhipertrofia\\s+sinovial\\b',
 '\\bengrosamiento\\s+sinovial\\b',
 '\\bsinovit\\b',
 '\\bsinovyal\\s+(?:kalınlaşma|hipertrofi)\\b',
 '\\bsynovialitis\\b',
 '\\bsynoviale\\s+Verdickung\\b',
 '\\bsinovitis\\b',
 '\\bzadebljanje\\s+sinovije\\b']

SYNOVITIS_UNCERTAIN_PATTERNS = ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|questionable|may\\s+represent|could\\s+represent)\\b.{0,60}\\bsynovitis\\b',
 '\\bsynovitis\\b.{0,60}\\b(?:possible|possibly|suspected|suspect|suggestive|likely)\\b',
 '\\b(?:possible|suspected|suggestive|likely)\\b.{0,60}\\bsynovial\\s+(?:thickening|hypertrophy)\\b',
 '\\b(?:posible|probable|sospecha|sospechoso)\\b.{0,60}\\bsinovitis\\b',
 '\\b(?:olası|şüpheli|muhtemel)\\b.{0,60}\\bsinovit\\b']

BAKERS_NEGATIVE_PATTERNS = ["\\bno baker['’]?s?\\s+cyst\\b",
 '\\bno popliteal cyst\\b',
 "\\bwithout baker['’]?s?\\s+cyst\\b",
 '\\bwithout popliteal cyst\\b',
 "\\bbaker['’]?s?\\s+cyst\\b.{0,30}\\bnone\\b",
 '\\bpopliteal cyst\\b.{0,30}\\bnone\\b',
 '\\bno hay\\b.{0,40}\\bquiste[s]?\\s+poplíteos?\\b',
 '\\bsin\\b.{0,40}\\bquiste[s]?\\s+poplíteos?\\b',
 '\\bno hay\\b.{0,40}\\bquiste de baker\\b',
 '\\bbaker kisti\\b.{0,30}\\b(?:yok|izlenmedi)\\b',
 '\\bkeine\\b.{0,40}\\bBaker[- ]Zyste\\b',
 '\\bbe[zź]\\b.{0,40}\\bpoplitealna cista\\b']

BAKERS_POSITIVE_PATTERNS = ["\\bbaker['’]?s?\\s+cyst\\b",
 '\\bpopliteal cyst\\b',
 '\\bpopliteal fluid collection\\b',
 '\\bcyst\\b.{0,60}\\bpopliteal\\b',
 '\\bquiste poplíteo\\b',
 '\\bquiste de baker\\b',
 '\\bbaker kisti\\b',
 '\\bpopliteal kist\\b',
 '\\bBaker[- ]Zyste\\b',
 '\\bpopliteale Zyste\\b',
 '\\bpoplitealna cista\\b',
 '\\bBakerova cista\\b']

BAKERS_UNCERTAIN_PATTERNS = ["\\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\\b.{0,60}\\b(?:baker['’]?s?\\s+cyst|popliteal "
 'cyst)\\b',
 "\\b(?:baker['’]?s?\\s+cyst|popliteal "
 'cyst)\\b.{0,60}\\b(?:possible|possibly|suspected|suspect|suggestive|likely|suspicious|questionable)\\b',
 "\\br/o\\b.{0,100}\\b(?:baker['’]?s?\\s+cyst|popliteal cyst)\\b",
 "\\brule\\s+out\\b.{0,100}\\b(?:baker['’]?s?\\s+cyst|popliteal cyst)\\b",
 '\\b(?:posible|probable|sospecha|sospechoso)\\b.{0,60}\\b(?:quiste '
 'poplíteo|quiste de baker)\\b',
 '\\b(?:olası|şüpheli|muhtemel)\\b.{0,60}\\b(?:baker kisti|popliteal kist)\\b']

MEDIAL_OA_NEGATIVE_PATTERNS = ['\\bno\\s+(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\\s+of\\s+the\\s+medial\\s+compartment\\b',
 '\\bno\\s+(?:medial\\s+)?(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\\b',
 '\\bmedial\\s+compartment\\b.{0,80}\\bwithout\\s+degenerative\\s+changes\\b',
 '\\bmedial\\s+compartment\\b.{0,80}\\bwithout\\s+chondrosis\\b',
 '\\bmedial\\s+compartment\\b.{0,80}\\bno\\s+(?:focal\\s+)?chondrosis\\b',
 '\\bmedial\\s+compartment\\b.{0,80}\\bno\\s+(?:focal\\s+)?chondral\\s+(?:lesion|injury|defect)\\b',
 '\\bmedial\\s+compartment\\s+cartilage\\b.{0,60}\\b(?:normal|intact|preserved)\\b',
 '\\bmedial\\s+(?:femorotibial|articular)\\s+cartilage\\b.{0,60}\\b(?:normal|intact|preserved)\\b',
 '\\bmedial\\s+femorotibial\\s+compartment\\b.{0,60}\\bwithout\\s+chondrosis\\b',
 '\\bmedial\\s+femorotibial\\s+compartment\\b.{0,60}\\bwithout\\s+degenerative\\s+changes\\b',
 '\\bmedial\\s+femorotibial\\s+compartment\\b.{0,60}\\bno\\s+chondrosis\\b',
 '\\b(?:compartimento medial|femorotibial medial)\\b.{0,80}\\b(?:sin '
 'alteraciones|sin cambios degenerativos|sin condropatía|normal)\\b',
 '\\bmedial\\s+compartment\\b\\s+(?:is|appears\\s+to\\s+be|looks\\s+)\\s*(?:normal|unremarkable|preserved|intact)\\b',
 '\\bthe\\s+medial\\s+compartment\\b\\s+(?:is|appears\\s+to\\s+be|looks\\s+)\\s*(?:normal|unremarkable|preserved|intact)\\b']

MEDIAL_OA_POSITIVE_PATTERNS = ['\\bmedial\\s+(?:compartment|femorotibial '
 'compartment)\\b.{0,100}\\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\\b',
 '\\b(?:osteoarthritis|osteoarthrosis|oa|arthrosis)\\b.{0,100}\\bmedial\\s+(?:compartment|femorotibial '
 'compartment)\\b',
 '\\bmedial\\s+compartment\\b.{0,100}\\b(?:chondrosis|chondropathy|cartilage '
 'loss|cartilage thinning|cartilage defect|degenerative changes)\\b',
 '\\bdegenerative '
 'changes\\b.{0,100}\\bmedial\\s+(?:compartment|femorotibial)\\b',
 '\\bmedial\\s+femorotibial\\b.{0,100}\\b(?:chondrosis|chondropathy|osteoarthritis|oa|cartilage '
 'loss|cartilage thinning|degenerative changes)\\b',
 '\\bmedial\\s+(?:femoral condyle|tibial plateau)\\b.{0,100}\\b(?:cartilage '
 'loss|cartilage thinning|chondrosis|chondropathy|degenerative)\\b',
 '\\bmedial\\s+(?:joint '
 'line|compartment|femorotibial)\\b.{0,100}\\bosteophytes?\\b',
 '\\b(?:artrosis|osteoartrosis|oa)\\b.{0,100}\\b(?:compartimento '
 'medial|femorotibial medial)\\b',
 '\\b(?:compartimento medial|femorotibial '
 'medial)\\b.{0,100}\\b(?:condropatía|artrosis|osteoartrosis)\\b',
 '\\bcondropatía\\s+femorotibial\\s+medial\\b',
 '\\bmedial\\b.{0,80}\\b(?:oa|osteoartrit|artroz|kondromalazi)\\b',
 '\\bmedijal(?:ni|nom)\\b.{0,100}\\b(?:OA|artroza|degenerativne '
 'promjene|hondromalacija)\\b']

MEDIAL_OA_UNCERTAIN_PATTERNS = ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|questionable|may '
 'represent|could '
 'represent)\\b.{0,100}\\bmedial\\b.{0,100}\\b(?:oa|osteoarthritis|arthrosis|chondrosis)\\b',
 '\\bmedial\\b.{0,100}\\b(?:possible|possibly|suspected|suspect|suggestive|likely)\\b.{0,60}\\b(?:oa|osteoarthritis|arthrosis|chondrosis)\\b',
 '\\b(?:posible|probable|sospecha|sospechoso)\\b.{0,100}\\b(?:compartimento '
 'medial|femorotibial medial)\\b',
 '\\br/o\\b.{0,120}\\bmedial\\b.{0,80}\\b(?:oa|osteoarthritis|arthrosis|chondrosis)\\b']

TARGET_CONFIG = {"Baker's": {'negative_patterns': ["\\bno\\b.{0,30}\\bbaker['’]s?\\s+cyst\\b",
                                   '\\bno\\b.{0,30}\\bpopliteal cyst\\b',
                                   "\\bwithout\\b.{0,30}\\bbaker['’]s?\\s+cyst\\b",
                                   "\\bbaker['’]s?\\s+cyst\\b.{0,30}\\bnone\\b"],
             'positive_patterns': ["\\bbaker['’]s?\\s+cyst\\b",
                                   '\\bpopliteal cyst\\b',
                                   '\\bpopliteal fluid collection\\b',
                                   '\\bcyst\\b.{0,50}\\bpopliteal\\b'],
             'uncertain_patterns': ["\\b(?:possible|suspected|suggestive)\\b.{0,40}\\b(?:baker['’]s?\\s+cyst|popliteal "
                                    'cyst)\\b']},
 'Contusion': {'negative_patterns': ['\\bno\\b.{0,30}\\b(?:bone contusion|bone '
                                     'bruise)\\b',
                                     '\\bwithout\\b.{0,30}\\b(?:bone '
                                     'contusion|bone bruise)\\b',
                                     '\\bno\\b.{0,40}\\bbone\\b.{0,30}\\bcontusion\\b'],
               'positive_patterns': ['\\bbone contusion\\b',
                                     '\\bbone bruise\\b',
                                     '\\bbone marrow '
                                     'edema\\b.{0,80}\\b(?:contusion|bruise)\\b',
                                     '\\bcontusion\\b.{0,50}\\b(?:bone|bony|osseous|marrow)\\b',
                                     '\\boste?e?dema\\b'],
               'uncertain_patterns': ['\\b(?:possible|suspected|suspect|suggestive)\\b.{0,50}\\b(?:bone '
                                      'contusion|bone bruise)\\b']},
 'Effusion': {'negative_patterns': ['\\bno\\b.{0,30}\\b(?:joint|knee)?\\s*effusion\\b',
                                    '\\bno evidence of\\b.{0,30}\\beffusion\\b',
                                    '\\bwithout\\b.{0,30}\\beffusion\\b'],
              'positive_patterns': ['\\b(?:joint|knee|intra[- '
                                    ']articular)\\b.{0,50}\\b(?:effusion|fluid|collection)\\b',
                                    '\\b(?:effusion|joint fluid|fluid '
                                    'accumulation)\\b',
                                    '\\b(?:mild|moderate|small|large|trace|massive)\\b.{0,20}\\b(?:joint '
                                    'effusion|effusion)\\b',
                                    '\\b(?:derrame|derrame articular)\\b',
                                    '\\b(?:výpotek|izljev)\\b'],
              'uncertain_patterns': ['\\b(?:possible|suspected|suggestive)\\b.{0,40}\\beffusion\\b']},
 'Fracture': {'negative_patterns': ['\\bno\\b.{0,30}\\bfracture\\b',
                                    '\\bwithout\\b.{0,30}\\bfracture\\b',
                                    '\\bno evidence of\\b.{0,30}\\bfracture\\b',
                                    '\\bfracture\\b.{0,30}\\bnone\\b'],
              'positive_patterns': ['\\bfracture\\b',
                                    '\\bfractured\\b',
                                    '\\bfracture line\\b',
                                    '\\bcortical break\\b',
                                    '\\binsufficiency fracture\\b',
                                    '\\bimpaction fracture\\b',
                                    '\\bosteochondral fracture\\b',
                                    '\\bavulsion fracture\\b',
                                    '\\bfractura\\b',
                                    '\\bfraktura\\b',
                                    '\\bfraktur\\b'],
              'uncertain_patterns': ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot '
                                     'exclude)\\b.{0,50}\\bfracture\\b']},
 'Lateral Meniscus': {'negative_patterns': ['\\blateral '
                                            'meniscus\\b.{0,50}\\b(?:intact|normal|unremarkable|preserved|not '
                                            'torn)\\b',
                                            '\\blateral '
                                            'meniscus\\b.{0,50}\\bwithout\\b.{0,40}\\btear\\b',
                                            '\\bno\\b.{0,40}\\b(?:tear|rupture)\\b.{0,40}\\blateral '
                                            'meniscus\\b',
                                            '\\blateral '
                                            'meniscus\\b.{0,40}\\bno\\b.{0,30}\\btear\\b'],
                      'positive_patterns': ['\\blateral '
                                            'meniscus\\b.{0,80}\\b(?:tear|torn|rupture|maceration|macerated|fragment)\\b',
                                            '\\blateral '
                                            'meniscal\\b.{0,80}\\b(?:tear|torn|rupture)\\b',
                                            '\\bmeniscus\\b.{0,50}\\b(?:lateral|lateral '
                                            'compartment)\\b.{0,50}\\b(?:tear|rupture)\\b'],
                      'uncertain_patterns': ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot '
                                             'exclude)\\b.{0,60}\\blateral '
                                             'meniscus\\b',
                                             '\\blateral '
                                             'meniscus\\b.{0,60}\\b(?:possible|suspected|suggestive|likely)\\b']},
 'Lateral OA': {'negative_patterns': ['\\blateral '
                                      'compartment\\b.{0,60}\\b(?:normal|no '
                                      'abnormality|without abnormality)\\b',
                                      '\\bno\\b.{0,50}\\blateral\\b.{0,50}\\b(?:osteoarthritis|oa)\\b'],
                'positive_patterns': ['\\blateral '
                                      'compartment\\b.{0,100}\\b(?:osteoarthritis|oa|chondrosis|chondral '
                                      'loss|cartilage loss|degenerative '
                                      'changes|osteophytes)\\b',
                                      '\\blateral\\b.{0,50}\\b(?:osteoarthritis|oa|chondrosis)\\b',
                                      '\\b(?:osteoarthritis|oa|chondrosis)\\b.{0,50}\\blateral '
                                      'compartment\\b',
                                      '\\blateral '
                                      'femorotibial\\b.{0,80}\\b(?:osteoarthritis|oa|chondrosis|cartilage '
                                      'loss)\\b'],
                'uncertain_patterns': ['\\b(?:possible|suspected|suggestive)\\b.{0,60}\\b(?:lateral '
                                       'oa|lateral osteoarthritis|lateral '
                                       'compartment)\\b']},
 'MCL': {'negative_patterns': ['\\bmcl\\b.{0,40}\\b(?:intact|normal|unremarkable|preserved)\\b',
                               '\\bmedial collateral '
                               'ligament\\b.{0,40}\\b(?:intact|normal|unremarkable|preserved)\\b',
                               '\\b(?:normal|intact|unremarkable|preserved)\\b.{0,40}\\bmcl\\b',
                               '\\bno\\b.{0,40}\\bmcl\\b.{0,40}\\b(?:tear|rupture|injury)\\b'],
         'positive_patterns': ['\\bmcl\\b.{0,60}\\b(?:tear|rupture|ruptured|injury|sprain)\\b',
                               '\\b(?:tear|rupture|ruptured|injury|sprain)\\b.{0,60}\\bmcl\\b',
                               '\\bmedial collateral '
                               'ligament\\b.{0,60}\\b(?:tear|rupture|injury|sprain)\\b',
                               '\\b(?:tear|rupture|injury|sprain)\\b.{0,60}\\bmedial '
                               'collateral ligament\\b'],
         'uncertain_patterns': ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot '
                                'exclude)\\b.{0,60}\\bmcl\\b',
                                '\\b(?:mcl|medial collateral '
                                'ligament)\\b.{0,60}\\b(?:possible|suspected|suggestive|likely)\\b']},
 'Medial Meniscus': {'negative_patterns': ['\\bmedial '
                                           'meniscus\\b.{0,50}\\b(?:intact|normal|unremarkable|preserved|not '
                                           'torn)\\b',
                                           '\\b(?:medial '
                                           'meniscus)\\b.{0,50}\\bwithout\\b.{0,40}\\btear\\b',
                                           '\\bno\\b.{0,40}\\b(?:tear|rupture)\\b.{0,40}\\bmedial '
                                           'meniscus\\b',
                                           '\\b(?:medial '
                                           'meniscus)\\b.{0,40}\\bno\\b.{0,30}\\btear\\b'],
                     'positive_patterns': ['\\bmedial '
                                           'meniscus\\b.{0,80}\\b(?:tear|torn|rupture|maceration|macerated|fragment)\\b',
                                           '\\bmedial '
                                           'meniscal\\b.{0,80}\\b(?:tear|torn|rupture)\\b',
                                           '\\bmeniscus\\b.{0,50}\\b(?:medial|medial '
                                           'compartment)\\b.{0,50}\\b(?:tear|rupture)\\b'],
                     'uncertain_patterns': ['\\b(?:possible|possibly|suspected|suspect|suggestive|likely|cannot '
                                            'exclude)\\b.{0,60}\\bmedial '
                                            'meniscus\\b',
                                            '\\bmedial '
                                            'meniscus\\b.{0,60}\\b(?:possible|suspected|suggestive|likely)\\b']},
 'Medial OA': {'negative_patterns': ['\\bmedial '
                                     'compartment\\b.{0,60}\\b(?:normal|no '
                                     'abnormality|without abnormality)\\b',
                                     '\\bno\\b.{0,50}\\bmedial\\b.{0,50}\\b(?:osteoarthritis|oa)\\b'],
               'positive_patterns': ['\\bmedial '
                                     'compartment\\b.{0,100}\\b(?:osteoarthritis|oa|chondrosis|chondral '
                                     'loss|cartilage loss|degenerative '
                                     'changes|osteophytes)\\b',
                                     '\\bmedial\\b.{0,50}\\b(?:osteoarthritis|oa|chondrosis)\\b',
                                     '\\b(?:osteoarthritis|oa|chondrosis)\\b.{0,50}\\bmedial '
                                     'compartment\\b',
                                     '\\bmedial '
                                     'femorotibial\\b.{0,80}\\b(?:osteoarthritis|oa|chondrosis|cartilage '
                                     'loss)\\b'],
               'uncertain_patterns': ['\\b(?:possible|possible '
                                      'early|suspected|suggestive)\\b.{0,60}\\b(?:medial '
                                      'oa|medial osteoarthritis|medial '
                                      'compartment)\\b']},
 'PF OA': {'negative_patterns': ['\\bpatellofemoral\\b.{0,60}\\b(?:normal|intact|no '
                                 'abnormality)\\b',
                                 '\\bno\\b.{0,50}\\bpatellofemoral\\b.{0,50}\\b(?:oa|osteoarthritis|chondrosis)\\b'],
           'positive_patterns': ['\\bpatellofemoral\\b.{0,100}\\b(?:osteoarthritis|oa|chondrosis|chondral '
                                 'loss|cartilage loss|degenerative '
                                 'changes|osteophytes)\\b',
                                 '\\bpatellofemoral\\b.{0,80}\\b(?:arthrosis|arthritis)\\b',
                                 '\\b(?:patellar|trochlear)\\b.{0,70}\\b(?:chondrosis|cartilage '
                                 'loss|osteoarthritis|oa)\\b',
                                 '\\bpatellar\\b.{0,50}\\b(?:chondromalacia|chondropathy)\\b'],
           'uncertain_patterns': ['\\b(?:possible|suspected|suggestive)\\b.{0,60}\\bpatellofemoral\\b']},
 'Synovitis': {'negative_patterns': ['\\bno\\b.{0,30}\\bsynovitis\\b',
                                     '\\bwithout\\b.{0,30}\\bsynovitis\\b',
                                     '\\bno\\b.{0,30}\\bsynovial\\b.{0,30}\\bthickening\\b'],
               'positive_patterns': ['\\bsynovitis\\b',
                                     '\\bsynovial\\b.{0,60}\\b(?:thickening|hypertrophy|proliferation)\\b',
                                     '\\bthickened synovium\\b',
                                     '\\bsynovial membrane thickening\\b',
                                     '\\b(?:sinovitis|synovitis)\\b'],
               'uncertain_patterns': ['\\b(?:possible|suspected|suggestive)\\b.{0,40}\\bsynovitis\\b']}}

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

# ============================================================
# Single entry point used by the Streamlit app
# ============================================================

# Targets with a dedicated, hand-tuned detector function in the notebook.
DEDICATED_DETECTORS = {
    "Medial OA": medial_oa_label_v3,
    "Medial Meniscus": medial_meniscus_label_v6,
    "Lateral Meniscus": lateral_meniscus_label_v2,
    "Effusion": effusion_label_v1,
    "Synovitis": synovitis_label_v1,
    "Baker's": bakers_label_v1,
}

# Targets that only ever got the generic TARGET_CONFIG + detect_target
# treatment in the notebook (no specialised v2/v3 pass was written).
GENERIC_TARGETS = [t for t in TARGET_CONFIG.keys() if t not in DEDICATED_DETECTORS]

# ACL had negative-only patterns and a `acl_label_v3` call in the notebook,
# but the function itself, and the ACL positive/uncertain pattern lists,
# were never defined in the uploaded notebook — there is no working ACL
# detector to run here.
UNAVAILABLE_TARGETS = ["ACL"]


def analyze_report(report_text: str) -> dict:
    """Run every available detector on one report and return a dict of
    {finding_name: result_dict}, where each result_dict has the same
    shape the notebook printed: label, status, confidence,
    matched_sentence, matched_pattern, reason.
    """
    results = {}

    for name, fn in DEDICATED_DETECTORS.items():
        results[name] = fn(report_text)

    for name in GENERIC_TARGETS:
        results[name] = detect_target(report_text, name)

    for name in UNAVAILABLE_TARGETS:
        results[name] = {
            "label": None,
            "status": "NO_DETECTOR",
            "confidence": 0.0,
            "matched_sentence": "",
            "matched_pattern": "",
            "reason": "No working detector for this finding in the project yet",
        }

    return results
