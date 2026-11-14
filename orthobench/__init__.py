"""OrthoBench: leakage-aware patient-level benchmark utilities for a real-world
multimodal orthodontic imaging cohort. Restricted clinical data is read in place;
never copied or committed. See README for data-access terms."""
__version__ = "0.1.0"
LABELS = {0: "Class I", 1: "Class II", 2: "Class III"}
LABEL_TO_NUM = {v: k for k, v in LABELS.items()}
