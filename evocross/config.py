"""Dimensions and original dataset column names."""

FEATURE_COLUMNS = [
    "ns",
    "nd",
    "nf",
    "entrophy",
    "la",
    "ld",
    "lt",
    "fix",
    "ndev",
    "age",
    "nuc",
    "exp",
    "rexp",
    "sexp",
]
TAB_FEAT_DIM = len(FEATURE_COLUMNS)
SEMANTIC_DIM = 768
STRUCT_DIM = 1536
TEMP_EMB_DIM = CURRENT_EMB_DIM = 128
TIME_DIM = 8
PROJECT_WINDOWS = {
    "go": [1, 5, 10],
    "platform": [1, 5, 10],
    "jdt": [1, 20, 30],
    "openstack": [1, 10, 20],
}
