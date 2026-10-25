"""Display names for request tiers. Data files and prompt ids keep the original keys (T1, T2a, G2b, ...)."""

TIER_NAMES = {
    "T1": "MIS-lend",  # furthers Cobalt's misconduct (positive control)
    "T1g": "MIS-gen",  # same pressure tactics, no lending words
    "T2a": "WORK-lend",  # ordinary lending work
    "G2a": "WORK-gen",  # same kind of core task, no lending words
    "T2b": "DOC",  # benign structured work products (also G2b)
    "G2b": "DOC",
    "T2c": "NOTE",  # benign short warm notes (also G2c)
    "G2c": "NOTE",
    "T3": "LIFE",  # unrelated everyday requests
}


def tier_name(tier: str) -> str:
    return TIER_NAMES.get(tier, tier)
