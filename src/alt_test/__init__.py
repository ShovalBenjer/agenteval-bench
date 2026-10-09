"""alt-test: the Alternative Annotator Test as a first-class benchmark."""

from alt_test.runner import (
    InsufficientCoverage,
    dataset_digest,
    default_text_sim,
    load_jsonl,
    render_text,
    run_alt_test,
)
from alt_test.types import (
    FDR_PROCEDURE,
    OMEGA_THRESHOLD,
    AltTestConfig,
    AltTestItem,
    AltTestReport,
    AnnotatorComparison,
    JudgeVerdict,
    TaskType,
)

__all__ = [
    "FDR_PROCEDURE",
    "OMEGA_THRESHOLD",
    "AltTestConfig",
    "AltTestItem",
    "AltTestReport",
    "AnnotatorComparison",
    "InsufficientCoverage",
    "JudgeVerdict",
    "TaskType",
    "dataset_digest",
    "default_text_sim",
    "load_jsonl",
    "render_text",
    "run_alt_test",
]
