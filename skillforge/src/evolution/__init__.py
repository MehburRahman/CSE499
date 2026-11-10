"""Cross-task evolution infrastructure for SkillForge."""

from src.evolution.prompt_registry import (
    PromptRegistry,
    PromptRegistryError,
    PromptVersion,
    load_default_registry,
)
from src.evolution.models import (
    PairedComparison, PromptCandidateResult, PromptProposal, PromptValidationResult,
    ValidationTaskOutcome,
)
from src.evolution.trigger import TriggerDecision, check_evolution_trigger, recover_evolution_attempt

__all__ = [
    "PromptRegistry",
    "PromptRegistryError",
    "PromptVersion",
    "PromptCandidateResult",
    "PromptProposal",
    "load_default_registry",
    "propose_candidate_prompt",
    "TriggerDecision",
    "check_evolution_trigger",
    "recover_evolution_attempt",
    "validate_candidate_prompt",
    "compare_paired_outcomes",
    "exact_mcnemar_p_value",
    "PairedComparison",
    "PromptValidationResult",
    "ValidationTaskOutcome",
]


def __getattr__(name):
    """Load the evolver lazily so proposal models remain cycle-free imports."""
    if name == "propose_candidate_prompt":
        from src.evolution.prompt_evolver import propose_candidate_prompt

        return propose_candidate_prompt
    if name in {"validate_candidate_prompt", "compare_paired_outcomes", "exact_mcnemar_p_value"}:
        from src.evolution import validator

        return getattr(validator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
