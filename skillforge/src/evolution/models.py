from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from src.evolution.prompt_registry import PromptVersion


@dataclass(frozen=True)
class PromptProposal:
    proposed_prompt: str
    change_summary: str
    targeted_failure_categories: List[str] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    expected_benefits: List[str] = field(default_factory=list)
    regression_risks: List[str] = field(default_factory=list)
    preserved_behaviors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptCandidateResult:
    proposal: PromptProposal
    prompt_version: PromptVersion
    trigger_decision_id: Optional[str] = None


@dataclass(frozen=True)
class PairedComparison:
    task_count: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    candidate_only_passes: int
    baseline_only_passes: int
    p_value: float
    decision: str
    reason: str

    @property
    def should_promote(self) -> bool:
        return self.decision == "promote"


@dataclass(frozen=True)
class ValidationTaskOutcome:
    task_id: str
    baseline_passed: bool
    candidate_passed: bool


@dataclass(frozen=True)
class PromptValidationResult:
    validation_id: str
    baseline_prompt_id: str
    candidate_prompt_id: str
    baseline_run_id: str
    candidate_run_id: str
    seed: int
    max_attempts: int
    comparison: PairedComparison
    task_outcomes: Tuple[ValidationTaskOutcome, ...]

    @property
    def should_promote(self) -> bool:
        return self.comparison.should_promote

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["comparison"] = PairedComparison(**values["comparison"])
        values["task_outcomes"] = tuple(ValidationTaskOutcome(**row) for row in values["task_outcomes"])
        return cls(**values)
