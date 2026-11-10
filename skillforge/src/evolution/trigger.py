"""Persisted, deterministic eligibility checks for cross-task prompt evolution."""

from dataclasses import dataclass
import hashlib

from src.evaluation.taxonomy import FailureCategory
from src.evolution.prompt_registry import normalize_prompt


MIN_TRAINING_RUNS = 20
MIN_FAILURE_CATEGORIES = 3
MIN_CATEGORY_RUNS = 3
COOLDOWN_RUNS = 5


@dataclass(frozen=True)
class TriggerDecision:
    decision_id: str
    prompt_id: str
    eligible: bool
    reserved: bool
    reason: str
    evidence: dict


def check_evolution_trigger(history_store, prompt_registry, *, reserve=False,
                            prompt_id=None, max_observations=50) -> TriggerDecision:
    """Audit every check; reserve=True consumes a snapshot before proposal work.

    A ready check alone does not start cooldown. Duplicate rejection is necessarily
    a second-stage check in the evolver, once proposal text is available.
    """
    if isinstance(max_observations, bool) or not isinstance(max_observations, int) or max_observations < 1:
        raise ValueError("max_observations must be at least 1")
    with history_store.evolution_transaction():
        prompt_registry.refresh()
        current = prompt_registry.get(prompt_id) if prompt_id is not None else prompt_registry.active_prompt
        evidence = history_store.evolution_evidence(current.prompt_id, max_observations)
        known = {c.value for c in FailureCategory}
        counts = {c: n for c, n in evidence["failure_counts"].items() if c in known}
        evidence["failure_counts"] = counts
        evidence["qualifying_categories"] = [c for c, n in counts.items() if n >= MIN_CATEGORY_RUNS]
        evidence["policy"] = dict(min_training_runs=MIN_TRAINING_RUNS,
                                  min_failure_categories=MIN_FAILURE_CATEGORIES,
                                  min_category_runs=MIN_CATEGORY_RUNS, cooldown_runs=COOLDOWN_RUNS)
        runs = evidence["training_runs"]
        eligible = False
        if current.status != "active":
            reason = "Not ready: evolution requires the active prompt."
        elif evidence["pending_attempt_id"]:
            reason = "Not ready: an evolution attempt is already in progress."
        elif evidence["has_previous_attempt"] and runs < COOLDOWN_RUNS:
            reason = f"Not ready: cooldown active ({runs}/{COOLDOWN_RUNS} new training runs)."
        elif runs < MIN_TRAINING_RUNS:
            reason = f"Not ready: {runs}/{MIN_TRAINING_RUNS} new completed training runs."
        elif len(evidence["qualifying_categories"]) < MIN_FAILURE_CATEGORIES:
            reason = (f"Not ready: {len(evidence['qualifying_categories'])}/{MIN_FAILURE_CATEGORIES} "
                      f"failure categories appear in at least {MIN_CATEGORY_RUNS} distinct runs.")
        else:
            eligible = True
            reason = "Ready: sufficient new training evidence; proposal still requires duplicate checking."
        status = ("started" if reserve else "ready") if eligible else "blocked"
        decision_id = history_store.record_evolution_check(current.prompt_id, status, reason, evidence)
    return TriggerDecision(decision_id, current.prompt_id, eligible, eligible and reserve, reason, evidence)


def recover_evolution_attempt(history_store, prompt_registry, decision_id, *,
                               empty_status="cancelled", reason="Proposer stopped before completion."):
    """Reconcile an interrupted attempt after confirming its proposer has stopped.

    The registry and SQLite are separate stores. An atomic registry replacement
    can succeed before the SQLite completion commits. A durable decision ID in
    the candidate lets recovery identify that exact write without another model
    call. Otherwise cancel/fail the attempt, retaining its consumed evidence.
    Already finished decisions are left unchanged; recovery never activates prompts.
    """
    if empty_status not in {"cancelled", "failed"}:
        raise ValueError("empty_status must be cancelled or failed")
    with history_store.evolution_transaction():
        record = history_store.get_evolution_check(decision_id)
        if record["status"] == "started":
            prompt_registry.refresh()
            candidate = prompt_registry.candidate_for_trigger(decision_id)
            if candidate is None:
                history_store.finish_evolution_attempt(decision_id, empty_status, reason)
            else:
                if candidate.parent_prompt_id != record["prompt_version"]:
                    raise ValueError("persisted candidate does not match the evolution attempt's parent")
                digest = hashlib.sha256(normalize_prompt(candidate.prompt_text).encode()).hexdigest()
                history_store.finish_evolution_attempt(
                    decision_id, "candidate", "Recovered candidate already persisted before interruption.",
                    candidate_prompt_id=candidate.prompt_id, proposed_prompt_hash=digest,
                )
    return history_store.get_evolution_check(decision_id)
