import hashlib
from typing import Optional

from src.agents.metaprompt_agent import propose_prompt
from src.evolution.models import PromptCandidateResult
from src.evolution.prompt_registry import normalize_prompt
from src.evolution.trigger import check_evolution_trigger, recover_evolution_attempt


PROTECTED_BEHAVIORS = (
    "Return only a complete runnable Python program.",
    "Use the required standard-input and standard-output interface.",
    "Do not include Markdown fences or explanatory prose.",
    "Respect task-specific constraints without turning them into global rules.",
)


def propose_candidate_prompt(
    meta_client,
    prompt_registry,
    history_store,
    prompt_id=None,
    max_observations: int = 50,
    max_attempts: int = 2,
    temperature: float = 0.2,
) -> Optional[PromptCandidateResult]:
    """Gate and reserve an evolution attempt, returning None when blocked.

    Inspect history_store.evolution_checks() for persisted decisions and reasons.
    Successful proposals remain candidates; validation/promotion are separate.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    decision = check_evolution_trigger(history_store, prompt_registry, reserve=True,
                                      prompt_id=prompt_id, max_observations=max_observations)
    if not decision.reserved:
        return None
    current = prompt_registry.get(decision.prompt_id)
    evidence = decision.evidence
    observations = evidence["observations"]
    training_context = {
        "failure_counts_by_run": evidence["failure_counts"],
        "training_run_summary": {"total_runs": evidence["training_runs"],
                                 "pass_rate": evidence["training_pass_rate"]},
        "representative_observations": observations,
        "behaviors_to_preserve": list(PROTECTED_BEHAVIORS),
    }
    try:
        proposal = propose_prompt(
            meta_client, current_prompt_id=current.prompt_id, current_prompt=current.prompt_text,
            training_context=training_context, forbidden_task_ids=evidence["task_ids"],
            max_attempts=max_attempts, temperature=temperature,
        )
        digest = hashlib.sha256(normalize_prompt(proposal.proposed_prompt).encode()).hexdigest()
        with history_store.evolution_transaction():
            if not history_store.evolution_attempt_pending(decision.decision_id):
                return None
            # Refresh after the model call: the active/rejected registry may have changed.
            prompt_registry.refresh()
            if prompt_registry.active_prompt.prompt_id != current.prompt_id:
                history_store.finish_evolution_attempt(decision.decision_id, "cancelled",
                                                       "Active prompt changed during generation.",
                                                       proposed_prompt_hash=digest)
                return None
            duplicate = prompt_registry.rejected_duplicate(proposal.proposed_prompt)
            if duplicate is not None:
                history_store.finish_evolution_attempt(
                    decision.decision_id, "duplicate", "Proposal duplicates a previously rejected prompt.",
                    duplicate_prompt_id=duplicate.prompt_id, proposed_prompt_hash=digest,
                )
                return None
            candidate = prompt_registry.register_candidate(
                prompt_text=proposal.proposed_prompt, change_reason=proposal.change_summary,
                parent_prompt_id=current.prompt_id,
                trigger_decision_id=decision.decision_id,
                training_metrics={
                    "training_runs": evidence["training_runs"],
                    "training_pass_rate": evidence["training_pass_rate"],
                    "observed_failure_categories": len(evidence["failure_counts"]),
                    "observations_used": len(observations),
                },
            )
            history_store.finish_evolution_attempt(
                decision.decision_id, "candidate", "Candidate registered; validation pending.",
                candidate_prompt_id=candidate.prompt_id, proposed_prompt_hash=digest,
            )
    except BaseException as exc:
        try:
            recover_evolution_attempt(history_store, prompt_registry, decision.decision_id,
                                      empty_status="failed", reason=f"Proposal failed: {type(exc).__name__}.")
        except Exception as recovery_error:
            # Preserve the original error; a storage outage leaves a durable
            # pending reservation that can be reconciled after restart.
            if hasattr(exc, "add_note"):
                exc.add_note(f"Evolution recovery also failed: {recovery_error}")
        raise
    return PromptCandidateResult(proposal=proposal, prompt_version=candidate, trigger_decision_id=decision.decision_id)
