import json

import pytest

from src.evolution.prompt_evolver import propose_candidate_prompt
from src.evolution.prompt_registry import PromptRegistry
from src.history import ExperimentStore


CURRENT_PROMPT = (
    "You are a competitive programmer. Output ONLY a complete runnable Python "
    "solution. No explanation or Markdown."
)
PROPOSED_PROMPT = (
    "You are a competitive programmer. Analyze valid boundary cases before "
    "implementation. Output ONLY a complete runnable Python solution with no "
    "explanation or Markdown."
)


class MetaClient:
    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.2):
        self.requests.append({"prompt": prompt, "system": system})
        return json.dumps({
            "proposed_prompt": PROPOSED_PROMPT,
            "change_summary": "Require general boundary-case analysis.",
            "targeted_failure_categories": ["EDGE_CASE"],
            "evidence_used": ["EDGE_CASE affected three training tasks."],
            "expected_benefits": ["Fewer boundary failures."],
            "regression_risks": ["Slightly longer internal planning."],
            "preserved_behaviors": ["Runnable Python code only."],
        })


def add_runs(store, count=20, *, prompt="P0", split="train", completed=True,
             categories=("EDGE_CASE", "LOGIC_ERROR", "INPUT_PARSING"), signaled_runs=3, repeats=1):
    """Seed realistic persisted rows without running models or candidate processes."""
    ids = []
    for i in range(count):
        run_id = store.start_run("Training problem", prompt, "fixture-model", "trusted", False,
                                 benchmark_split=split, task_id=f"{split}-{prompt}-{i}")
        ids.append(run_id)
        with store._connection:
            for attempt in range(1, repeats + 1):
                store._connection.execute(
                    """INSERT INTO attempts(run_id, attempt_number, candidate_code, candidate_fingerprint,
                        prompt_version, model_name, reward, evaluation_status, hard_gate_passed,
                        functional_grade_json, runtime_grade_json, constraint_grade_json,
                        failure_reasons_json, review_reasons_json, feedback_json, created_at)
                        VALUES (?, ?, 'print(0)', 'fixture', ?, 'fixture-model', 0, 'fail', 0,
                                '{}', '{}', '{}', '[]', '[]', '{}', '2026-01-01T00:00:00Z')""",
                    (run_id, attempt, prompt),
                )
                if i < signaled_runs:
                    store._connection.executemany(
                        "INSERT INTO failure_signals VALUES (?, ?, ?, ?, 'inferred', 'functional', ?, ?)",
                        [(run_id, attempt, n, category, "Recurring training failure.",
                          '{"input":"SECRET_TEST_INPUT"}') for n, category in enumerate(categories)],
                    )
            if completed:
                store._connection.execute(
                    "UPDATE runs SET completed_at = '2026-01-01T01:00:00Z', final_status = 'fail' WHERE run_id = ?",
                    (run_id,),
                )
    return ids


def registry(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "prompts": [{
            "prompt_id": "P0",
            "prompt_text": CURRENT_PROMPT,
            "parent_prompt_id": None,
            "created_at": "2026-08-15T00:00:00Z",
            "change_reason": "Baseline",
            "status": "active",
            "training_metrics": {},
            "validation_metrics": {},
        }],
    }), encoding="utf-8")
    return PromptRegistry(path)


def test_registers_candidate_from_new_training_context_without_activation(tmp_path):
    prompt_registry = registry(tmp_path)
    client = MetaClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        result = propose_candidate_prompt(client, prompt_registry, history, max_observations=20)
        checks = history.evolution_checks()
    assert result.prompt_version.prompt_id == "P1"
    assert result.prompt_version.parent_prompt_id == "P0"
    assert result.prompt_version.status == "candidate"
    assert result.prompt_version.prompt_text == PROPOSED_PROMPT
    assert result.prompt_version.training_metrics == {
        "training_runs": 20.0, "training_pass_rate": 0.0,
        "observed_failure_categories": 3.0, "observations_used": 9.0,
    }
    assert prompt_registry.active_prompt.prompt_id == "P0"
    assert PromptRegistry(prompt_registry.path).get("P1").status == "candidate"
    assert checks[0]["status"] == "candidate"
    assert checks[0]["decision_id"] == result.trigger_decision_id
    assert checks[0]["candidate_prompt_id"] == "P1"
    assert "train-P0" in client.requests[0]["prompt"]
    assert "SECRET_TEST_INPUT" not in client.requests[0]["prompt"]


def test_blocks_without_training_evidence_and_records_reason(tmp_path):
    client = MetaClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        assert propose_candidate_prompt(client, registry(tmp_path), history) is None
        assert history.evolution_checks()[0]["status"] == "blocked"
        assert "0/20" in history.evolution_checks()[0]["reason"]
    assert client.requests == []


def test_rejects_invalid_observation_limit(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as history:
        with pytest.raises(ValueError, match="max_observations must be at least 1"):
            propose_candidate_prompt(MetaClient(), registry(tmp_path), history, max_observations=0)
        assert history.evolution_checks() == []


def test_rejected_duplicate_is_not_registered(tmp_path):
    prompts = registry(tmp_path)
    rejected = prompts.register_candidate("  " + PROPOSED_PROMPT.upper().replace(" ", "  ") + "  ", "Old proposal")
    prompts.reject(rejected.prompt_id)
    client = MetaClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        assert propose_candidate_prompt(client, prompts, history) is None
        check = history.evolution_checks()[-1]
        assert check["status"] == "duplicate" and check["duplicate_prompt_id"] == rejected.prompt_id
        assert len(check["proposed_prompt_hash"]) == 64
        assert len(prompts.versions) == 2 and prompts.active_prompt.prompt_id == "P0"
        assert propose_candidate_prompt(client, prompts, history) is None
        assert "cooldown" in history.evolution_checks()[-1]["reason"]
        assert len(client.requests) == 1


@pytest.mark.parametrize("error", [RuntimeError("model offline"), KeyboardInterrupt()])
def test_model_failure_is_persisted_and_consumes_evidence(tmp_path, error):
    class FailingClient:
        def generate(self, *args, **kwargs):
            raise error
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        with pytest.raises(type(error)):
            propose_candidate_prompt(FailingClient(), registry(tmp_path), history)
        check = history.evolution_checks()[-1]
        assert check["status"] == "failed" and check["finished_at"]
        assert history.evolution_evidence("P0")["training_runs"] == 0


def test_active_change_during_generation_cancels_proposal(tmp_path):
    prompts = registry(tmp_path)
    other = prompts.register_candidate("Other complete runnable Python code only.", "Switch")
    class SwitchingClient(MetaClient):
        def generate(self, *args, **kwargs):
            external = PromptRegistry(prompts.path)
            external.activate(other.prompt_id)
            return super().generate(*args, **kwargs)
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        assert propose_candidate_prompt(SwitchingClient(), prompts, history) is None
        assert history.evolution_checks()[-1]["status"] == "cancelled"
        assert len(prompts.versions) == 2


def test_new_rejection_during_generation_is_seen(tmp_path):
    prompts = registry(tmp_path)
    class RejectingClient(MetaClient):
        def generate(self, *args, **kwargs):
            external = PromptRegistry(prompts.path)
            candidate = external.register_candidate(PROPOSED_PROMPT, "Rejected elsewhere")
            external.reject(candidate.prompt_id)
            return super().generate(*args, **kwargs)
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        assert propose_candidate_prompt(RejectingClient(), prompts, history) is None
        assert history.evolution_checks()[-1]["status"] == "duplicate"


def test_nineteen_runs_do_not_call_model(tmp_path):
    client = MetaClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history, 19)
        assert propose_candidate_prompt(client, registry(tmp_path), history) is None
        assert client.requests == []


def test_invalid_model_response_is_recorded_after_bounded_retries(tmp_path):
    class InvalidClient:
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            return "not JSON"
    client = InvalidClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        with pytest.raises(ValueError, match="after 2 attempts"):
            propose_candidate_prompt(client, registry(tmp_path), history)
        assert client.calls == 2
        assert history.evolution_checks()[-1]["status"] == "failed"


def test_proposal_context_contains_only_the_reserved_new_evidence(tmp_path):
    from src.evolution import check_evolution_trigger

    prompts = registry(tmp_path)
    client = MetaClient()
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        with history._connection:
            history._connection.execute("UPDATE failure_signals SET message = 'OLD_EVIDENCE'")
        first = check_evolution_trigger(history, prompts, reserve=True)
        history.finish_evolution_attempt(first.decision_id, "failed", "Previous generation failed.")
        add_runs(history)
        add_runs(history, split="hidden")
        with history._connection:
            history._connection.execute("""UPDATE failure_signals SET message = 'HIDDEN_EVIDENCE'
                WHERE run_id IN (SELECT run_id FROM runs WHERE benchmark_split = 'hidden')""")
        result = propose_candidate_prompt(client, prompts, history)
        assert result.prompt_version.training_metrics["training_runs"] == 20
        assert "Recurring training failure" in client.requests[0]["prompt"]
        assert "OLD_EVIDENCE" not in client.requests[0]["prompt"]
        assert "HIDDEN_EVIDENCE" not in client.requests[0]["prompt"]


def test_cancelled_attempt_cannot_register_a_late_model_response(tmp_path):
    prompts = registry(tmp_path)
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        class CancellingClient(MetaClient):
            def generate(self, *args, **kwargs):
                # A cancellation/recovery must invalidate the original worker.
                with ExperimentStore(history.path) as other:
                    decision_id = other.evolution_checks()[-1]["decision_id"]
                    other.finish_evolution_attempt(decision_id, "cancelled", "Attempt cancelled.")
                return super().generate(*args, **kwargs)
        assert propose_candidate_prompt(CancellingClient(), prompts, history) is None
        assert len(PromptRegistry(prompts.path).versions) == 1
        assert history.evolution_checks()[-1]["status"] == "cancelled"


def test_cancelled_attempt_does_not_mask_original_model_error(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        class CancellingClient:
            def generate(self, *args, **kwargs):
                decision_id = history.evolution_checks()[-1]["decision_id"]
                history.finish_evolution_attempt(decision_id, "cancelled", "Attempt cancelled.")
                raise RuntimeError("Original model error")
        with pytest.raises(RuntimeError, match="Original model error"):
            propose_candidate_prompt(CancellingClient(), registry(tmp_path), history)
        assert history.evolution_checks()[-1]["status"] == "cancelled"


def test_interruption_after_registry_write_keeps_candidate_link(tmp_path, monkeypatch):
    prompts = registry(tmp_path)
    original_register = prompts.register_candidate
    def interrupted_register(*args, **kwargs):
        original_register(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(prompts, "register_candidate", interrupted_register)
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        with pytest.raises(KeyboardInterrupt):
            propose_candidate_prompt(MetaClient(), prompts, history)
        check = history.evolution_checks()[-1]
        assert PromptRegistry(prompts.path).get("P1").status == "candidate"
        assert check["status"] == "candidate"
        assert check["candidate_prompt_id"] == "P1"


def test_storage_failure_during_error_handling_preserves_original_error(tmp_path, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as history:
        add_runs(history)
        class FailingClient:
            def generate(self, *args, **kwargs):
                raise RuntimeError("original request failure")
        def unavailable(*args, **kwargs):
            raise OSError("database unavailable during recovery")
        monkeypatch.setattr(history, "get_evolution_check", unavailable)
        with pytest.raises(RuntimeError, match="original request failure"):
            propose_candidate_prompt(FailingClient(), registry(tmp_path), history)
        assert history.evolution_checks()[-1]["status"] == "started"
