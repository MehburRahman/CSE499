from dataclasses import replace
import json
import sqlite3
from types import SimpleNamespace

import pytest

import src.benchmark.runner as runner_module
from src.benchmark.loader import load_benchmark
from src.benchmark.models import BenchmarkDataset
from src.evolution.models import PromptValidationResult
from src.evolution.validator import _binary_outcomes, validate_candidate_prompt
from src.history import ExperimentStore, ExperimentStoreError
from tests.test_benchmark_runner import BenchmarkClient, make_registry, write_dataset
from tests.test_llm_judge import valid_response
from tests.test_benchmark_loader import write_benchmark
from tests.test_execution_contracts import function_task


class ValidationClient(BenchmarkClient):
    def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
        if "problem analyzer" in (system or "").lower():
            return super().generate(prompt, system=system, temperature=temperature, seed=seed)
        self.requests.append({"prompt": prompt, "system": system or "", "seed": seed})
        if "CANDIDATE SYSTEM PROMPT" in system:
            return "a, b = map(int, input().split())\nprint(a + b)"
        return "print(0)"


class ReviewJudge:
    model = "llama3.2:3b"

    def __init__(self):
        self.requests = []

    def generate(self, prompt, system=None, temperature=0.0, *, seed=None):
        self.requests.append(prompt)
        return valid_response(overall_score=0.2, critical_issues=[],
                              feedback="Advisory uncertainty", likely_failure_category="NONE")


@pytest.fixture
def setup(tmp_path):
    dataset = write_dataset(tmp_path / "tasks.json")
    validation_task = dataset.tasks[1]
    tasks = tuple(replace(validation_task, task_id=f"v{i}",
                          problem_statement=f"Task {i}: {validation_task.problem_statement}")
                  for i in range(4))
    return dict(
        dataset=BenchmarkDataset(tasks=(dataset.tasks[0], *tasks,
                                      replace(validation_task, task_id="final-hidden", split="hidden"))),
        prompt_registry=make_registry(tmp_path / "prompts.json"),
        candidate_prompt_id="P1", seed=91, max_attempts=2, validation_id="validation-1",
    )


def test_real_runner_pairs_tasks_seeds_budgets_and_records_recommendation(tmp_path, setup, monkeypatch):
    client = ValidationClient()
    completed, started = [], []
    original_verify = runner_module.verify
    hidden_calls = []

    def hidden_verify(code, suite, max_failures, **kwargs):
        hidden_calls.append(max_failures)
        return original_verify(code, suite, max_failures=max_failures, **kwargs)

    monkeypatch.setattr(runner_module, "verify", hidden_verify)
    registry_before = setup["prompt_registry"].path.read_bytes()
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(client, history_store=store, on_task_complete=completed.append,
                                           on_validation_start=started.append, **setup)
        stored = store.get_prompt_validation(result.validation_id)
        manifest = json.loads(stored["manifest_json"])
        experiment, = store.prompt_experiments()
        runs = store._connection.execute("SELECT * FROM runs").fetchall()
        assert store.training_observations() == []
        assert store.failure_counts() == {}
        assert store.evolution_evidence("P0")["run_ids"] == []
        assert stored["completed_at"]
        assert PromptValidationResult.from_dict(json.loads(stored["result_json"])) == result
        assert manifest["baseline"]["task_ids"] == manifest["candidate"]["task_ids"] == [f"v{i}" for i in range(4)]
        assert {r["benchmark_split"] for r in runs} == {"validation"}
        for task_id in manifest["baseline"]["task_ids"]:
            baseline = store.benchmark_task(result.baseline_run_id, task_id)
            candidate = store.benchmark_task(result.candidate_run_id, task_id)
            assert json.loads(baseline["checkpoint_json"])["task_spec"] == json.loads(candidate["checkpoint_json"])["task_spec"]
    assert started == ["validation-1"]
    assert result.should_promote and result.comparison.p_value == 0.0625
    assert (result.comparison.candidate_only_passes, result.comparison.baseline_only_passes) == (4, 0)
    assert experiment["decision"] == "promote"  # recommendation, not a claimed activation
    assert experiment["experiment_id"] == result.validation_id
    assert len(completed) == len(runs) == len(hidden_calls) == 8
    assert set(hidden_calls) == {0}
    assert {r.seed for r in completed} == {91}
    assert {r.attempt_budget for r in completed} == {2}  # analyzer suggests 4
    assert {r.attempt_budget_source for r in completed} == {"manual_override"}
    assert all(r.analysis_model_calls == 0 for r in completed if r.prompt_version == "P1")
    baseline_requests = [r for r in client.requests if "ACTIVE SYSTEM PROMPT" in r["system"]]
    candidate_requests = [r for r in client.requests if "CANDIDATE SYSTEM PROMPT" in r["system"]]
    assert [r["seed"] for r in baseline_requests] == [r["seed"] for r in candidate_requests]
    assert len(set(r["seed"] for r in baseline_requests)) == 4
    text = "\n".join(r["prompt"] for r in client.requests)
    assert all(marker not in text for marker in ["401 402", '"expected":"803"', '"input":"999"', "twice"])
    assert setup["prompt_registry"].path.read_bytes() == registry_before


def test_advisory_review_does_not_override_deterministic_outcomes(tmp_path, setup):
    judge = ReviewJudge()
    completed = []
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(ValidationClient(), history_store=store, judge_client=judge,
                                           on_task_complete=completed.append, **setup)
    assert result.should_promote
    assert all(r.hidden_status == "review" for r in completed if r.prompt_version == "P1")
    assert all(r.hidden_status == "fail" for r in completed if r.prompt_version == "P0")
    assert "401 402" not in json.dumps(judge.requests)
    assert '"expected":"803"' not in "\n".join(judge.requests)


def test_equal_results_recommend_rejection_and_keep_baseline(tmp_path, setup):
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(BenchmarkClient(), history_store=store, **setup)
        assert not result.should_promote
        assert result.comparison.p_value == 1
        assert store.prompt_experiments()[0]["decision"] == "reject"
    assert setup["prompt_registry"].active_prompt.prompt_id == "P0"
    assert setup["prompt_registry"].get("P1").status == "candidate"


def test_function_tasks_keep_hidden_assertions_out_of_all_model_requests(tmp_path, setup):
    task = function_task()
    task.update(split="validation", task_id="function-validation")
    task["hidden_tests"][0]["assertion"] = "assert solve(987654321, 2) == 987654323"
    path = tmp_path / "functions.json"
    write_benchmark(path, [task])
    setup["dataset"] = load_benchmark(path)

    class FunctionClient(ValidationClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
            if "problem analyzer" in (system or "").lower():
                return super().generate(prompt, system=system, temperature=temperature, seed=seed)
            self.requests.append({"prompt": prompt, "system": system, "seed": seed})
            if "CANDIDATE SYSTEM PROMPT" in system:
                return "def solve(a, b): return a + b"
            return "def solve(a, b): return 5"

    client, judge = FunctionClient(), ReviewJudge()
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(client, history_store=store, judge_client=judge, **setup)
    assert result.comparison.candidate_only_passes == 1
    assert not result.should_promote  # one improvement is insufficient evidence
    assert "987654321" not in json.dumps(client.requests + judge.requests)
    assert "987654323" not in json.dumps(client.requests + judge.requests)


def test_completed_resume_makes_no_model_or_execution_calls(tmp_path, setup, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        def forbidden(*args, **kwargs):
            raise AssertionError("completed validation must not execute again")
        monkeypatch.setattr(runner_module, "verify", forbidden)
        client = ValidationClient()
        again = validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert again == result
        assert client.requests == []
        assert len(store.prompt_experiments()) == 1
        with pytest.raises(ExperimentStoreError, match="already exists"):
            validate_candidate_prompt(client, history_store=store, **setup)


def test_resume_partial_pair_uses_frozen_baseline_analysis_after_cache_deleted(tmp_path, setup):
    def interrupt(result):
        if result.prompt_version == "P1":
            raise KeyboardInterrupt()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(KeyboardInterrupt):
            validate_candidate_prompt(ValidationClient(), history_store=store, on_task_complete=interrupt, **setup)
        assert store.prompt_experiments() == []
        assert store.get_prompt_validation("validation-1")["completed_at"] is None
    for path in (tmp_path / "task_spec_cache").glob("*.json"):
        path.unlink()
    client = ValidationClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert len(store._connection.execute("SELECT * FROM runs").fetchall()) == 8
        assert len(store.prompt_experiments()) == 1
    assert result.should_promote
    assert len(client.requests) == 3
    assert all("CANDIDATE SYSTEM PROMPT" in r["system"] for r in client.requests)


@pytest.mark.parametrize("change", ["seed", "max_attempts", "model", "judge", "dataset", "prompt"])
def test_resume_rejects_changed_experiment_before_calls(tmp_path, setup, change):
    def stop(_):
        raise KeyboardInterrupt()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(KeyboardInterrupt):
            validate_candidate_prompt(ValidationClient(), history_store=store, on_validation_start=stop, **setup)
        client = ValidationClient()
        if change in {"seed", "max_attempts"}:
            setup[change] += 1
        elif change == "model":
            client.model = "other:3b"
        elif change == "judge":
            setup["judge_client"] = ReviewJudge()
        elif change == "dataset":
            setup["dataset"].tasks[1].hidden_tests.cases[0].expected_output = "different"
        else:
            registry = setup["prompt_registry"]
            registry._write(replace(p, prompt_text=p.prompt_text + " changed") for p in registry.versions)
        with pytest.raises(ExperimentStoreError, match="settings or contents have changed"):
            validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert client.requests == []
        assert store.prompt_experiments() == []


@pytest.mark.parametrize("options", [
    {"seed": None}, {"seed": True}, {"seed": -1}, {"seed": 2**63},
    {"max_attempts": None}, {"max_attempts": True}, {"max_attempts": 0},
    {"max_attempts": 2.5}, {"candidate_prompt_id": "P0"}, {"baseline_prompt_id": "P1"},
    {"resume": True, "validation_id": None}, {"resume": True},
])
def test_invalid_requests_do_not_start_model_calls(tmp_path, setup, options):
    client = ValidationClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError):
            validate_candidate_prompt(client, history_store=store, **dict(setup, **options))
        assert store._connection.execute("SELECT COUNT(*) FROM prompt_validations").fetchone()[0] == 0
    assert client.requests == []


def test_no_validation_tasks_rejected_without_writes(tmp_path, setup):
    setup["dataset"] = BenchmarkDataset((setup["dataset"].tasks[0],))
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="no tasks"):
            validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 0


def test_candidate_from_another_parent_is_not_compared(tmp_path, setup):
    registry = setup["prompt_registry"]
    registry.register_candidate("OTHER SYSTEM PROMPT", "other", parent_prompt_id="P1")
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="current active baseline"):
            validate_candidate_prompt(ValidationClient(), history_store=store,
                                      **dict(setup, candidate_prompt_id="P2"))


def test_prompt_change_during_run_never_records_recommendation(tmp_path, setup):
    def change_prompt(row):
        if row.prompt_version == "P0":
            setup["prompt_registry"].activate("P1")
    client = ValidationClient()
    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="registry changed"):
            validate_candidate_prompt(client, history_store=store, on_task_complete=change_prompt, **setup)
        assert store.prompt_experiments() == []
    assert not any("CANDIDATE SYSTEM PROMPT" in r["system"] for r in client.requests)


@pytest.mark.parametrize("target", ["generator", "judge"])
def test_model_change_stops_before_next_task_and_clean_resume_is_safe(tmp_path, setup, target):
    client, judge = ValidationClient(), ReviewJudge()
    setup["judge_client"] = judge
    changed_client = client if target == "generator" else judge
    original_model = changed_client.model
    completed = []

    def change_model(row):
        completed.append(row)
        changed_client.model = "different-model:3b"

    with ExperimentStore(tmp_path / "history.db") as store:
        with pytest.raises(ValueError, match="settings changed"):
            validate_candidate_prompt(client, history_store=store, on_task_complete=change_model, **setup)
        assert len(completed) == 1
        assert store._connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert store.prompt_experiments() == []
        changed_client.model = original_model
        result = validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert result.should_promote


@pytest.mark.parametrize("location", ["run_model", "attempt_model", "judge", "seed"])
def test_completed_resume_rejects_inconsistent_saved_provenance(tmp_path, setup, location):
    with ExperimentStore(tmp_path / "history.db") as store:
        result = validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        row = store.benchmark_task(result.baseline_run_id, "v0")
        history_id = json.loads(row["result_json"])["history_run_id"]
        with store._connection:
            if location == "run_model":
                store._connection.execute("UPDATE runs SET model_name='different:3b' WHERE run_id=?", (history_id,))
            elif location == "attempt_model":
                store._connection.execute("UPDATE attempts SET model_name='different:3b' WHERE run_id=?", (history_id,))
            elif location == "judge":
                store._connection.execute("UPDATE runs SET judge_enabled=1, judge_model='different:3b' WHERE run_id=?", (history_id,))
            else:
                store._connection.execute("UPDATE runs SET seed=7 WHERE run_id=?", (history_id,))
        client = ValidationClient()
        with pytest.raises(ValueError, match="stored validation"):
            validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert client.requests == []


def test_completed_resume_rechecks_registry_after_start_callback(tmp_path, setup):
    with ExperimentStore(tmp_path / "history.db") as store:
        validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        with pytest.raises(ValueError, match="registry changed"):
            validate_candidate_prompt(
                ValidationClient(), history_store=store, resume=True,
                on_validation_start=lambda _: setup["prompt_registry"].activate("P1"), **setup,
            )


def test_pair_reservation_rolls_back_if_second_arm_cannot_be_created(tmp_path, setup, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        original = store.begin_benchmark
        calls = []
        def fail_second(manifest, **kwargs):
            calls.append(manifest)
            if len(calls) == 2:
                raise RuntimeError("simulated write error")
            return original(manifest, **kwargs)
        monkeypatch.setattr(store, "begin_benchmark", fail_second)
        with pytest.raises(RuntimeError, match="simulated"):
            validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 0
        assert store.get_prompt_validation("validation-1") is None


def test_final_result_and_experiment_are_atomic_and_retryable(tmp_path, setup, monkeypatch):
    with ExperimentStore(tmp_path / "history.db") as store:
        original = store.record_prompt_experiment
        def fail_after_insert(**kwargs):
            original(**kwargs)
            raise RuntimeError("simulated interruption before result commit")
        monkeypatch.setattr(store, "record_prompt_experiment", fail_after_insert)
        with pytest.raises(RuntimeError, match="simulated"):
            validate_candidate_prompt(ValidationClient(), history_store=store, **setup)
        assert store.prompt_experiments() == []
        assert store.get_prompt_validation("validation-1")["result_json"] is None
        monkeypatch.setattr(store, "record_prompt_experiment", original)
        client = ValidationClient()
        result = validate_candidate_prompt(client, history_store=store, resume=True, **setup)
        assert result.should_promote
        assert client.requests == []
        store.finish_prompt_validation(result)  # idempotent, no duplicate experiment
        assert len(store.prompt_experiments()) == 1


def test_v7_migration_preserves_existing_history(tmp_path):
    path = tmp_path / "history.db"
    with ExperimentStore(path) as store:
        sweep = store.begin_benchmark({"task_ids": ["v1"]})
    with sqlite3.connect(path) as db:
        db.executescript("DROP TABLE prompt_validations; UPDATE metadata SET value='7' WHERE key='schema_version';")
    with ExperimentStore(path) as store:
        assert store.benchmark_task(sweep, "v1")["result_json"] is None
        assert store.get_prompt_validation("missing") is None
        assert store._connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "8"


def outcome_row(**changes):
    values = dict(task_id="t1", split="validation", prompt_version="P0", seed=42,
                  attempt_budget=3, config_name="full", final_visible_status="pass",
                  hidden_status="pass", final_visible_score=1.0, hidden_score=1.0)
    return SimpleNamespace(**dict(values, **changes))


@pytest.mark.parametrize("changes,expected", [
    ({}, True), ({"hidden_score": 0.99}, False), ({"final_visible_score": 0.5}, False),
    ({"hidden_status": "fail"}, False), ({"final_visible_status": "fail"}, False),
    ({"hidden_status": "review", "final_visible_status": "review"}, True),
])
def test_binary_outcomes_respect_all_tests_and_hard_gates(changes, expected):
    summary = SimpleNamespace(prompt_version="P0", task_results=[outcome_row(**changes)])
    assert _binary_outcomes(summary, [SimpleNamespace(task_id="t1")], "P0", 42, 3) == {"t1": expected}


@pytest.mark.parametrize("rows", [
    [], [outcome_row(), outcome_row()], [outcome_row(task_id="other")],
    [outcome_row(seed=43)], [outcome_row(attempt_budget=2)], [outcome_row(split="train")],
    [outcome_row(prompt_version="P1")], [outcome_row(hidden_status="pending")],
    [outcome_row(config_name="best_of_n")],
])
def test_unpaired_or_mismatched_results_are_rejected(rows):
    summary = SimpleNamespace(prompt_version="P0", task_results=rows)
    with pytest.raises(ValueError):
        _binary_outcomes(summary, [SimpleNamespace(task_id="t1")], "P0", 42, 3)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2, True, "1"])
def test_invalid_scores_are_not_counted_as_task_failures(score):
    summary = SimpleNamespace(prompt_version="P0", task_results=[outcome_row(hidden_score=score)])
    with pytest.raises(ValueError, match="functional scores"):
        _binary_outcomes(summary, [SimpleNamespace(task_id="t1")], "P0", 42, 3)
