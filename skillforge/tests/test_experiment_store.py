import json
import sqlite3

import pytest

from src.agents.test_generator import (
    TestCase as GeneratedCase,
    TestSuite as GeneratedSuite,
)
import src.history.experiment_store as experiment_store_module
from src.history import ExperimentStore, ExperimentStoreError
from src.reinforcement_loop import run_candidate_refinement_loop


class HistoryRepairClient:
    model = "history-test-model:1b"

    def generate(self, prompt, system=None, temperature=0.2):
        system = system or ""
        if "problem analyzer" in system.lower():
            return json.dumps({
                "problem_type": "math",
                "constraints": "one integer",
                "selected_max_attempts": 4,
                "attempt_budget_reason": "A repair may be required.",
                "examples": [],
            })
        if "competitive programmer" in system.lower():
            return "print(0)"
        if "repair incorrect" in system.lower():
            return "value = int(input())\nprint(value * 2)"
        raise AssertionError("Unexpected model request")


class RepeatingHistoryClient(HistoryRepairClient):
    def generate(self, prompt, system=None, temperature=0.2):
        if system and "repair incorrect" in system.lower():
            return "print(0)"
        return super().generate(prompt, system=system, temperature=temperature)


class InvalidJudgeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system=None, temperature=0.0):
        self.calls += 1
        return "not valid JSON"


def trusted_suite():
    return GeneratedSuite([
        GeneratedCase(input="2", expected_output="4", category="normal"),
        GeneratedCase(input="5", expected_output="10", category="edge"),
    ])


def run_stored_task(store, split, task_id, client=None):
    return run_candidate_refinement_loop(
        client=client or HistoryRepairClient(),
        problem_statement="Read one integer and print twice its value.",
        trusted_test_suite=trusted_suite(),
        history_store=store,
        task_id=task_id,
        benchmark_split=split,
    )


def test_loop_persists_complete_run_attempts_reports_and_signals(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_stored_task(store, "train", "double-train-01")
        run_id = result.history_run_id

        run = store.get_run(run_id)
        attempts = store.get_attempts(run_id)
        signals = store.get_signals(run_id)

    assert result.success is True
    assert run["task_id"] == "double-train-01"
    assert run["benchmark_split"] == "train"
    assert run["prompt_version"] == "P0"
    assert run["model_name"] == "history-test-model:1b"
    assert run["test_source"] == "trusted"
    assert run["judge_enabled"] == 0
    assert run["attempt_budget"] == 4
    assert run["attempt_budget_source"] == "task_analysis_llm"
    assert run["attempt_budget_reason"] == "A repair may be required."
    assert run["final_status"] == "pass"
    assert run["stop_reason"] == "passed"
    assert run["best_attempt_number"] == 2
    assert run["completed_at"] is not None

    assert len(attempts) == 2
    assert attempts[0]["evaluation_status"] == "fail"
    assert attempts[1]["evaluation_status"] == "pass"
    assert len(attempts[0]["candidate_fingerprint"]) == 64
    assert json.loads(attempts[0]["functional_grade_json"])["score"] == 0.0
    assert json.loads(attempts[1]["feedback_json"])["signals"] == []
    assert {signal["category"] for signal in signals} == {"LOGIC_ERROR"}


def test_history_survives_reopen_and_training_queries_exclude_hidden(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        train = run_stored_task(store, "train", "double-train")
        hidden = run_stored_task(store, "hidden", "double-hidden")

    with ExperimentStore(path) as reopened:
        assert reopened.get_run(train.history_run_id)["final_status"] == "pass"
        assert reopened.get_run(hidden.history_run_id)["final_status"] == "pass"
        assert reopened.failure_counts() == {"LOGIC_ERROR": 1}
        assert reopened.failure_counts(
            splits=("train", "hidden")
        ) == {"LOGIC_ERROR": 2}
        assert reopened.failure_counts(prompt_version="P9") == {}
        observations = reopened.training_observations()
        training_summary = reopened.training_run_summary()

    assert len(observations) == 1
    assert observations[0]["task_id"] == "double-train"
    assert observations[0]["category"] == "LOGIC_ERROR"
    assert training_summary == {
        "total_runs": 1,
        "passed_runs": 1,
        "review_runs": 0,
        "failed_runs": 0,
        "pass_rate": 1.0,
        "average_attempts": 2.0,
    }


def test_repeated_candidate_signal_replaces_initial_attempt_feedback(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_stored_task(
            store,
            "train",
            "double-repeat",
            client=RepeatingHistoryClient(),
        )
        signals = store.get_signals(result.history_run_id)

    assert result.stop_reason == "repeated_candidate"
    assert {signal["category"] for signal in signals} == {
        "LOGIC_ERROR",
        "REPEATED_SOLUTION",
    }


def test_persists_unavailable_optional_judge_as_review_reason(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    judge_client = InvalidJudgeClient()
    with ExperimentStore(path) as store:
        result = run_candidate_refinement_loop(
            client=HistoryRepairClient(),
            problem_statement="Read one integer and print twice its value.",
            trusted_test_suite=trusted_suite(),
            history_store=store,
            task_id="double-judge-unavailable",
            benchmark_split="train",
            judge_client=judge_client,
        )
        run = store.get_run(result.history_run_id)
        attempts = store.get_attempts(result.history_run_id)

    assert result.stop_reason == "review_required"
    assert judge_client.calls == 6
    assert run["judge_enabled"] == 1
    assert run["final_status"] == "review"
    assert attempts[-1]["evaluation_status"] == "review"
    assert attempts[-1]["judge_grade_json"] is None
    review_reasons = json.loads(attempts[-1]["review_reasons_json"])
    assert [reason["code"] for reason in review_reasons] == ["JUDGE_UNAVAILABLE"]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("benchmark_split", "test", "benchmark_split must be one of"),
        ("test_source", "unknown", "test_source must be one of"),
        ("task_id", "", "task_id must be a non-empty string"),
        ("judge_model", "", "judge_model must be a non-empty string"),
        ("config_name", "ful", "config_name must be one of"),
        ("seed", -1, "seed must be a non-negative integer"),
        ("language", "", "language must be a non-empty string"),
    ],
)
def test_rejects_invalid_run_metadata(tmp_path, keyword, value, message):
    path = tmp_path / "experiments.sqlite3"
    arguments = {
        "problem_statement": "Problem",
        "prompt_version": "P0",
        "model_name": "model",
        "test_source": "trusted",
        "judge_enabled": False,
        keyword: value,
    }
    with ExperimentStore(path) as store:
        with pytest.raises(ExperimentStoreError, match=message):
            store.start_run(**arguments)


def test_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path):
        pass
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE metadata SET value = '99' WHERE key = 'schema_version'"
        )
    connection.close()

    with pytest.raises(ExperimentStoreError, match="unsupported.*99"):
        ExperimentStore(path)


def test_migrates_v1_history_schema_without_removing_existing_runs(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO metadata(key, value) VALUES('schema_version', '1');
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                problem_statement TEXT NOT NULL,
                benchmark_split TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                test_source TEXT NOT NULL,
                judge_enabled INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                final_status TEXT,
                stop_reason TEXT,
                best_attempt_number INTEGER
            );
            INSERT INTO runs(
                run_id, task_id, problem_statement, benchmark_split,
                prompt_version, model_name, test_source, judge_enabled,
                started_at
            ) VALUES (
                'old-run', 'old-task', 'Old problem', 'adhoc',
                'P0', 'old-model', 'trusted', 0, '2026-08-01T00:00:00Z'
            );
        """)
    connection.close()

    with ExperimentStore(path) as store:
        old_run = store.get_run("old-run")
        schema_version = store._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()["value"]

    assert schema_version == "8"
    assert old_run["task_id"] == "old-task"
    assert old_run["attempt_budget"] is None
    assert old_run["attempt_budget_source"] is None
    assert old_run["attempt_budget_reason"] is None
    assert old_run["judge_model"] is None


def test_training_query_requires_positive_limit(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(ExperimentStoreError, match="limit must be at least 1"):
            store.training_observations(limit=0)


def test_records_the_judge_model_separately_from_the_generator(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        run_id = store.start_run(
            problem_statement="Problem",
            prompt_version="P0",
            model_name="qwen2.5-coder:7b",
            test_source="trusted",
            judge_enabled=True,
            judge_model="llama3.1:8b",
        )
        run = store.get_run(run_id)

    assert run["model_name"] == "qwen2.5-coder:7b"
    assert run["judge_model"] == "llama3.1:8b"
    assert run["judge_enabled"] == 1


def test_leaves_the_judge_model_empty_when_the_judge_is_off(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        run_id = store.start_run(
            problem_statement="Problem",
            prompt_version="P0",
            model_name="qwen2.5-coder:7b",
            test_source="trusted",
            judge_enabled=False,
        )
        run = store.get_run(run_id)

    assert run["judge_enabled"] == 0
    assert run["judge_model"] is None


def test_migrates_v2_history_schema_without_removing_existing_runs(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO metadata(key, value) VALUES('schema_version', '2');
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                problem_statement TEXT NOT NULL,
                benchmark_split TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                test_source TEXT NOT NULL,
                judge_enabled INTEGER NOT NULL,
                attempt_budget INTEGER,
                attempt_budget_source TEXT,
                attempt_budget_reason TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                final_status TEXT,
                stop_reason TEXT,
                best_attempt_number INTEGER
            );
            INSERT INTO runs(
                run_id, task_id, problem_statement, benchmark_split,
                prompt_version, model_name, test_source, judge_enabled,
                attempt_budget, attempt_budget_source, attempt_budget_reason,
                started_at
            ) VALUES (
                'old-run', 'old-task', 'Old problem', 'adhoc',
                'P0', 'old-model', 'trusted', 1,
                3, 'cli_override', 'Fixed for the ablation.',
                '2026-08-01T00:00:00Z'
            );
        """)
    connection.close()

    with ExperimentStore(path) as store:
        old_run = store.get_run("old-run")
        new_run_id = store.start_run(
            problem_statement="Problem",
            prompt_version="P0",
            model_name="qwen2.5-coder:7b",
            test_source="trusted",
            judge_enabled=True,
            judge_model="llama3.1:8b",
        )
        new_run = store.get_run(new_run_id)
        schema_version = store._connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()["value"]

    assert schema_version == "8"
    assert old_run["task_id"] == "old-task"
    assert old_run["attempt_budget"] == 3
    assert new_run["judge_model"] == "llama3.1:8b"
    for column in (
        "judge_model",
        "config_name",
        "seed",
        "language",
        "model_calls",
        "wall_clock_seconds",
    ):
        assert old_run[column] is None, column


def test_adds_columns_appended_after_the_database_reached_this_version(
    tmp_path, monkeypatch
):
    """A later task appending to RUNS_ADDED_COLUMNS must reach old databases."""
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path):
        pass

    monkeypatch.setattr(
        experiment_store_module,
        "RUNS_ADDED_COLUMNS",
        experiment_store_module.RUNS_ADDED_COLUMNS + (("seed", "INTEGER"),),
    )
    with ExperimentStore(path) as store:
        columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(runs)")
        }

    assert "seed" in columns


class SeparateJudgeClient(InvalidJudgeClient):
    """A judge from a different family than HistoryRepairClient."""

    model = "history-judge-model:1b"


def test_run_records_the_judge_model_the_loop_actually_used(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_candidate_refinement_loop(
            client=HistoryRepairClient(),
            problem_statement="Read one integer and print twice its value.",
            trusted_test_suite=trusted_suite(),
            history_store=store,
            task_id="double-judge-separate",
            benchmark_split="train",
            judge_client=SeparateJudgeClient(),
        )
        run = store.get_run(result.history_run_id)

    assert run["model_name"] == "history-test-model:1b"
    assert run["judge_model"] == "history-judge-model:1b"
    assert run["model_name"] != run["judge_model"]


def test_run_without_a_judge_records_no_judge_model(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    with ExperimentStore(path) as store:
        result = run_stored_task(store, "train", "double-no-judge")
        run = store.get_run(result.history_run_id)

    assert run["judge_enabled"] == 0
    assert run["judge_model"] is None


class PassingJudgeClient:
    """A judge that returns one valid, clean grade per candidate."""

    model = "history-judge-model:1b"

    def generate(self, prompt, system=None, temperature=0.0):
        return json.dumps({
            "problem_understanding": 0.9,
            "algorithm_suitability": 0.9,
            "edge_case_handling": 0.9,
            "requirement_adherence": 0.9,
            "code_quality": 0.9,
            "overall_score": 0.9,
            "critical_issues": [],
            "feedback": "No semantic issue remains.",
            "likely_failure_category": "NONE",
        })


def test_run_records_what_it_cost(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_stored_task(store, "train", "double-cost")
        run = store.get_run(result.history_run_id)

    # Task analysis, the first generation, and one repair. The trusted suite
    # means no test-generation call.
    assert result.model_calls == 3
    assert run["model_calls"] == 3
    assert result.wall_clock_seconds > 0
    assert run["wall_clock_seconds"] > 0


def test_judge_calls_count_towards_what_the_run_cost(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        without_judge = run_stored_task(store, "train", "double-cost-nojudge")
        with_judge = run_candidate_refinement_loop(
            client=HistoryRepairClient(),
            problem_statement="Read one integer and print twice its value.",
            trusted_test_suite=trusted_suite(),
            history_store=store,
            task_id="double-cost-judge",
            benchmark_split="train",
            judge_client=PassingJudgeClient(),
        )
        judged_run = store.get_run(with_judge.history_run_id)

    # The same three generator calls, plus one judge call per evaluated
    # candidate. A judge is not free and the cost column has to say so.
    assert without_judge.model_calls == 3
    assert with_judge.model_calls == 5
    assert judged_run["model_calls"] == 5


def test_records_the_experiment_arm_labels(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_candidate_refinement_loop(
            client=HistoryRepairClient(),
            problem_statement="Read one integer and print twice its value.",
            trusted_test_suite=trusted_suite(),
            history_store=store,
            task_id="double-arm",
            benchmark_split="train",
            config_name="full",
            language="python",
        )
        run = store.get_run(result.history_run_id)

    assert run["config_name"] == "full"
    assert run["language"] == "python"
    # No seed was requested for this run.
    assert run["seed"] is None


def test_migrates_v3_preserving_model_costs_and_adds_resume_tables(tmp_path):
    path = tmp_path / "v3.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES('schema_version', '3');
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                problem_statement TEXT NOT NULL, benchmark_split TEXT NOT NULL,
                prompt_version TEXT NOT NULL, model_name TEXT NOT NULL,
                test_source TEXT NOT NULL, judge_enabled INTEGER NOT NULL,
                judge_model TEXT, config_name TEXT, seed INTEGER, language TEXT,
                model_calls INTEGER, wall_clock_seconds REAL,
                attempt_budget INTEGER, attempt_budget_source TEXT, attempt_budget_reason TEXT,
                started_at TEXT NOT NULL, completed_at TEXT, final_status TEXT,
                stop_reason TEXT, best_attempt_number INTEGER
            );
            INSERT INTO runs(run_id, task_id, problem_statement, benchmark_split,
                prompt_version, model_name, test_source, judge_enabled,
                config_name, seed, language, model_calls, wall_clock_seconds, started_at)
            VALUES('legacy', 'task', 'Problem', 'train', 'P0', 'model', 'trusted',
                0, 'full', 42, 'python', 5, 12.5, '2026-09-01');
        """)
    with ExperimentStore(path) as store:
        row = store.get_run("legacy")
        assert row["model_calls"] == 5
        assert row["wall_clock_seconds"] == 12.5
        assert row["seed"] == 42
        assert row["benchmark_run_id"] is None
        assert row["analysis_model_calls"] is None
        assert row["task_spec_cache_hit"] is None
        assert store._connection.execute("SELECT value FROM metadata").fetchone()[0] == "8"
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0] == 0
        assert store._connection.execute("SELECT COUNT(*) FROM benchmark_tasks").fetchone()[0] == 0


def test_migrates_v4_without_changing_existing_runs_or_benchmark_checkpoints(tmp_path):
    path = tmp_path / "v4.db"
    with ExperimentStore(path) as store:
        run_id = store.start_run("Problem", "P0", "model", "trusted", False)
        original_run = store.get_run(run_id)
        with store._connection:
            store._connection.executescript("""
                DROP TABLE judge_sanity_results;
                DROP TABLE judge_sanity_runs;
                UPDATE metadata SET value = '4' WHERE key = 'schema_version';
                INSERT INTO benchmark_runs(run_id, manifest_json, started_at)
                VALUES ('sweep', '{"seed":42}', '2026-09-01');
                INSERT INTO benchmark_tasks(benchmark_run_id, task_id, checkpoint_json)
                VALUES ('sweep', 'task', '{"attempt":2}');
            """)
    with ExperimentStore(path) as store:
        assert store.get_run(run_id) == original_run
        assert store._connection.execute("SELECT value FROM metadata").fetchone()[0] == "8"
        assert store._connection.execute("SELECT checkpoint_json FROM benchmark_tasks").fetchone()[0] == '{"attempt":2}'
        assert store._connection.execute("SELECT COUNT(*) FROM judge_sanity_runs").fetchone()[0] == 0
        assert store._connection.execute("SELECT COUNT(*) FROM judge_sanity_results").fetchone()[0] == 0


def _experiment_arguments(**overrides):
    arguments = {
        "baseline_prompt_id": "P0",
        "candidate_prompt_id": "P1",
        "benchmark_split": "validation",
        "task_count": 40,
        "baseline_pass_rate": 0.55,
        "candidate_pass_rate": 0.70,
        "candidate_only_passes": 9,
        "baseline_only_passes": 3,
        "p_value": 0.073,
        "decision": "promoted",
        "reason": "Candidate beat the baseline on discordant pairs.",
    }
    arguments.update(overrides)
    return arguments


def test_prompt_experiment_round_trips(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        experiment_id = store.record_prompt_experiment(**_experiment_arguments())
        records = store.prompt_experiments()

    assert len(records) == 1
    record = records[0]
    assert record["experiment_id"] == experiment_id
    assert record["baseline_prompt_id"] == "P0"
    assert record["candidate_prompt_id"] == "P1"
    assert record["candidate_only_passes"] == 9
    assert record["baseline_only_passes"] == 3
    assert record["p_value"] == pytest.approx(0.073)
    assert record["decision"] == "promoted"
    assert record["created_at"] is not None


def test_prompt_experiments_return_newest_first_and_filter_by_candidate(tmp_path):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.record_prompt_experiment(
            **_experiment_arguments(
                candidate_prompt_id="P1",
                decision="rejected",
                created_at="2026-09-01T00:00:00Z",
            )
        )
        store.record_prompt_experiment(
            **_experiment_arguments(
                candidate_prompt_id="P2",
                created_at="2026-09-20T00:00:00Z",
            )
        )
        newest_first = store.prompt_experiments()
        only_p1 = store.prompt_experiments(candidate_prompt_id="P1")

    assert [record["candidate_prompt_id"] for record in newest_first] == ["P2", "P1"]
    assert [record["decision"] for record in only_p1] == ["rejected"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"decision": "maybe"}, "decision must be one of"),
        ({"benchmark_split": "test"}, "benchmark_split must be one of"),
        ({"candidate_prompt_id": "P0"}, "candidate_prompt_id must differ"),
        ({"task_count": 0}, "task_count must be a positive integer"),
        ({"candidate_only_passes": -1}, "candidate_only_passes must be a non-negative"),
        ({"p_value": 1.4}, "p_value must be between 0 and 1"),
        ({"reason": "  "}, "reason must be a non-empty string"),
        (
            {"task_count": 5, "candidate_only_passes": 4, "baseline_only_passes": 3},
            "discordant pairs cannot exceed task_count",
        ),
    ],
)
def test_rejects_invalid_prompt_experiment(tmp_path, overrides, message):
    with ExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(ExperimentStoreError, match=message):
            store.record_prompt_experiment(**_experiment_arguments(**overrides))
