from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from src.evolution import check_evolution_trigger, recover_evolution_attempt
from src.evolution.prompt_registry import PromptRegistry
from src.history import ExperimentStore, ExperimentStoreError
from tests.test_prompt_evolver import add_runs, registry


@pytest.mark.parametrize("count,expected", [(0, False), (19, False), (20, True), (21, True)])
def test_completed_training_threshold(tmp_path, count, expected):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store, count)
        decision = check_evolution_trigger(store, registry(tmp_path))
        assert decision.eligible is expected
        assert decision.reserved is False
        assert decision.evidence["training_runs"] == count
        assert store.evolution_checks()[0]["status"] == ("ready" if expected else "blocked")
        assert store.evolution_evidence("P0")["training_runs"] == count


@pytest.mark.parametrize("categories,signaled_runs,expected", [
    (("EDGE_CASE", "LOGIC_ERROR", "INPUT_PARSING"), 2, False),
    (("EDGE_CASE", "LOGIC_ERROR"), 10, False),
    (("EDGE_CASE", "LOGIC_ERROR", "INPUT_PARSING"), 3, True),
    (("EDGE_CASE", "LOGIC_ERROR", "NONE", "INVALID"), 3, False),
])
def test_categories_count_distinct_runs_not_attempts(tmp_path, categories, signaled_runs, expected):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store, categories=categories, signaled_runs=signaled_runs, repeats=4)
        decision = check_evolution_trigger(store, registry(tmp_path))
        assert decision.eligible is expected
        assert decision.evidence["failure_counts"]["EDGE_CASE"] == signaled_runs


def test_excludes_other_splits_prompts_and_unfinished_runs(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as store:
        for split in ("validation", "hidden", "adhoc"):
            add_runs(store, split=split)
        add_runs(store, prompt="P9")
        add_runs(store, completed=False)
        decision = check_evolution_trigger(store, registry(tmp_path))
        assert not decision.eligible
        assert decision.evidence["training_runs"] == 0
        assert decision.evidence["failure_counts"] == {}
        assert decision.evidence["observations"] == []


def test_attempt_prompt_must_match_active_run_prompt(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        with store._connection:
            store._connection.execute("UPDATE attempts SET prompt_version = 'P9'")
        decision = check_evolution_trigger(store, registry(tmp_path))
        assert not decision.eligible
        assert decision.evidence["failure_counts"] == {}


def test_reservation_restart_cooldown_and_new_window(tmp_path):
    path = tmp_path / "history.db"
    prompts = registry(tmp_path)
    with ExperimentStore(path) as store:
        old_ids = add_runs(store)
        first = check_evolution_trigger(store, prompts, reserve=True)
        assert first.reserved
    with ExperimentStore(path) as store:
        blocked = check_evolution_trigger(store, prompts, reserve=True)
        assert not blocked.reserved and "in progress" in blocked.reason
        # Explicitly recover a crashed proposer; inspection never clears it automatically.
        store.finish_evolution_attempt(first.decision_id, "cancelled", "Previous process stopped.")
        blocked = check_evolution_trigger(store, prompts, reserve=True)
        assert "cooldown" in blocked.reason
        add_runs(store, 4)
        assert "cooldown" in check_evolution_trigger(store, prompts).reason
        add_runs(store, 1)
        assert "5/20" in check_evolution_trigger(store, prompts).reason
        add_runs(store, 15)
        second = check_evolution_trigger(store, prompts, reserve=True)
        assert second.reserved and second.evidence["training_runs"] == 20
        assert not set(old_ids) & set(second.evidence["run_ids"])
        assert len(store.evolution_checks()) == 6
        with pytest.raises(ExperimentStoreError, match="already finished"):
            store.finish_evolution_attempt(first.decision_id, "failed", "Cannot finalize twice.")


def test_old_failures_do_not_qualify_a_new_window(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as store:
        prompts = registry(tmp_path)
        add_runs(store)
        first = check_evolution_trigger(store, prompts, reserve=True)
        store.finish_evolution_attempt(first.decision_id, "failed", "Model error")
        add_runs(store, signaled_runs=0)
        decision = check_evolution_trigger(store, prompts)
        assert decision.evidence["training_runs"] == 20
        assert not decision.eligible and decision.evidence["failure_counts"] == {}


def test_late_completion_is_new_evidence_even_with_old_start_time(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as store:
        prompts = registry(tmp_path)
        pending = add_runs(store, completed=False)
        add_runs(store)
        first = check_evolution_trigger(store, prompts, reserve=True)
        store.finish_evolution_attempt(first.decision_id, "failed", "Model error")
        with store._connection:
            store._connection.executemany(
                "UPDATE runs SET completed_at = '2026-01-01T00:00:00Z', final_status = 'fail' WHERE run_id = ?",
                [(run_id,) for run_id in pending],
            )
        second = check_evolution_trigger(store, prompts)
        assert second.eligible and set(second.evidence["run_ids"]) == set(pending)


def test_nonactive_prompt_is_blocked(tmp_path):
    prompts = registry(tmp_path)
    candidate = prompts.register_candidate("Candidate", "Reason")
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store, prompt=candidate.prompt_id)
        decision = check_evolution_trigger(store, prompts, prompt_id=candidate.prompt_id, reserve=True)
        assert not decision.reserved and "active prompt" in decision.reason
        assert store.evolution_checks()[-1]["status"] == "blocked"


def test_two_connections_cannot_reserve_the_same_evidence(tmp_path):
    path = tmp_path / "history.db"
    prompts = registry(tmp_path)
    with ExperimentStore(path) as store:
        add_runs(store)
    def reserve(_):
        with ExperimentStore(path) as store:
            return check_evolution_trigger(store, PromptRegistry(prompts.path), reserve=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert sum(r.reserved for r in results) == 1
    with ExperimentStore(path) as store:
        assert len(store.evolution_checks()) == 2


def test_v6_migration_preserves_history_and_adds_trigger_tables(tmp_path):
    path = tmp_path / "history.db"
    with ExperimentStore(path) as store:
        run_id = add_runs(store, 1)[0]
        original = store.get_run(run_id)
        signals = store.get_signals(run_id)
    with sqlite3.connect(path) as db:
        db.executescript("""
            DROP TABLE evolution_attempt_runs;
            DROP TABLE evolution_checks;
            UPDATE metadata SET value = '6' WHERE key = 'schema_version';
        """)
    with ExperimentStore(path) as store:
        assert store.get_run(run_id) == original
        assert store.get_signals(run_id) == signals
        assert store.evolution_checks() == []
        assert store._connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "8"
        assert not check_evolution_trigger(store, registry(tmp_path)).eligible


@pytest.mark.parametrize("candidate_written", [False, True])
def test_crash_recovery_after_reopen_is_idempotent(tmp_path, candidate_written):
    path = tmp_path / "history.db"
    prompts = registry(tmp_path)
    with ExperimentStore(path) as store:
        add_runs(store)
        decision = check_evolution_trigger(store, prompts, reserve=True)
        if candidate_written:
            prompts.register_candidate("Recovered prompt", "Proposal", parent_prompt_id="P0",
                                       trigger_decision_id=decision.decision_id)
        # Simulate death before the SQLite finalization; only durable state remains.
    with ExperimentStore(path) as store:
        prompts = PromptRegistry(prompts.path)
        result = recover_evolution_attempt(store, prompts, decision.decision_id)
        assert result["status"] == ("candidate" if candidate_written else "cancelled")
        assert result["candidate_prompt_id"] == ("P1" if candidate_written else None)
        assert prompts.active_prompt.prompt_id == "P0"
        assert store.evolution_evidence("P0")["training_runs"] == 0
        assert recover_evolution_attempt(store, prompts, decision.decision_id) == result


def test_finalization_does_not_commit_its_owners_transaction(tmp_path):
    with ExperimentStore(tmp_path / "history.db") as store:
        add_runs(store)
        decision = check_evolution_trigger(store, registry(tmp_path), reserve=True)
        with pytest.raises(RuntimeError, match="rollback"):
            with store.evolution_transaction():
                store.finish_evolution_attempt(decision.decision_id, "failed", "Test failure")
                raise RuntimeError("rollback")
        assert store.evolution_attempt_pending(decision.decision_id)
        assert store.get_evolution_check(decision.decision_id)["finished_at"] is None
