"""Paired validation and a persisted recommendation; registry wiring is step 2.3."""

from copy import deepcopy
import json

from src.agents.llm_client import validate_judge_pair
from src.agents.task_spec_cache import TaskSpecCache, task_spec_from_dict
from src.benchmark.models import BenchmarkDataset
from src.benchmark.runner import benchmark_manifest, run_benchmark, select_benchmark_tasks
from src.evaluation.statistics import compare_paired_outcomes, exact_mcnemar_p_value
from src.evolution.models import PromptValidationResult, ValidationTaskOutcome
from src.experiments import DEFAULT_EXPERIMENT_ATTEMPTS, validate_seed


class _FrozenPrompts:
    """Both arms use the prompt texts captured at the start of validation."""

    def __init__(self, baseline, candidate):
        self.active_prompt = baseline
        self.versions = {p.prompt_id: p for p in (baseline, candidate)}

    def get(self, prompt_id):
        return self.versions[prompt_id]


class _BaselineAnalysis:
    """Reuse the exact baseline analysis, even when the disk cache was cleared."""

    def __init__(self, history_store, run_id, tasks):
        self.store, self.run_id = history_store, run_id
        self.task_ids = {}
        for task in tasks:
            key = (task.problem_statement, task.language, task.execution_mode, task.entry_point)
            self.task_ids.setdefault(key, []).append(task.task_id)

    def get_or_analyze(self, client, problem_statement, language="python", *,
                       execution_mode="stdio", entry_point=None):
        key = (problem_statement, language, execution_mode, entry_point)
        specs = []
        for task_id in self.task_ids[key]:
            saved = self.store.benchmark_task(self.run_id, task_id)
            if saved["checkpoint_json"] is None or saved["result_json"] is None:
                raise ValueError("baseline task must be complete before candidate analysis")
            specs.append(json.loads(saved["checkpoint_json"])["task_spec"])
        # Identical execution contracts share a cache key. Refuse ambiguous
        # baseline analyses instead of silently pairing one task with another.
        if any(spec != specs[0] for spec in specs[1:]):
            raise ValueError("identical validation task contracts have different baseline analyses")
        return task_spec_from_dict(specs[0], problem_statement), True, 0


def _binary_outcomes(summary, tasks, prompt_id, seed, max_attempts):
    expected = {task.task_id for task in tasks}
    actual = [row.task_id for row in summary.task_results]
    if summary.prompt_version != prompt_id or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("validation results must contain each selected task exactly once")
    outcomes = {}
    for row in summary.task_results:
        if (row.split != "validation" or row.prompt_version != prompt_id
                or row.seed != seed or row.attempt_budget != max_attempts or row.config_name != "full"):
            raise ValueError("validation result does not match the paired experiment settings")
        if row.final_visible_status not in {"pass", "fail", "review"} or row.hidden_status not in {"pass", "fail", "review"}:
            raise ValueError("validation requires completed pass/fail/review reports")
        for score in (row.final_visible_score, row.hidden_score):
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError("validation requires valid functional scores")
        # A perfect functional score alone can hide a runtime/constraint failure.
        # 'review' has passed hard gates; advisory judge concerns are not failures.
        outcomes[row.task_id] = (
            row.final_visible_score == 1.0 and row.hidden_score == 1.0
            and row.final_visible_status != "fail" and row.hidden_status != "fail"
        )
    return outcomes


def _check_task_history(result, task, history_store, run_id, manifest):
    """Audit saved provenance too: restoring settings cannot repair mixed runs."""
    run = history_store.get_run(result.history_run_id)
    expected = {
        "task_id": task.task_id, "problem_statement": task.problem_statement,
        "benchmark_split": "validation", "benchmark_run_id": run_id,
        "prompt_version": manifest["prompt_id"], "model_name": manifest["generator_model"],
        "config_name": manifest["config_name"], "seed": manifest["seed"],
        "attempt_budget": manifest["max_attempts"], "test_source": "trusted",
        "judge_enabled": int(manifest["judge_model"] is not None),
        "language": task.language, "execution_mode": task.execution_mode, "entry_point": task.entry_point,
    }
    if run is None or run["completed_at"] is None or any(run[key] != value for key, value in expected.items()):
        raise ValueError("stored validation task history does not match the paired settings")
    judge_model = (run["judge_model"] or "unknown") if run["judge_enabled"] else None
    if judge_model != manifest["judge_model"]:
        raise ValueError("stored validation judge does not match the paired settings")
    attempts = history_store.get_attempts(result.history_run_id)
    if (not 1 <= len(attempts) == result.attempt_count <= manifest["max_attempts"]
            or any(a["model_name"] != manifest["generator_model"]
                   or a["prompt_version"] != manifest["prompt_id"] for a in attempts)):
        raise ValueError("stored validation attempts do not match the paired settings")


def validate_candidate_prompt(
    client, dataset, prompt_registry, history_store, candidate_prompt_id, *,
    baseline_prompt_id=None, seed=42, max_attempts=DEFAULT_EXPERIMENT_ATTEMPTS,
    judge_client=None, task_spec_cache=None, validation_id=None, resume=False,
    on_task_complete=None, on_validation_start=None,
) -> PromptValidationResult:
    """Run the same validation tasks and budget under baseline and candidate.

    ``on_validation_start(id)`` exposes the durable ID before any model calls.
    Resume with that ID and identical settings to reuse completed tasks and
    frozen code. Saved task provenance is checked even for completed comparisons.
    Incomplete/errored pairs never produce a statistical decision.
    The optional task callback receives each TaskBenchmarkResult (including
    its prompt_version). This function never activates or rejects a prompt.
    """
    validate_seed(seed)
    if seed is None:
        raise ValueError("paired validation requires an explicit seed")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    validate_judge_pair(client, judge_client)
    if history_store is None:
        raise ValueError("paired validation requires history storage")
    prompt_registry.refresh()
    baseline = prompt_registry.get(baseline_prompt_id) if baseline_prompt_id else prompt_registry.active_prompt
    candidate = prompt_registry.get(candidate_prompt_id)
    if baseline.status != "active" or candidate.status != "candidate" or candidate.parent_prompt_id != baseline.prompt_id:
        raise ValueError("validation requires a candidate of the current active baseline")
    prompts = _FrozenPrompts(baseline, candidate)
    # Copy only validation data, so callbacks/caller mutation cannot change an
    # arm midway through the pair. Training and final hidden splits stay unused.
    tasks = tuple(deepcopy(select_benchmark_tasks(dataset, splits=["validation"])))
    paired_dataset = BenchmarkDataset(tasks=tasks)
    cache = task_spec_cache if task_spec_cache is not None else TaskSpecCache(
        history_store.path.parent / "task_spec_cache"
    )
    settings = dict(config_name="full", seed=seed, max_attempts=max_attempts,
                    judge_client=judge_client, task_spec_cache=cache)
    manifest = {
        "protocol": 1,
        "test": "exact-one-sided-mcnemar", "alpha": 0.10,
        "outcome_policy": "visible-and-hidden-functional-and-hard-gates-v1",
        "analysis_policy": "reuse-baseline-checkpoint-v1",
        "baseline": benchmark_manifest(client, paired_dataset, tasks, baseline, **settings),
        "candidate": benchmark_manifest(client, paired_dataset, tasks, candidate, **settings),
    }
    saved = history_store.begin_prompt_validation(manifest, validation_id, resume)
    validation_id = saved["validation_id"]
    if on_validation_start is not None:
        on_validation_start(validation_id)

    def check_unchanged(*, full_manifest=True):
        prompt_registry.refresh()
        if (prompt_registry.active_prompt != baseline
                or prompt_registry.get(candidate_prompt_id) != candidate):
            raise ValueError("prompt registry changed during validation; no decision recorded")
        models = {
            "generator_model": getattr(client, "model", "unknown"),
            "judge_model": getattr(judge_client, "model", "unknown") if judge_client is not None else None,
        }
        if any(manifest["baseline"][key] != value for key, value in models.items()):
            raise ValueError("validation settings changed during execution; no decision recorded")
        # Tasks and options are private snapshots. Rehash the full dataset and
        # implementation at arm boundaries, rather than once per progress hook.
        for name, prompt in (("baseline", baseline), ("candidate", candidate)):
            if full_manifest and benchmark_manifest(client, paired_dataset, tasks, prompt, **settings) != manifest[name]:
                raise ValueError("validation settings changed during execution; no decision recorded")

    tasks_by_id = {task.task_id: task for task in tasks}

    def task_callback(arm):
        def completed(row):
            check_unchanged(full_manifest=False)
            _check_task_history(row, tasks_by_id[row.task_id], history_store,
                                saved[f"{arm}_run_id"], manifest[arm])
            if on_task_complete is not None:
                on_task_complete(row)
                # Stop before the next task can use settings changed by a hook.
                check_unchanged(full_manifest=False)
        return completed

    check_unchanged()
    baseline_summary = run_benchmark(
        client, paired_dataset, prompt_registry=prompts, prompt_id=baseline.prompt_id,
        history_store=history_store, run_id=saved["baseline_run_id"], resume=True,
        on_task_complete=task_callback("baseline"), **settings,
    )
    check_unchanged()
    candidate_settings = dict(settings, task_spec_cache=_BaselineAnalysis(
        history_store, saved["baseline_run_id"], tasks,
    ))
    candidate_summary = run_benchmark(
        client, paired_dataset, prompt_registry=prompts, prompt_id=candidate.prompt_id,
        history_store=history_store, run_id=saved["candidate_run_id"], resume=True,
        on_task_complete=task_callback("candidate"), **candidate_settings,
    )
    baseline_outcomes = _binary_outcomes(baseline_summary, tasks, baseline.prompt_id, seed, max_attempts)
    candidate_outcomes = _binary_outcomes(candidate_summary, tasks, candidate.prompt_id, seed, max_attempts)
    result = PromptValidationResult(
        validation_id=validation_id, baseline_prompt_id=baseline.prompt_id,
        candidate_prompt_id=candidate.prompt_id, baseline_run_id=saved["baseline_run_id"],
        candidate_run_id=saved["candidate_run_id"], seed=seed, max_attempts=max_attempts,
        comparison=compare_paired_outcomes(baseline_outcomes, candidate_outcomes),
        task_outcomes=tuple(ValidationTaskOutcome(key, baseline_outcomes[key], candidate_outcomes[key])
                            for key in sorted(baseline_outcomes)),
    )
    check_unchanged()
    history_store.finish_prompt_validation(result)
    return result
