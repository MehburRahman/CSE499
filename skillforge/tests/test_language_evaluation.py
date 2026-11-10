"""Language-aware evaluation through real Node.js runs and stubbed model calls."""

import json
import shutil
import sys
from unittest.mock import patch

import pytest

import run_benchmark as benchmark_cli
import run_reinforcement
import src.evaluation.pipeline as pipeline
from src.agents.task_spec import TaskSpec
from src.agents.test_generator import FunctionTestCase, TestCase as Case, TestSuite as Suite
from src.benchmark.loader import load_benchmark
from src.benchmark.runner import run_benchmark
from src.evaluation.constraint_grader import grade_constraint_adherence
from src.evaluation.models import EvaluationStatus
from src.history import ExperimentStore, ExperimentStoreError
from src.reinforcement_loop import run_candidate_refinement_loop
from src.verifier.outcome_verifier import verify
from tests.test_benchmark_loader import write_benchmark
from tests.test_evaluation_pipeline import judge_response
from tests.test_execution_contracts import function_task
from tests.test_experiment_configs import RecordingClient


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
JS_SUM = r"const [a,b] = require('node:fs').readFileSync(0,'utf8').trim().split(/\s+/).map(Number); console.log(a+b);"
JS_FUNCTION = "function solve(a, b) { return a + b; }"


def spec(constrained=False):
    return TaskSpec(
        "Return the sum from solve(a, b). Use recursion." if constrained else "Add two integers.",
        "math", "integers", required_operations=["recursion"] if constrained else [],
    )


class RecordingJudge:
    model = "independent-judge"

    def __init__(self):
        self.payloads = []

    def generate(self, prompt, system=None, temperature=0, *, seed=None):
        data = prompt.split("<BEGIN_UNTRUSTED_EVALUATION_DATA>\n", 1)[1].split("\n<END_UNTRUSTED_EVALUATION_DATA>", 1)[0]
        self.payloads.append(json.loads(data))
        return judge_response(score=1.0)


@pytest.mark.parametrize("language", ["javascript", "cpp", "java"])
@pytest.mark.parametrize("constrained", [False, True])
def test_non_python_constraints_never_use_python_parser(language, constrained):
    task = spec(constrained)
    with patch("src.evaluation.constraint_grader.ast.parse", side_effect=AssertionError("Python parser used")):
        grade = grade_constraint_adherence("function solve(a,b) { return a+b; }", task, language=language)
    assert grade.score is None
    assert grade.coverage == (0.0 if constrained else 1.0)
    assert grade.unverified_constraints == task.required_operations
    assert grade.checked_constraints == grade.passed_constraints == grade.violations == []
    assert grade.analysis_error is None
    assert not grade.critical_violation


def test_javascript_syntax_that_is_also_python_is_not_checked_as_python():
    task = TaskSpec("Do not use sort(). Use recursion.", "sorting", "",
                    prohibited_operations=["sort()"], required_operations=["recursion"])
    grade = grade_constraint_adherence("values.sort()", task, language="javascript")
    assert grade.unverified_constraints == ["sort()", "recursion"]
    assert grade.violations == []
    assert grade.coverage == 0


@pytest.mark.parametrize("language", ["rust", None, "JavaScript"])
def test_unknown_grading_language_is_rejected(language):
    with pytest.raises(ValueError):
        grade_constraint_adherence("print(1)", spec(), language=language)


def test_python_default_and_explicit_language_produce_identical_grades():
    for code in ("def solve(a,b): return a+b", "def broken(:", "def solve(a,b): return solve(a,b)"):
        assert grade_constraint_adherence(code, spec(True)) == grade_constraint_adherence(code, spec(True), language="python")


@needs_node
@pytest.mark.parametrize("code,reason", [
    ("console.log(5);", None),
    ("console.log(0);", "FUNCTIONAL_TEST_FAILURE"),
    ("throw new Error('crash');", "RUNTIME_EXCEPTION"),
    ("const broken = ;", "RUNTIME_EXCEPTION"),
    ("while (true) {}", "TIMEOUT"),
])
@pytest.mark.parametrize("constrained", [False, True])
def test_javascript_gates_preserve_failures_even_with_perfect_judge(code, reason, constrained):
    suite = Suite([Case("", "5", "normal")])
    verification = verify(code, suite, language="javascript", timeout=0.5)
    judge = RecordingJudge()
    report, _ = pipeline.evaluate_candidate(spec(constrained), code, verification, "trusted",
                                           language="javascript", judge_client=judge)
    expected = EvaluationStatus.FAIL if reason else EvaluationStatus.REVIEW if constrained else EvaluationStatus.PASS
    assert report.status == expected
    assert report.hard_gate_passed is (reason is None)
    assert report.constraint_grade.analysis_error is None
    assert report.judge_grade.overall_score == 1
    assert [r.code for r in report.review_reasons] == (["UNVERIFIED_CONSTRAINTS"] if constrained else [])
    if reason:
        assert reason in [r.code for r in report.failure_reasons]
    assert judge.payloads[0]["execution_contract"] == {
        "language": "javascript", "execution_mode": "stdio", "entry_point": None,
    }


@needs_node
@pytest.mark.parametrize("mode", ["stdio", "function"])
@pytest.mark.parametrize("config", ["single_shot", "best_of_n", "scalar_repair", "full"])
def test_javascript_candidate_loop_all_arms(mode, config):
    function_mode = mode == "function"
    bad = "function solve(a,b) { return 0; }" if function_mode else "console.log(0);"
    good = JS_FUNCTION if function_mode else JS_SUM
    client = RecordingClient((bad, good, bad))
    suite = Suite([FunctionTestCase("assert.strictEqual(solve(2,3),5);")] if function_mode else [Case("2 3", "5", "normal")])
    result = run_candidate_refinement_loop(
        client, "Return the sum from solve(a, b)." if function_mode else "Read two integers and print their sum.",
        trusted_test_suite=suite, language="javascript", execution_mode=mode,
        entry_point="solve" if function_mode else None, config_name=config, seed=42,
    )
    assert result.success is (config != "single_shot")
    assert len(result.attempts) == (1 if config == "single_shot" else 3 if config == "best_of_n" else 2)
    assert all(a.evaluation_report.constraint_grade.analysis_error is None for a in result.attempts)
    assert all("JavaScript running in Node.js" in r["system"] for r in client.code_requests)
    if config == "scalar_repair":
        assert "Passed: 0/1" in client.code_requests[1]["prompt"]
        assert "Failures:" not in client.code_requests[1]["prompt"]


@needs_node
@pytest.mark.parametrize("constrained", [False, True])
def test_mixed_language_benchmark_history_hidden_evaluation_and_resume(tmp_path, constrained):
    python_task, js_task = function_task(), function_task("javascript")
    python_task["task_id"] = "python-sum"
    js_task["task_id"] = "js-sum"
    js_task["hidden_tests"] = [{"assertion": "assert.strictEqual(solve(913,427),1340);", "category": "edge"}]
    if constrained:
        js_task["problem_statement"] += " Use recursion."
    path = tmp_path / "catalog.json"
    write_benchmark(path, [python_task, js_task])

    class Analyzer(RecordingClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
            response = super().generate(prompt, system, temperature, seed=seed)
            if "problem analyzer" in system and "Use recursion." in prompt:
                response = json.loads(response)
                response["required_operations"] = ["recursion"]
                return json.dumps(response)
            return response

    client = Analyzer(("def solve(a,b): return a+b", JS_FUNCTION))
    judge = RecordingJudge()
    options = dict(dataset=load_benchmark(path), run_id="mixed", config_name="full", seed=42, judge_client=judge)
    with ExperimentStore(tmp_path / "history.db") as store:
        original = run_benchmark(client, history_store=store, **options)
        py_result, js_result = original.task_results
        assert py_result.hidden_status == "pass"
        assert js_result.final_visible_status == js_result.hidden_status == ("review" if constrained else "pass")
        assert js_result.stop_reason == ("review_required" if constrained else "passed")
        assert js_result.hidden_score == 1
        assert js_result.language == "javascript"
        assert store.get_run(js_result.history_run_id)["language"] == "javascript"
        hidden_evidence = judge.payloads[-1]
        assert hidden_evidence["execution_contract"] == {
            "language": "javascript", "execution_mode": "function", "entry_point": "solve",
        }
        def payload_leaves(value):
            if isinstance(value, dict):
                for item in (*value.keys(), *value.values()):
                    yield from payload_leaves(item)
            elif isinstance(value, list):
                for item in value:
                    yield from payload_leaves(item)
            else:
                yield value

        # Hidden literals must not reach the judge. Searching serialized JSON
        # also matched incidental digits inside measured runtime floats.
        for value in payload_leaves(judge.payloads):
            if isinstance(value, str):
                assert "913" not in value and "1340" not in value
            else:
                assert value not in (913, 1340)
        assert all("913" not in r["prompt"] for r in client.requests)
        resumed_client = Analyzer(())
        resumed = run_benchmark(resumed_client, history_store=store, resume=True, **options)
        assert resumed.task_results == original.task_results
        assert resumed_client.requests == [] and len(judge.payloads) == 4


@needs_node
@pytest.mark.parametrize("mode", ["stdio", "function"])
def test_javascript_reinforcement_cli(mode, tmp_path, monkeypatch, capsys):
    client = RecordingClient((JS_FUNCTION if mode == "function" else JS_SUM,))
    monkeypatch.setattr(run_reinforcement, "build_clients", lambda **kwargs: (client, None))
    tests = tmp_path / "tests.json"
    records = [{"assertion": "assert.strictEqual(solve(2,3),5);", "category": "normal"}] if mode == "function" else [
        {"input": "2 3", "expected_output": "5", "category": "normal"},
    ]
    tests.write_text(json.dumps(records))
    args = ["run_reinforcement.py", "Add two integers; the function name is solve(a, b)." if mode == "function" else
            "Read two integers and print their sum.", "--language", "javascript", "--execution-mode", mode,
            "--tests", str(tests), "--no-history"]
    if mode == "function":
        args += ["--entry-point", "solve"]
    monkeypatch.setattr(sys, "argv", args)
    run_reinforcement.main()
    assert "SUCCESS" in capsys.readouterr().out


@needs_node
def test_javascript_benchmark_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "catalog.json"
    write_benchmark(path, [function_task("javascript")])
    client = RecordingClient((JS_FUNCTION,))
    monkeypatch.setattr(benchmark_cli, "build_clients", lambda **kwargs: (client, None))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", str(path), "--language", "javascript",
                                      "--no-history", "--no-task-spec-cache"])
    benchmark_cli.main()
    output = capsys.readouterr().out
    report = json.loads(output.split("BENCHMARK SUMMARY\n", 1)[1])
    assert report["task_results"][0]["hidden_status"] == "pass"


@needs_node
@pytest.mark.parametrize("change", ["version", "missing"])
def test_resume_rejects_changed_evaluation_policy_before_model_calls(tmp_path, monkeypatch, change):
    path = tmp_path / "catalog.json"
    write_benchmark(path, [function_task("javascript")])
    options = dict(dataset=load_benchmark(path), run_id="policy", config_name="single_shot")
    with ExperimentStore(tmp_path / "history.db") as store:
        run_benchmark(RecordingClient((JS_FUNCTION,)), history_store=store, **options)
        manifest = json.loads(store._connection.execute("SELECT manifest_json FROM benchmark_runs").fetchone()[0])
        assert manifest["evaluation_policy"] == pipeline.evaluation_policy_identity()
        if change == "version":
            monkeypatch.setattr(pipeline, "EVALUATION_POLICY_VERSION", "changed")
        else:
            manifest.pop("evaluation_policy")
            with store._connection:
                store._connection.execute("UPDATE benchmark_runs SET manifest_json=?", (json.dumps(manifest),))
        client = RecordingClient(())
        with pytest.raises(ExperimentStoreError, match="settings or contents have changed"):
            run_benchmark(client, history_store=store, resume=True, **options)
        assert client.requests == []


@needs_node
def test_javascript_stdio_supports_generated_tests():
    class GeneratedTestsClient(RecordingClient):
        def generate(self, prompt, system=None, temperature=0.2, *, seed=None):
            response = super().generate(prompt, system, temperature, seed=seed)
            if "test cases" in system.lower():
                return json.dumps([{"input": "2 3", "expected_output": "5", "category": "normal"}])
            return response

    client = GeneratedTestsClient((JS_SUM,))
    result = run_candidate_refinement_loop(client, "Read two integers and print their sum.", language="javascript")
    assert result.success
    assert result.test_source == "generated"
    assert result.model_calls == 3  # Analysis, test generation, and the candidate.
