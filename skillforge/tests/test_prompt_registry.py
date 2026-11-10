import json

import pytest

from src.evolution.prompt_registry import (
    PromptRegistry,
    PromptRegistryError,
    load_default_registry,
)


P0 = {
    "prompt_id": "P0",
    "prompt_text": "Return only runnable Python code.",
    "parent_prompt_id": None,
    "created_at": "2026-08-14T00:00:00Z",
    "change_reason": "Initial prompt",
    "status": "active",
    "training_metrics": {},
    "validation_metrics": {},
}


def write_registry(path, prompts=None, schema_version=1):
    path.write_text(
        json.dumps({
            "schema_version": schema_version,
            "prompts": prompts if prompts is not None else [P0],
        }),
        encoding="utf-8",
    )


def test_default_registry_contains_active_p0():
    registry = load_default_registry()

    assert registry.active_prompt.prompt_id == "P0"
    assert "complete, runnable Python solution" in registry.active_prompt.prompt_text


def test_loads_and_gets_prompt(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)

    registry = PromptRegistry(path)

    assert registry.get("P0") == registry.active_prompt
    assert registry.active_prompt.training_metrics == {}


def test_registers_candidate_without_changing_active_prompt(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)

    candidate = registry.register_candidate(
        "Check edge cases before coding.",
        "Repeated boundary failures",
        training_metrics={"functional_accuracy": 0.75},
        created_at="2026-08-14T01:00:00Z",
    )

    assert candidate.prompt_id == "P1"
    assert candidate.parent_prompt_id == "P0"
    assert candidate.status == "candidate"
    assert candidate.training_metrics == {"functional_accuracy": 0.75}
    assert registry.active_prompt.prompt_id == "P0"
    assert PromptRegistry(path).get("P1") == candidate


def test_candidate_ids_continue_after_highest_existing_id(tmp_path):
    path = tmp_path / "prompts.json"
    p3 = {
        **P0,
        "prompt_id": "P3",
        "parent_prompt_id": "P0",
        "status": "archived",
    }
    write_registry(path, [P0, p3])
    registry = PromptRegistry(path)

    candidate = registry.register_candidate("New prompt", "New evidence")

    assert candidate.prompt_id == "P4"


def test_manual_activation_archives_previous_baseline(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)
    candidate = registry.register_candidate("Candidate prompt", "Validation pending")

    activated = registry.activate(candidate.prompt_id)

    assert activated.status == "active"
    assert registry.get("P0").status == "archived"
    assert PromptRegistry(path).active_prompt.prompt_id == "P1"


def test_rejects_non_active_candidate(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)
    candidate = registry.register_candidate("Candidate prompt", "Not better")

    rejected = registry.reject(candidate.prompt_id)

    assert rejected.status == "rejected"
    assert registry.active_prompt.prompt_id == "P0"
    with pytest.raises(PromptRegistryError, match="rejected prompt cannot be activated"):
        registry.activate(candidate.prompt_id)


def test_active_prompt_cannot_be_rejected(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)

    with pytest.raises(PromptRegistryError, match="active prompt cannot be rejected"):
        registry.reject("P0")


@pytest.mark.parametrize(
    ("prompts", "message"),
    [
        ([], "at least one prompt"),
        ([P0, P0], "duplicate prompt IDs"),
        ([{**P0, "status": "candidate"}], "exactly one active prompt"),
        (
            [P0, {**P0, "prompt_id": "P1"}],
            "exactly one active prompt",
        ),
        ([{**P0, "status": "unknown"}], "status must be one of"),
        (
            [
                P0,
                {
                    **P0,
                    "prompt_id": "P1",
                    "parent_prompt_id": "P9",
                    "status": "candidate",
                },
            ],
            "references unknown parent P9",
        ),
        (
            [{**P0, "training_metrics": {"accuracy": "high"}}],
            "training_metrics.accuracy.*numeric value",
        ),
    ],
)
def test_rejects_invalid_registry_data(tmp_path, prompts, message):
    path = tmp_path / "prompts.json"
    write_registry(path, prompts)

    with pytest.raises(PromptRegistryError, match=message):
        PromptRegistry(path)


def test_unknown_prompt_id_has_clear_error(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)

    with pytest.raises(PromptRegistryError, match="unknown prompt ID: P9"):
        registry.get("P9")


def test_trigger_link_is_persistent_and_unique(tmp_path):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)
    candidate = registry.register_candidate("Candidate", "Reason", trigger_decision_id="decision-1")
    assert PromptRegistry(path).candidate_for_trigger("decision-1") == candidate
    assert PromptRegistry(path).get("P0").trigger_decision_id is None
    with pytest.raises(PromptRegistryError, match="duplicate trigger_decision_id"):
        registry.register_candidate("Another candidate", "Reason", trigger_decision_id="decision-1")
    assert len(PromptRegistry(path).versions) == 2


@pytest.mark.parametrize("invalid", ["", " ", 42, []])
def test_invalid_trigger_link_does_not_change_registry(tmp_path, invalid):
    path = tmp_path / "prompts.json"
    write_registry(path)
    registry = PromptRegistry(path)
    with pytest.raises(PromptRegistryError, match="trigger_decision_id"):
        registry.register_candidate("Candidate", "Reason", trigger_decision_id=invalid)
    assert len(PromptRegistry(path).versions) == 1
