from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Dict, Iterable, Optional, Tuple


SCHEMA_VERSION = 1
VALID_STATUSES = frozenset({"active", "candidate", "rejected", "archived"})
PROMPT_ID_PATTERN = re.compile(r"^P(\d+)$")
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "prompts.json"


class PromptRegistryError(ValueError):
    """Raised when prompt registry data or an operation is invalid."""


def normalize_prompt(text: str) -> str:
    """Ignore capitalization and whitespace when detecting repeated proposals."""
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class PromptVersion:
    prompt_id: str
    prompt_text: str
    parent_prompt_id: Optional[str]
    created_at: str
    change_reason: str
    status: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    trigger_decision_id: Optional[str] = None


def _metric_map(value, field_name: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PromptRegistryError(f"'{field_name}' must be an object")

    metrics = {}
    for name, score in value.items():
        if not isinstance(name, str) or not name.strip():
            raise PromptRegistryError(f"'{field_name}' keys must be non-empty strings")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise PromptRegistryError(
                f"'{field_name}.{name}' must be a numeric value"
            )
        metrics[name] = float(score)
    return metrics


def _required_string(record: dict, field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PromptRegistryError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _prompt_from_record(record, index: int) -> PromptVersion:
    if not isinstance(record, dict):
        raise PromptRegistryError(f"prompt record {index} must be an object")

    prompt_id = _required_string(record, "prompt_id")
    if PROMPT_ID_PATTERN.fullmatch(prompt_id) is None:
        raise PromptRegistryError(
            f"prompt record {index} has invalid prompt_id '{prompt_id}'"
        )

    parent_prompt_id = record.get("parent_prompt_id")
    if parent_prompt_id is not None:
        if not isinstance(parent_prompt_id, str) or not parent_prompt_id.strip():
            raise PromptRegistryError(
                f"prompt record {index} parent_prompt_id must be null or a prompt ID"
            )
        parent_prompt_id = parent_prompt_id.strip()
        if PROMPT_ID_PATTERN.fullmatch(parent_prompt_id) is None:
            raise PromptRegistryError(
                f"prompt record {index} has invalid parent_prompt_id '{parent_prompt_id}'"
            )

    status = _required_string(record, "status")
    if status not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise PromptRegistryError(
            f"prompt record {index} status must be one of: {allowed}"
        )

    return PromptVersion(
        prompt_id=prompt_id,
        prompt_text=_required_string(record, "prompt_text"),
        parent_prompt_id=parent_prompt_id,
        created_at=_required_string(record, "created_at"),
        change_reason=_required_string(record, "change_reason"),
        status=status,
        training_metrics=_metric_map(
            record.get("training_metrics"), "training_metrics"
        ),
        validation_metrics=_metric_map(
            record.get("validation_metrics"), "validation_metrics"
        ),
        trigger_decision_id=record.get("trigger_decision_id"),
    )


def _validate_versions(versions: Iterable[PromptVersion]) -> Tuple[PromptVersion, ...]:
    versions = tuple(versions)
    if not versions:
        raise PromptRegistryError("prompt registry must contain at least one prompt")

    prompt_ids = [version.prompt_id for version in versions]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise PromptRegistryError("prompt registry contains duplicate prompt IDs")

    known_ids = set(prompt_ids)
    trigger_ids = set()
    for version in versions:
        trigger_id = version.trigger_decision_id
        if trigger_id is not None:
            if not isinstance(trigger_id, str) or not trigger_id.strip():
                raise PromptRegistryError("trigger_decision_id must be a non-empty string or null")
            if trigger_id in trigger_ids:
                raise PromptRegistryError("duplicate trigger_decision_id in prompt registry")
            trigger_ids.add(trigger_id)
        if version.parent_prompt_id is not None:
            if version.parent_prompt_id == version.prompt_id:
                raise PromptRegistryError(
                    f"prompt {version.prompt_id} cannot be its own parent"
                )
            if version.parent_prompt_id not in known_ids:
                raise PromptRegistryError(
                    f"prompt {version.prompt_id} references unknown parent "
                    f"{version.parent_prompt_id}"
                )

    active = [version for version in versions if version.status == "active"]
    if len(active) != 1:
        raise PromptRegistryError("prompt registry must contain exactly one active prompt")

    return versions


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PromptRegistry:
    """Persistent version store for code-generator system prompts."""

    def __init__(self, path):
        self.path = Path(path)
        self._versions = self._read()

    def _read(self) -> Tuple[PromptVersion, ...]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PromptRegistryError(
                f"could not read prompt registry {self.path}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PromptRegistryError(
                f"prompt registry contains invalid JSON: {exc.msg}"
            ) from exc

        if not isinstance(data, dict):
            raise PromptRegistryError("prompt registry root must be an object")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise PromptRegistryError(
                f"prompt registry schema_version must be {SCHEMA_VERSION}"
            )
        records = data.get("prompts")
        if not isinstance(records, list):
            raise PromptRegistryError("prompt registry 'prompts' must be a list")

        versions = (
            _prompt_from_record(record, index)
            for index, record in enumerate(records, start=1)
        )
        return _validate_versions(versions)

    def _write(self, versions: Iterable[PromptVersion]) -> None:
        validated = _validate_versions(versions)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "prompts": [asdict(version) for version in validated],
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise PromptRegistryError(
                f"could not write prompt registry {self.path}: {exc}"
            ) from exc

        self._versions = validated

    @property
    def versions(self) -> Tuple[PromptVersion, ...]:
        return self._versions

    def refresh(self) -> None:
        self._versions = self._read()

    def rejected_duplicate(self, prompt_text: str) -> Optional[PromptVersion]:
        normalized = normalize_prompt(prompt_text)
        return next((v for v in self._versions
                     if v.status == "rejected" and normalize_prompt(v.prompt_text) == normalized), None)

    def candidate_for_trigger(self, decision_id: str) -> Optional[PromptVersion]:
        return next((v for v in self._versions if v.trigger_decision_id == decision_id), None)

    @property
    def active_prompt(self) -> PromptVersion:
        return next(
            version for version in self._versions if version.status == "active"
        )

    def get(self, prompt_id: str) -> PromptVersion:
        for version in self._versions:
            if version.prompt_id == prompt_id:
                return version
        raise PromptRegistryError(f"unknown prompt ID: {prompt_id}")

    def _next_prompt_id(self) -> str:
        numbers = [
            int(PROMPT_ID_PATTERN.fullmatch(version.prompt_id).group(1))
            for version in self._versions
        ]
        return f"P{max(numbers) + 1}"

    def register_candidate(
        self,
        prompt_text: str,
        change_reason: str,
        parent_prompt_id: Optional[str] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        created_at: Optional[str] = None,
        trigger_decision_id: Optional[str] = None,
    ) -> PromptVersion:
        """Persist a candidate without changing the current active prompt."""
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise PromptRegistryError("candidate prompt_text must be non-empty")
        if not isinstance(change_reason, str) or not change_reason.strip():
            raise PromptRegistryError("candidate change_reason must be non-empty")

        parent_id = parent_prompt_id or self.active_prompt.prompt_id
        self.get(parent_id)
        candidate = PromptVersion(
            prompt_id=self._next_prompt_id(),
            prompt_text=prompt_text.strip(),
            parent_prompt_id=parent_id,
            created_at=created_at or _utc_timestamp(),
            change_reason=change_reason.strip(),
            status="candidate",
            training_metrics=_metric_map(training_metrics, "training_metrics"),
            validation_metrics=_metric_map(
                validation_metrics, "validation_metrics"
            ),
            trigger_decision_id=trigger_decision_id,
        )
        self._write((*self._versions, candidate))
        return candidate

    def activate(self, prompt_id: str) -> PromptVersion:
        """Manually activate a candidate or archived prompt version."""
        target = self.get(prompt_id)
        if target.status == "rejected":
            raise PromptRegistryError("a rejected prompt cannot be activated")
        if target.status == "active":
            return target

        updated = []
        for version in self._versions:
            if version.status == "active":
                updated.append(replace(version, status="archived"))
            elif version.prompt_id == prompt_id:
                updated.append(replace(version, status="active"))
            else:
                updated.append(version)
        self._write(updated)
        return self.get(prompt_id)

    def reject(self, prompt_id: str) -> PromptVersion:
        """Mark a non-active prompt as rejected."""
        target = self.get(prompt_id)
        if target.status == "active":
            raise PromptRegistryError("the active prompt cannot be rejected")
        if target.status == "rejected":
            return target

        updated = [
            replace(version, status="rejected")
            if version.prompt_id == prompt_id
            else version
            for version in self._versions
        ]
        self._write(updated)
        return self.get(prompt_id)


def load_default_registry() -> PromptRegistry:
    return PromptRegistry(DEFAULT_REGISTRY_PATH)
