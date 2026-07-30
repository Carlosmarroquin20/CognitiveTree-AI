"""Validates run archival, rehydration fidelity, and format guarding."""

import json
from pathlib import Path

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.llm.demo import TASK, build_offline_controller
from cognitivetree.node import NodeStatus
from cognitivetree.observability import RunMetrics
from cognitivetree.persistence import (
    ARCHIVE_FORMAT,
    ARCHIVE_VERSION,
    ArchiveFormatError,
    load_run,
    result_to_archive_document,
    save_run,
)
from cognitivetree.search import SearchOutcome, TreeSearchController
from cognitivetree.tree import ThoughtTree


@pytest.fixture
def solved_result():
    return build_offline_controller().run(TASK)


@pytest.fixture
def timed_out_result():
    from tests.test_search import NeverTerminalEvaluator, SlowGenerator

    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=50,
            max_depth=50,
            branching_factor=2,
            max_wall_seconds=0.02,
            seed=1,
        ),
        generator=SlowGenerator(delay_seconds=0.15),
        evaluator=NeverTerminalEvaluator(),
    )
    return controller.run("a slow task")


class TestTreeSerialization:
    """Round-trip guarantees of the node and tree serializers."""

    def test_metadata_is_excluded_by_default(self, solved_result) -> None:
        payload = solved_result.tree.to_dict()
        assert "metadata" not in payload["root"]
        assert all("metadata" not in c for c in payload["root"]["children"])

    def test_metadata_is_included_on_request(self, solved_result) -> None:
        payload = solved_result.tree.to_dict(include_metadata=True)
        child = payload["root"]["children"][0]
        assert set(child["metadata"]) >= {"execution", "critique"}

    def test_round_trip_is_byte_identical(self, solved_result) -> None:
        payload = solved_result.tree.to_dict(include_metadata=True)
        assert ThoughtTree.from_dict(payload).to_dict(include_metadata=True) == payload

    def test_round_trip_restores_structure_and_links(self, solved_result) -> None:
        rebuilt = ThoughtTree.from_dict(
            solved_result.tree.to_dict(include_metadata=True)
        )
        assert len(rebuilt) == len(solved_result.tree)
        assert rebuilt.render() == solved_result.tree.render()
        assert rebuilt.root.parent is None
        for child in rebuilt.root.children:
            assert child.parent is rebuilt.root
            assert rebuilt.get(child.id) is child

    def test_round_trip_preserves_statuses_and_scores(self, solved_result) -> None:
        rebuilt = ThoughtTree.from_dict(
            solved_result.tree.to_dict(include_metadata=True)
        )
        original = {n.id: n for n in solved_result.tree.nodes()}
        for node in rebuilt.nodes():
            source = original[node.id]
            assert node.status is source.status
            assert node.score == pytest.approx(source.score)
            assert node.visits == source.visits
            assert node.value_sum == pytest.approx(source.value_sum)

    def test_tree_missing_metadata_key_loads_as_empty(self) -> None:
        # Archives written without metadata must stay readable.
        tree = ThoughtTree("task")
        tree.add_child(tree.root, "step")
        rebuilt = ThoughtTree.from_dict(tree.to_dict())
        assert all(node.metadata == {} for node in rebuilt.nodes())


class TestArchiveRoundTrip:
    """Fidelity of a full save/load cycle."""

    def test_solved_run_round_trips(self, solved_result, tmp_path: Path) -> None:
        path = save_run(solved_result, tmp_path / "run.json")
        restored = load_run(path).result

        assert restored.outcome is SearchOutcome.SUCCEEDED
        assert restored.solution == solved_result.solution
        assert restored.iterations == solved_result.iterations
        assert restored.node_count == solved_result.node_count
        assert restored.error == solved_result.error

    def test_timed_out_run_round_trips_with_its_cause(
        self, timed_out_result, tmp_path: Path
    ) -> None:
        path = save_run(timed_out_result, tmp_path / "timeout.json")
        restored = load_run(path).result

        assert restored.outcome is SearchOutcome.TIMED_OUT
        assert restored.solution is None
        assert "wall-clock budget" in restored.phase_history[-1].note

    def test_phase_history_round_trips_exactly(self, solved_result, tmp_path: Path) -> None:
        path = save_run(solved_result, tmp_path / "run.json")
        restored = load_run(path).result

        assert len(restored.phase_history) == len(solved_result.phase_history)
        for loaded, source in zip(
            restored.phase_history, solved_result.phase_history, strict=True
        ):
            assert loaded.source is source.source
            assert loaded.target is source.target
            assert loaded.note == source.note
            assert loaded.timestamp == pytest.approx(source.timestamp)

    def test_best_path_resolves_to_tree_nodes(self, solved_result, tmp_path: Path) -> None:
        path = save_run(solved_result, tmp_path / "run.json")
        restored = load_run(path).result

        assert [n.id for n in restored.best_path] == [
            n.id for n in solved_result.best_path
        ]
        # Path entries must be the tree's own nodes, not detached copies.
        for node in restored.best_path:
            assert restored.tree.get(node.id) is node

    def test_metrics_survive_the_round_trip(self, solved_result, tmp_path: Path) -> None:
        original = RunMetrics.from_result(solved_result)
        path = save_run(solved_result, tmp_path / "run.json", metrics=original.to_dict())
        archive = load_run(path)

        assert archive.metrics == original.to_dict()
        # Recomputing from the rehydrated result must agree with the original,
        # which only holds if node metadata round-tripped.
        assert RunMetrics.from_result(archive.result).to_dict() == original.to_dict()

    def test_revision_accounting_requires_metadata_fidelity(
        self, solved_result, tmp_path: Path
    ) -> None:
        assert RunMetrics.from_result(solved_result).revisions_granted == 1
        path = save_run(solved_result, tmp_path / "run.json")
        restored = load_run(path).result
        assert RunMetrics.from_result(restored).revisions_granted == 1

    def test_execution_records_are_recoverable(self, solved_result, tmp_path: Path) -> None:
        path = save_run(solved_result, tmp_path / "run.json")
        restored = load_run(path).result

        pruned = [n for n in restored.tree.nodes() if n.status is NodeStatus.PRUNED]
        assert pruned
        for node in pruned:
            assert node.metadata["execution"]["exit_code"] == 1
            assert "critique" in node.metadata

    def test_metrics_are_optional(self, solved_result, tmp_path: Path) -> None:
        archive = load_run(save_run(solved_result, tmp_path / "run.json"))
        assert archive.metrics is None

    def test_task_is_exposed_and_directories_are_created(
        self, solved_result, tmp_path: Path
    ) -> None:
        path = save_run(solved_result, tmp_path / "nested" / "deep" / "run.json")
        assert path.exists()
        assert load_run(path).task == TASK

    def test_document_declares_format_and_version(self, solved_result) -> None:
        document = result_to_archive_document(solved_result)
        assert document["format"] == ARCHIVE_FORMAT
        assert document["version"] == ARCHIVE_VERSION
        assert document["saved_at"].endswith("+00:00")


class TestArchiveValidation:
    """Guarding against documents that are not readable archives."""

    def write(self, tmp_path: Path, document) -> Path:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_invalid_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ArchiveFormatError, match="not valid JSON"):
            load_run(path)

    def test_non_object_document_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ArchiveFormatError, match="JSON object"):
            load_run(self.write(tmp_path, [1, 2, 3]))

    def test_foreign_format_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ArchiveFormatError, match="unrecognized archive format"):
            load_run(self.write(tmp_path, {"format": "something-else", "version": 1}))

    def test_future_version_is_rejected(self, tmp_path: Path) -> None:
        document = {"format": ARCHIVE_FORMAT, "version": ARCHIVE_VERSION + 1}
        with pytest.raises(ArchiveFormatError, match="unsupported archive version"):
            load_run(self.write(tmp_path, document))

    def test_truncated_document_is_rejected(
        self, solved_result, tmp_path: Path
    ) -> None:
        document = result_to_archive_document(solved_result)
        del document["phase_history"]
        with pytest.raises(ArchiveFormatError, match="missing or malformed"):
            load_run(self.write(tmp_path, document))

    def test_unknown_phase_value_is_rejected(
        self, solved_result, tmp_path: Path
    ) -> None:
        document = result_to_archive_document(solved_result)
        document["phase_history"][0]["target"] = "teleporting"
        with pytest.raises(ArchiveFormatError, match="invalid value"):
            load_run(self.write(tmp_path, document))

    def test_dangling_best_path_reference_is_rejected(
        self, solved_result, tmp_path: Path
    ) -> None:
        document = result_to_archive_document(solved_result)
        document["best_path"] = ["nonexistent-node-id"]
        with pytest.raises(ArchiveFormatError, match="missing or malformed"):
            load_run(self.write(tmp_path, document))

    def test_missing_file_raises_os_error(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            load_run(tmp_path / "absent.json")
