"""Validates node statistics, tree indexing, pruning, and path extraction."""

import math

import pytest

from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.tree import ThoughtTree


def test_add_child_links_and_indexes() -> None:
    tree = ThoughtTree("task")
    child = tree.add_child(tree.root, "step one")
    assert child.parent is tree.root
    assert child.depth == 1
    assert tree.root.children == [child]
    assert child.id in tree
    assert tree.get(child.id) is child
    assert len(tree) == 2


def test_add_child_rejects_foreign_parent() -> None:
    tree = ThoughtTree("task")
    foreign = ThoughtNode(content="not indexed")
    with pytest.raises(KeyError):
        tree.add_child(foreign, "orphan")


def test_record_visit_accumulates_statistics() -> None:
    node = ThoughtNode(content="step")
    assert node.mean_value == 0.0
    node.record_visit(0.4)
    node.record_visit(0.8)
    assert node.visits == 2
    assert node.mean_value == pytest.approx(0.6)


def test_uct_prefers_unvisited_then_balances() -> None:
    root = ThoughtNode(content="task")
    a = root.attach_child("a")
    b = root.attach_child("b")
    assert a.uct_score(1.414) == math.inf

    root.visits = 10
    a.visits, a.value_sum = 5, 4.0
    b.visits, b.value_sum = 1, 0.9
    expected_a = 0.8 + 1.414 * math.sqrt(math.log(10) / 5)
    expected_b = 0.9 + 1.414 * math.sqrt(math.log(10) / 1)
    assert a.uct_score(1.414) == pytest.approx(expected_a)
    assert b.uct_score(1.414) == pytest.approx(expected_b)
    assert b.uct_score(1.414) > a.uct_score(1.414)


def test_uct_undefined_for_root() -> None:
    with pytest.raises(ValueError):
        ThoughtNode(content="task").uct_score(1.0)


def test_prune_subtree_spares_terminal_descendants() -> None:
    tree = ThoughtTree("task")
    branch = tree.add_child(tree.root, "branch")
    live = tree.add_child(branch, "live leaf")
    live.status = NodeStatus.EVALUATED
    terminal = tree.add_child(branch, "solved leaf")
    terminal.status = NodeStatus.TERMINAL

    changed = tree.prune_subtree(branch)

    assert changed == 2
    assert branch.status is NodeStatus.PRUNED
    assert live.status is NodeStatus.PRUNED
    assert terminal.status is NodeStatus.TERMINAL


def test_best_path_prefers_accepted_terminal() -> None:
    tree = ThoughtTree("task")
    good = tree.add_child(tree.root, "good")
    good.apply_evaluation(0.6, NodeStatus.EVALUATED)
    solved = tree.add_child(good, "solved")
    solved.apply_evaluation(1.0, NodeStatus.TERMINAL)
    decoy = tree.add_child(tree.root, "decoy")
    decoy.apply_evaluation(0.9, NodeStatus.EVALUATED)
    decoy.record_visit(0.9)

    assert tree.best_path() == [tree.root, good, solved]


def test_best_path_falls_back_to_greedy_descent() -> None:
    tree = ThoughtTree("task")
    weak = tree.add_child(tree.root, "weak")
    weak.apply_evaluation(0.3, NodeStatus.EVALUATED)
    weak.record_visit(0.3)
    strong = tree.add_child(tree.root, "strong")
    strong.apply_evaluation(0.8, NodeStatus.EVALUATED)
    strong.record_visit(0.8)
    pruned = tree.add_child(strong, "pruned tail")
    pruned.apply_evaluation(0.1, NodeStatus.PRUNED)

    assert tree.best_path() == [tree.root, strong]


def test_to_dict_round_trips_structure() -> None:
    tree = ThoughtTree("task")
    child = tree.add_child(tree.root, "step")
    child.apply_evaluation(0.5, NodeStatus.EVALUATED, rationale="plausible")

    snapshot = tree.to_dict()

    assert snapshot["size"] == 2
    assert snapshot["root"]["content"] == "task"
    assert snapshot["root"]["children"][0]["status"] == "evaluated"
    assert snapshot["root"]["children"][0]["rationale"] == "plausible"


def test_render_produces_one_line_per_node() -> None:
    tree = ThoughtTree("task")
    a = tree.add_child(tree.root, "a")
    tree.add_child(tree.root, "b")
    tree.add_child(a, "a1")

    rendering = tree.render()

    assert len(rendering.splitlines()) == len(tree)
    assert "<root>" not in rendering  # root content is non-empty here
