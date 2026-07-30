"""Thought node primitives coupling reasoning content with MCTS statistics."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any


@unique
class NodeStatus(Enum):
    """Lifecycle states a thought node moves through during search.

    PENDING marks freshly created nodes awaiting evaluation (the root remains
    PENDING for the whole run, as the task statement itself is never scored).
    TERMINAL is reserved for accepted solution endpoints; terminal thoughts
    that fail the acceptance threshold are pruned instead, since a completed
    line of reasoning cannot be extended.
    """

    PENDING = "pending"
    EVALUATED = "evaluated"
    TERMINAL = "terminal"
    PRUNED = "pruned"
    FAILED = "failed"


LIVE_STATUSES: frozenset[NodeStatus] = frozenset(
    {NodeStatus.PENDING, NodeStatus.EVALUATED}
)


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(slots=True)
class ThoughtNode:
    """Represents a single reasoning step and its accumulated search statistics.

    ``score`` holds the raw evaluator verdict for this node alone, while
    ``visits`` and ``value_sum`` accumulate backpropagated signal from the
    entire subtree beneath it. ``metadata`` is an open extension point for
    later phases (execution payloads, critique records, sandbox verdicts).
    """

    content: str
    parent: ThoughtNode | None = None
    depth: int = 0
    id: str = field(default_factory=_short_id)
    status: NodeStatus = NodeStatus.PENDING
    children: list[ThoughtNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    score: float = 0.0
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_live(self) -> bool:
        """Reports whether the node remains eligible for traversal and expansion."""
        return self.status in LIVE_STATUSES

    @property
    def mean_value(self) -> float:
        """Returns the average backpropagated value, or 0.0 before any visit."""
        return self.value_sum / self.visits if self.visits else 0.0

    def uct_score(self, exploration_weight: float) -> float:
        """Computes the UCT selection score relative to the parent node.

        Unvisited nodes score infinity so selection exhausts every fresh
        candidate before revisiting scored ones.
        """
        if self.parent is None:
            raise ValueError("UCT is undefined for the root node")
        if self.visits == 0:
            return math.inf
        exploration = exploration_weight * math.sqrt(
            math.log(max(self.parent.visits, 1)) / self.visits
        )
        return self.mean_value + exploration

    def attach_child(self, content: str) -> ThoughtNode:
        """Creates, links, and returns a child node one level deeper."""
        child = ThoughtNode(content=content, parent=self, depth=self.depth + 1)
        self.children.append(child)
        return child

    def record_visit(self, value: float) -> None:
        """Folds one backpropagated value sample into the node statistics."""
        self.visits += 1
        self.value_sum += value

    def apply_evaluation(self, score: float, status: NodeStatus, rationale: str = "") -> None:
        """Stores an evaluator verdict and advances the lifecycle status."""
        self.score = score
        self.status = status
        self.rationale = rationale

    def path_from_root(self) -> list[ThoughtNode]:
        """Returns the node chain from the root down to this node, inclusive."""
        path: list[ThoughtNode] = []
        node: ThoughtNode | None = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def walk(self) -> Iterator[ThoughtNode]:
        """Yields this node and every descendant in depth-first order."""
        stack: list[ThoughtNode] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        """Serializes the subtree into a JSON-compatible structure.

        ``include_metadata`` carries the per-node payloads (execution records,
        critiques, reward breakdowns) into the output. It stays off by default
        because live UI snapshots re-serialize the whole tree on every
        backpropagation and have no use for them; run archives switch it on,
        since those payloads are precisely what offline diagnosis needs.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "depth": self.depth,
            "visits": self.visits,
            "value_sum": round(self.value_sum, 6),
            "score": round(self.score, 6),
            "rationale": self.rationale,
            "children": [
                child.to_dict(include_metadata=include_metadata)
                for child in self.children
            ],
        }
        if include_metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], parent: ThoughtNode | None = None
    ) -> ThoughtNode:
        """Reconstructs a node and its subtree from :meth:`to_dict` output.

        Parent links are re-established from the nesting rather than stored,
        so the rebuilt subtree satisfies the same invariants the search core
        maintains. Absent metadata is treated as empty, which keeps archives
        written without it loadable.
        """
        node = cls(
            content=payload["content"],
            parent=parent,
            depth=int(payload["depth"]),
            id=payload["id"],
            status=NodeStatus(payload["status"]),
            visits=int(payload["visits"]),
            value_sum=float(payload["value_sum"]),
            score=float(payload["score"]),
            rationale=payload.get("rationale", ""),
            metadata=dict(payload.get("metadata") or {}),
        )
        node.children = [
            cls.from_dict(child, parent=node) for child in payload.get("children", ())
        ]
        return node

    def __repr__(self) -> str:
        preview = self.content[:40] + ("..." if len(self.content) > 40 else "")
        return (
            f"ThoughtNode(id={self.id!r}, depth={self.depth}, "
            f"status={self.status.value!r}, score={self.score:.3f}, "
            f"visits={self.visits}, content={preview!r})"
        )
