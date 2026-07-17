"""Tree container providing indexed access, frontier queries, and pruning."""

from __future__ import annotations

from typing import Any, Iterator

from cognitivetree.node import NodeStatus, ThoughtNode

_STATUS_GLYPHS: dict[NodeStatus, str] = {
    NodeStatus.PENDING: "?",
    NodeStatus.EVALUATED: "*",
    NodeStatus.TERMINAL: "#",
    NodeStatus.PRUNED: "x",
    NodeStatus.FAILED: "!",
}


class ThoughtTree:
    """Owns the thought tree rooted at the task statement.

    All node creation flows through :meth:`add_child` so the identifier index
    stays consistent with the linked structure; callers never attach nodes to
    parents directly.
    """

    def __init__(self, root_content: str) -> None:
        self._root = ThoughtNode(content=root_content)
        self._index: dict[str, ThoughtNode] = {self._root.id: self._root}

    @property
    def root(self) -> ThoughtNode:
        return self._root

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._index

    def get(self, node_id: str) -> ThoughtNode:
        """Resolves a node by identifier, raising ``KeyError`` when absent."""
        try:
            return self._index[node_id]
        except KeyError:
            raise KeyError(f"unknown node id {node_id!r}") from None

    def nodes(self) -> Iterator[ThoughtNode]:
        """Yields every node in the tree in depth-first order."""
        return self._root.walk()

    def add_child(self, parent: ThoughtNode, content: str) -> ThoughtNode:
        """Creates a child under ``parent`` and registers it in the index."""
        if parent.id not in self._index:
            raise KeyError(f"parent {parent.id!r} does not belong to this tree")
        child = parent.attach_child(content)
        self._index[child.id] = child
        return child

    def prune_subtree(self, node: ThoughtNode) -> int:
        """Marks ``node`` and every live descendant as pruned.

        Terminal and failed descendants keep their status so the trace of what
        happened inside an abandoned branch survives. Returns the number of
        nodes whose status changed.
        """
        pruned = 0
        for descendant in node.walk():
            if descendant.is_live:
                descendant.status = NodeStatus.PRUNED
                pruned += 1
        return pruned

    def best_terminal(self) -> ThoughtNode | None:
        """Returns the highest-scoring accepted terminal node, if any exists."""
        terminals = [n for n in self.nodes() if n.status is NodeStatus.TERMINAL]
        return max(terminals, key=lambda n: n.score) if terminals else None

    def best_path(self) -> list[ThoughtNode]:
        """Extracts the current best root-to-leaf reasoning chain.

        Prefers the path to the best accepted terminal. Absent one, descends
        greedily through live children ranked by mean backpropagated value,
        with raw score as the tie-breaker for unvisited nodes.
        """
        terminal = self.best_terminal()
        if terminal is not None:
            return terminal.path_from_root()
        path = [self._root]
        node = self._root
        while True:
            live = [c for c in node.children if c.is_live]
            if not live:
                return path
            node = max(live, key=lambda c: (c.mean_value, c.score, c.visits))
            path.append(node)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the full tree into a JSON-compatible snapshot."""
        return {"size": len(self), "root": self._root.to_dict()}

    def render(self) -> str:
        """Formats the tree as an ASCII diagram for terminal inspection."""
        lines: list[str] = []
        self._render_node(self._root, prefix="", is_last=True, lines=lines)
        return "\n".join(lines)

    def _render_node(
        self,
        node: ThoughtNode,
        prefix: str,
        is_last: bool,
        lines: list[str],
    ) -> None:
        glyph = _STATUS_GLYPHS[node.status]
        content = node.content if node.content else "<root>"
        if len(content) > 60:
            content = content[:57] + "..."
        label = (
            f"[{glyph}] d={node.depth} v={node.visits} "
            f"s={node.score:.2f} q={node.mean_value:.2f} | {content}"
        )
        if node.is_root:
            lines.append(label)
            child_prefix = ""
        else:
            connector = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "|   ")
        for index, child in enumerate(node.children):
            self._render_node(
                child,
                prefix=child_prefix,
                is_last=index == len(node.children) - 1,
                lines=lines,
            )
