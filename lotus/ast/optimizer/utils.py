"""Shared utilities for LOTUS LazyFrame optimizers.

* :class:`PathEntry` / :data:`PathToLF` — addresses LazyFrames nested
  inside node fields.
* :func:`rewrite_by_path` — generic immutable rewrite over a node tree
  with nested LazyFrames; centralises the recursion and parent-LazyFrame
  reconstruction shared between :class:`GEPAOptimizer` and
  :class:`AccuracyInferenceOptimizer`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..nodes import BaseNode, SourceNode

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame


@dataclass(frozen=True, slots=True)
class PathEntry:
    """One step in the path from a parent lazyframe to a nested LazyFrame.

    Addresses a LazyFrame at ``getattr(node, field_name)`` navigated
    further by ``sub_path``.  When ``sub_path`` is empty the field itself
    is the LazyFrame (e.g. a join's ``right_lf``).  When non-empty the
    sub-path indexes into a nested list/tuple/dict structure (e.g. a
    ``PandasOpNode.lf_args["key"]`` or ``ApplyFnNode.args[0][1]``).
    """

    node_idx: int
    field_name: str = field(default="")
    sub_path: tuple[Any, ...] = field(default=())

    # -- navigation --------------------------------------------------------

    def get_lf(self, node: BaseNode) -> "LazyFrame | None":
        """Extract the nested LazyFrame from *node*."""
        from ..lazyframe import LazyFrame

        root = getattr(node, self.field_name, None)
        if root is None:
            return None
        current = root
        for key in self.sub_path:
            if isinstance(current, (list, tuple)):
                if not isinstance(key, int) or key < 0 or key >= len(current):
                    return None
                current = current[key]
            elif isinstance(current, dict):
                if key not in current:
                    return None
                current = current[key]
            else:
                return None
        return current if isinstance(current, LazyFrame) else None

    def set_lf(self, node: BaseNode, new_lf: "LazyFrame") -> BaseNode:
        """Return a copy of *node* with the nested LazyFrame replaced."""
        if not self.sub_path:
            return node.model_copy(update={self.field_name: new_lf})
        root = getattr(node, self.field_name)
        updated = self._set_nested(root, self.sub_path, new_lf)
        return node.model_copy(update={self.field_name: updated})

    # -- nested-structure utilities ----------------------------------------

    @staticmethod
    def _get_nested(value: Any, path: tuple[Any, ...]) -> Any | None:
        """Navigate a nested list/tuple/dict and return the leaf value."""
        current = value
        for key in path:
            if isinstance(current, (list, tuple)):
                if not isinstance(key, int) or key < 0 or key >= len(current):
                    return None
                current = current[key]
            elif isinstance(current, dict):
                if key not in current:
                    return None
                current = current[key]
            else:
                return None
        return current

    @staticmethod
    def _set_nested(value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
        """Return a shallow copy of *value* with the leaf at *path* replaced."""
        if not path:
            return replacement

        key, rest = path[0], path[1:]

        if isinstance(value, (list, tuple)):
            if not isinstance(key, int) or key < 0 or key >= len(value):
                return value
            items = list(value)
            items[key] = PathEntry._set_nested(items[key], rest, replacement)
            return type(value)(items) if isinstance(value, tuple) else items

        if isinstance(value, dict):
            if key not in value:
                return value
            return {k: (PathEntry._set_nested(v, rest, replacement) if k == key else v) for k, v in value.items()}

        return value

    # -- collection --------------------------------------------------------

    @staticmethod
    def collect(node: BaseNode, node_idx: int) -> "list[tuple[PathEntry, LazyFrame]]":
        """Collect all nested LazyFrame refs from a single node."""
        from ..lazyframe import LazyFrame

        if isinstance(node, SourceNode):
            return []

        results: list[tuple[PathEntry, LazyFrame]] = []

        def _scan(value: Any, fname: str, sp: tuple[Any, ...]) -> None:
            if isinstance(value, LazyFrame):
                results.append((PathEntry(node_idx, fname, sp), value))
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    _scan(item, fname, sp + (idx,))
            elif isinstance(value, dict):
                for k, item in value.items():
                    _scan(item, fname, sp + (k,))

        for fname in type(node).model_fields:
            root = getattr(node, fname, None)
            if root is not None:
                _scan(root, fname, ())

        return results


# Convenience alias used by optimizers walking nested LazyFrame trees.
PathToLF = tuple[PathEntry, ...]


# ---------------------------------------------------------------------------
# Generic path-keyed rewrite
# ---------------------------------------------------------------------------


PathRewriter = Callable[[list[BaseNode], PathToLF], list[BaseNode]]


def rewrite_by_path(
    nodes: list[BaseNode],
    paths: Iterable[PathToLF],
    apply: PathRewriter,
    *,
    path: PathToLF = (),
) -> list[BaseNode]:
    """Recursively rewrite ``nodes`` at every path listed in ``paths``.

    ``apply(nodes_at_path, path)`` is called once for the current level with
    a mutable shallow copy of the node list. The callback performs whatever
    path-local rewrites it needs and returns the updated list. After the
    callback runs, this helper inspects ``paths`` for any direct child paths
    (exactly one ``PathEntry`` deeper than ``path``), recurses into the
    nested ``LazyFrame`` they address, reconstructs that LazyFrame from the
    rewritten child node list, and swaps it back into the parent node via
    :meth:`PathEntry.set_lf`.

    Args:
        nodes: The current node list.
        paths: All paths in the tree where ``apply`` should run. Membership
            and iteration must both be efficient — a ``set``, ``frozenset``,
            ``Mapping``, or ``Mapping.keys()`` view all work.
        apply: Path-local rewrite callback.
        path: Internal — the current path. Top-level callers pass ``()``.

    Returns:
        A new list of nodes with all path-local rewrites applied at every
        level of the tree.
    """
    from ..lazyframe import LazyFrame

    patched = apply(list(nodes), path)

    prefix_len = len(path)
    target_len = prefix_len + 1
    for child_path in paths:
        if len(child_path) != target_len or child_path[:prefix_len] != path:
            continue
        entry = child_path[-1]
        if entry.node_idx < 0 or entry.node_idx >= len(patched):
            continue
        parent_node = patched[entry.node_idx]
        nested_lf = entry.get_lf(parent_node)
        if nested_lf is None:
            continue

        new_nodes = rewrite_by_path(list(nested_lf._nodes), paths, apply, path=child_path)
        new_lf = LazyFrame(_nodes=new_nodes, _source=nested_lf._source)
        patched[entry.node_idx] = entry.set_lf(parent_node, new_lf)

    return patched
