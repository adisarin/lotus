"""Accuracy inference optimizer for LOTUS LazyFrames.

Implements the type inference system from Mohsen Lesani, *Accuracy
Specification and Derivation for Natural Language Relational Algebra*
(Jan 2026, Fig. 2). Given a top-level precision (pi), recall (rho), and
probability (p) target for the whole pipeline, the optimizer walks the AST,
emits the constraints implied by the typing rules, and solves them with Z3
to derive a feasible per-operator (pi, rho, p) assignment. Each
``SemFilterNode`` / ``SemJoinNode`` is then rewritten with a fresh
``CascadeArgs`` carrying its share of the accuracy obligation.

Supported rules: T-Table, T-Var, T-Prod (cross product underlying joins),
T-Sel (filter / join predicate), T-Proj (map / extract). Union (T-Union)
and difference (T-Minus) are intentionally out of scope for now.

The traversal mirrors :class:`GEPAOptimizer` — a recursive ``_walk`` over
nodes plus nested LazyFrames addressed by :class:`PathEntry`, and a
matching ``_apply_at_path`` rewrite that reconstructs parent ``LazyFrame``
nodes from the deepest path upward. This means operators nested inside a
``SemJoinNode.right_lf`` are both constraint-tracked and rewritten in
place.

The optimizer requires ``z3-solver``; the import is lazy so the module can
be imported without it::

    pip install z3-solver
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional, Sequence

import numpy as np
import pandas as pd

from lotus.types import CascadeArgs

from ..nodes import (
    BaseNode,
    PandasFilterNode,
    SemExtractNode,
    SemFilterNode,
    SemJoinNode,
    SemMapNode,
    SourceNode,
)
from .base import BaseOptimizer
from .utils import PathEntry, PathToLF, rewrite_by_path

if TYPE_CHECKING:
    from ..lazyframe import LazyFrame

logger = logging.getLogger(__name__)


# T-Table base case: an exact source has pi = rho = p = 1.
_EXACT: float = 1.0


@dataclass(frozen=True, slots=True)
class AccuracyTarget:
    """Solved per-operator accuracy obligation."""

    pi: float
    rho: float
    p: float


# ---------------------------------------------------------------------------
# Per-operator assignment record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OpAssignment:
    """Where to write back a solved (pi, rho, p) for one operator.

    ``path`` is the ``PathToLF`` from the root LazyFrame down to the list
    containing the node; ``node_idx`` indexes into that list. ``pi``/``rho``
    /``p`` are Z3 ``Real`` symbols that the solver assigns concrete values
    to.
    """

    node_idx: int
    path: PathToLF
    pi: Any
    rho: Any
    p: Any


# ---------------------------------------------------------------------------
# Inference state
# ---------------------------------------------------------------------------


@dataclass
class _InferResult:
    """Accumulated state of the type inference traversal.

    Mirrors the judgement ``Gamma |- q : <pi, rho, p>, c, C`` from the paper
    (cost ``c`` is not yet modeled). ``pi``/``rho``/``p`` are either Z3
    ``Real`` symbols or float constants representing the current accuracy
    type. ``constraints`` is the constraint set ``C``. ``variables`` lists
    every fresh ``Real`` we introduced so the solver can bound them.
    ``assignments`` records each per-operator triple together with its
    location for later rewriting.
    """

    pi: Any = _EXACT
    rho: Any = _EXACT
    p: Any = _EXACT
    constraints: list[Any] = field(default_factory=list)
    variables: list[Any] = field(default_factory=list)
    assignments: list[_OpAssignment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------


class _ConstraintBuilder:
    """Walks a LOTUS AST and emits the inference judgement constraints.

    Each branch of ``_walk`` corresponds to a typing rule in Fig. 2 of the
    paper. Recursion into ``SemJoinNode.right_lf`` happens via ``PathEntry``
    just like in :class:`GEPAOptimizer`.
    """

    def __init__(self) -> None:
        # Lazy: only construct the builder once z3 is available.
        from z3 import Real

        self._Real = Real
        self._counter = 0

    def _fresh(self, base: str) -> Any:
        self._counter += 1
        return self._Real(f"{base}_{self._counter}")

    def walk(self, nodes: list[BaseNode]) -> _InferResult:
        return self._walk(nodes, ())

    def _walk(self, nodes: list[BaseNode], path: PathToLF) -> _InferResult:
        logger.debug(
            "AccuracyInferenceOptimizer: walking %d node(s) at path depth %d",
            len(nodes),
            len(path),
        )
        current = _InferResult()

        for node_idx, node in enumerate(nodes):
            if isinstance(node, SourceNode):
                # T-Table: <1, 1, 1>, identity element of the chain.
                logger.debug(
                    "AccuracyInferenceOptimizer: T-Table at idx=%d (path depth=%d) -> <1, 1, 1>",
                    node_idx,
                    len(path),
                )
                continue

            if isinstance(node, SemFilterNode):
                # T-Sel
                logger.debug(
                    "AccuracyInferenceOptimizer: T-Sel for SemFilterNode at idx=%d (path depth=%d)",
                    node_idx,
                    len(path),
                )
                current = self._apply_t_sel(current, node_idx, path, label="sem_filter")
                continue

            if isinstance(node, SemJoinNode):
                # SemJoin = sigma_nu(q1 x q2): T-Prod followed by T-Sel.
                logger.debug(
                    "AccuracyInferenceOptimizer: SemJoinNode at idx=%d (path depth=%d) -> descending into right_lf",
                    node_idx,
                    len(path),
                )
                right = self._walk_right_lf(node, node_idx, path)
                prod = self._apply_t_prod(current, right)
                logger.debug(
                    "AccuracyInferenceOptimizer: T-Sel for SemJoinNode predicate at idx=%d (path depth=%d)",
                    node_idx,
                    len(path),
                )
                current = self._apply_t_sel(prod, node_idx, path, label="sem_join")
                continue

            if isinstance(node, (SemMapNode, SemExtractNode)):
                # T-Proj-like: map/extract add or transform columns without
                # filtering rows, so the row-level accuracy type is unchanged.
                logger.debug(
                    "AccuracyInferenceOptimizer: T-Proj for %s at idx=%d (path depth=%d) -> propagate",
                    type(node).__name__,
                    node_idx,
                    len(path),
                )
                continue


            logger.debug(
                "AccuracyInferenceOptimizer: skipping unsupported/irrelevant node %s",
                type(node).__name__,
            )

        return current

    def _walk_right_lf(
        self,
        join_node: SemJoinNode,
        join_idx: int,
        parent_path: PathToLF,
    ) -> _InferResult:
        right_lf = getattr(join_node, "right_lf", None)
        if right_lf is None or not hasattr(right_lf, "_nodes"):
            logger.debug(
                "AccuracyInferenceOptimizer: SemJoinNode at idx=%d has no right_lf, "
                "treating right side as exact",
                join_idx,
            )
            return _InferResult()
        entry = PathEntry(node_idx=join_idx, field_name="right_lf")
        return self._walk(list(right_lf._nodes), parent_path + (entry,))

    def _apply_t_sel(
        self,
        child: _InferResult,
        node_idx: int,
        path: PathToLF,
        label: str,
    ) -> _InferResult:
        """T-Sel: <pi1, rho1, p1> -> <pi1*pi, rho1*rho, p1*p> with fresh pi, rho, p."""
        pi_op = self._fresh(f"pi_{label}")
        rho_op = self._fresh(f"rho_{label}")
        p_op = self._fresh(f"p_{label}")

        pi_out = self._fresh("pi_out")
        rho_out = self._fresh("rho_out")
        p_out = self._fresh("p_out")

        constraints = child.constraints + [
            pi_out == child.pi * pi_op,
            rho_out == child.rho * rho_op,
            p_out == child.p * p_op,
        ]
        variables = child.variables + [pi_op, rho_op, p_op, pi_out, rho_out, p_out]
        assignments = child.assignments + [
            _OpAssignment(node_idx=node_idx, path=path, pi=pi_op, rho=rho_op, p=p_op)
        ]
        logger.debug(
            "AccuracyInferenceOptimizer:   fresh op vars %s, %s, %s; "
            "out vars %s, %s, %s; total ops so far: %d",
            pi_op,
            rho_op,
            p_op,
            pi_out,
            rho_out,
            p_out,
            len(assignments),
        )

        return _InferResult(
            pi=pi_out,
            rho=rho_out,
            p=p_out,
            constraints=constraints,
            variables=variables,
            assignments=assignments,
        )

    def _apply_t_prod(self, left: _InferResult, right: _InferResult) -> _InferResult:
        """T-Prod for the Cartesian-product step underlying SemJoinNode.

        The paper's full T-Prod also introduces a fresh per-product (pi, rho,
        p). For LOTUS' join, the cross product itself is exact, and the
        natural-language predicate's accuracy is captured by the T-Sel that
        the caller applies immediately after. We therefore only propagate
        ``<pi1*pi2, rho1*rho2, p1*p2>`` here and keep the operator-level
        fresh variables on the T-Sel side.
        """
        pi_out = self._fresh("pi_prod")
        rho_out = self._fresh("rho_prod")
        p_out = self._fresh("p_prod")

        constraints = left.constraints + right.constraints + [
            pi_out == left.pi * right.pi,
            rho_out == left.rho * right.rho,
            p_out == left.p * right.p,
        ]
        variables = left.variables + right.variables + [pi_out, rho_out, p_out]
        assignments = left.assignments + right.assignments
        logger.debug(
            "AccuracyInferenceOptimizer: T-Prod merging left/right -> out vars %s, %s, %s "
            "(cumulative ops: %d)",
            pi_out,
            rho_out,
            p_out,
            len(assignments),
        )

        return _InferResult(
            pi=pi_out,
            rho=rho_out,
            p=p_out,
            constraints=constraints,
            variables=variables,
            assignments=assignments,
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _model_value_to_float(model: Any, var: Any) -> float:
    val = model.eval(var, model_completion=True)
    try:
        return float(val.as_fraction())
    except Exception:
        return float(val.as_decimal(6).rstrip("?"))


def _z3_float(value: float) -> Any:
    from z3 import RealVal

    return RealVal(str(float(value)))


def _budget_to_target(budget: float, min_epsilon: float) -> float:
    return min(1.0, max(min_epsilon, math.exp(-budget)))


def _solve_uniform_z3(
    operator_keys: Sequence[Hashable],
    target_pi: float,
    target_rho: float,
    target_p: float,
    min_epsilon: float,
) -> Optional[dict[Hashable, AccuracyTarget]]:
    """Allocate the product-rule budget equally across operators (cost-blind).

    Used when no per-operator cost function is supplied. Each operator
    receives the same combined error budget
    ``(pi_i * rho_i * p_i) = (target_pi * target_rho * target_p) ** (1/N)``,
    with a deterministic tiebreaker so the per-axis split is reproducible.
    """
    from z3 import Optimize, Real, RealVal, Sum, sat

    max_budget = -math.log(min_epsilon)
    optimizer = Optimize()
    optimizer.set(priority="lex")

    budgets: dict[Hashable, tuple[Any, Any, Any]] = {}
    for idx, key in enumerate(operator_keys):
        pi_budget = Real(f"pi_budget_{idx}")
        rho_budget = Real(f"rho_budget_{idx}")
        p_budget = Real(f"p_budget_{idx}")
        for budget in (pi_budget, rho_budget, p_budget):
            optimizer.add(budget >= RealVal("0"))
            optimizer.add(budget <= _z3_float(max_budget))
        budgets[key] = (pi_budget, rho_budget, p_budget)

    def add_budget_constraint(metric_budgets: Sequence[Any], target: float) -> None:
        if target <= 0.0:
            total_budget = max_budget * len(operator_keys)
        else:
            total_budget = -math.log(float(target))
        optimizer.add(Sum(metric_budgets) <= _z3_float(total_budget))

    add_budget_constraint([budgets[k][0] for k in operator_keys], target_pi)
    add_budget_constraint([budgets[k][1] for k in operator_keys], target_rho)
    add_budget_constraint([budgets[k][2] for k in operator_keys], target_p)

    combined_budgets = [
        budgets[k][0] + budgets[k][1] + budgets[k][2] for k in operator_keys
    ]
    optimizer.maximize(Sum(combined_budgets))

    if len(combined_budgets) > 1:
        spread = Real("accuracy_target_spread")
        optimizer.add(spread >= RealVal("0"))
        for i, lhs in enumerate(combined_budgets):
            for rhs in combined_budgets[i + 1 :]:
                optimizer.add(spread >= lhs - rhs)
                optimizer.add(spread >= rhs - lhs)
        optimizer.minimize(spread)

    deterministic_tiebreaker = Sum(
        [RealVal(str(i + 1)) * budget for i, budget in enumerate(combined_budgets)]
    )
    optimizer.maximize(deterministic_tiebreaker)

    if optimizer.check() != sat:
        return None

    model = optimizer.model()
    return {
        key: AccuracyTarget(
            pi=_budget_to_target(
                _model_value_to_float(model, budgets[key][0]), min_epsilon
            ),
            rho=_budget_to_target(
                _model_value_to_float(model, budgets[key][1]), min_epsilon
            ),
            p=_budget_to_target(
                _model_value_to_float(model, budgets[key][2]), min_epsilon
            ),
        )
        for key in operator_keys
    }


def _solve_with_cost_fn(
    operator_keys: Sequence[Hashable],
    target_pi: float,
    target_rho: float,
    target_p: float,
    min_epsilon: float,
    cost_fn: Callable[[Hashable, float, float, float], float],
) -> Optional[dict[Hashable, AccuracyTarget]]:
    """Minimize ``sum_i cost_fn(key_i, pi_i, rho_i, p_i)`` under product
    constraints, via SLSQP in log-budget space.

    Variables (per operator): ``b_pi_i, b_rho_i, b_p_i`` with
    ``pi_i = exp(-b_pi_i)`` (and likewise for rho/p). Constraints are linear
    in budgets: ``sum_i b_pi_i <= -log(target_pi)`` etc. with each
    ``b_*_i in [0, -log(min_epsilon)]``.
    """
    from scipy.optimize import minimize

    n = len(operator_keys)
    max_budget = -math.log(min_epsilon)

    def _target_log_budget(target: float) -> float:
        if target <= 0.0:
            return max_budget * n
        return -math.log(float(min(1.0, target)))

    log_pi = _target_log_budget(target_pi)
    log_rho = _target_log_budget(target_rho)
    log_p = _target_log_budget(target_p)

    # Equal initial allocation per axis -- already feasible.
    x0 = np.empty(3 * n, dtype=float)
    x0[0::3] = log_pi / n
    x0[1::3] = log_rho / n
    x0[2::3] = log_p / n

    bounds = [(0.0, max_budget)] * (3 * n)

    def _objective(x: np.ndarray) -> float:
        total = 0.0
        for i, key in enumerate(operator_keys):
            pi_i = _budget_to_target(float(x[3 * i]), min_epsilon)
            rho_i = _budget_to_target(float(x[3 * i + 1]), min_epsilon)
            p_i = _budget_to_target(float(x[3 * i + 2]), min_epsilon)
            try:
                c = float(cost_fn(key, pi_i, rho_i, p_i))
            except Exception:
                c = float("inf")
            if not math.isfinite(c):
                c = 1e12
            total += c
        return total

    # Product-rule constraints translate to linear inequalities in log-budget
    # space: ``-log(target) - sum(budgets) >= 0``. SciPy treats "ineq"
    # constraints as ``fun(x) >= 0``.
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x, axis=axis, cap=cap: cap
            - float(np.sum(x[axis::3])),
        }
        for axis, cap in ((0, log_pi), (1, log_rho), (2, log_p))
    ]

    result = minimize(
        _objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-7, "disp": False},
    )

    if not result.success:
        # The optimizer reached its iteration cap or got stuck on a constraint
        # boundary. The starting point is already feasible (uniform split), so
        # fall back to that rather than returning ``None``.
        x_use = x0
    else:
        x_use = np.asarray(result.x, dtype=float)

    # F7: the EE cost objective is piecewise-constant (discrete thresholds + min
    # over layers), so SLSQP often gets ~0 gradient and either fails (falls back
    # to x0) or returns a point ~= x0 -- i.e. the "cost-aware" split silently
    # degenerates to the uniform/geometric one. Log it so a reader knows when the
    # allocation is actually uniform (not a bug: the guarantee still holds).
    import logging as _logging

    _is_uniform = bool(np.allclose(x_use, np.asarray(x0, dtype=float), atol=1e-9))
    _logging.getLogger(__name__).info(
        "accuracy SLSQP split: success=%s status=%s cost_aware_split_converged=%s "
        "(returned point %s uniform x0)",
        bool(result.success), getattr(result, "status", "?"),
        bool(result.success) and not _is_uniform,
        "==" if _is_uniform else "!=",
    )

    # Numerical slack on the linear constraints: SLSQP can violate by ~1e-8
    # which then propagates into ``pi*rho*p < target`` by a similar amount.
    # Project back onto the feasible polytope along each axis if needed.
    for axis, cap in ((0, log_pi), (1, log_rho), (2, log_p)):
        s = float(np.sum(x_use[axis::3]))
        if s > cap and s > 0.0:
            x_use[axis::3] = x_use[axis::3] * (cap / s)

    out: dict[Hashable, AccuracyTarget] = {}
    for i, key in enumerate(operator_keys):
        out[key] = AccuracyTarget(
            pi=_budget_to_target(float(x_use[3 * i]), min_epsilon),
            rho=_budget_to_target(float(x_use[3 * i + 1]), min_epsilon),
            p=_budget_to_target(float(x_use[3 * i + 2]), min_epsilon),
        )
    return out


def solve_cost_aware_product_targets(
    operator_keys: Sequence[Hashable],
    target_pi: float,
    target_rho: float,
    target_p: float,
    min_epsilon: float,
    *,
    cost_fn: Optional[Callable[[Hashable, float, float, float], float]] = None,
    cost_aware: bool = True,
) -> Optional[dict[Hashable, AccuracyTarget]]:
    """Solve per-operator product-rule accuracy targets.

    Implements the cost rule from Fig. 2 of Lesani, *Accuracy Specification
    and Derivation for Natural Language Relational Algebra* (Jan 2026):

        minimize   sum_i cost_op_i(pi_i, rho_i, p_i)
        subject to product_i pi_i  >= target_pi
                   product_i rho_i >= target_rho
                   product_i p_i   >= target_p
                   pi_i, rho_i, p_i in [min_epsilon, 1]

    The product constraints are linearized in log-budget space (Sum of
    ``-log(pi_i)`` <= ``-log(target_pi)``, etc.) and the per-operator cost
    is supplied by ``cost_fn(key, pi, rho, p) -> float``. ``cost_fn`` is
    free to depend non-linearly on ``(pi, rho, p)``; common Stretto cost
    models (e.g. ``c_proxy*n + c_silver*n*UnsureFraction(pi, rho)``) fit
    naturally.

    When ``cost_fn`` is ``None`` or ``cost_aware=False``, the allocation
    falls back to an equal-split-per-axis uniform Z3 solve, useful as a
    cost-blind baseline.
    """
    if not 0.0 < min_epsilon < 1.0:
        raise ValueError(f"min_epsilon must be in (0, 1), got {min_epsilon!r}")
    if target_pi > 1.0 or target_rho > 1.0 or target_p > 1.0:
        return None
    if len(operator_keys) == 0:
        return (
            {}
            if target_pi <= 1.0 and target_rho <= 1.0 and target_p <= 1.0
            else None
        )

    if not cost_aware or cost_fn is None:
        return _solve_uniform_z3(
            operator_keys=operator_keys,
            target_pi=target_pi,
            target_rho=target_rho,
            target_p=target_p,
            min_epsilon=min_epsilon,
        )

    return _solve_with_cost_fn(
        operator_keys=operator_keys,
        target_pi=target_pi,
        target_rho=target_rho,
        target_p=target_p,
        min_epsilon=min_epsilon,
        cost_fn=cost_fn,
    )


def _solve(
    result: _InferResult,
    target_pi: float,
    target_rho: float,
    target_p: float,
    min_epsilon: float,
) -> list[tuple[_OpAssignment, dict[str, float]]] | None:
    """Solve the constraint system and return a list of solved assignments.

    Returns ``None`` if the targets are infeasible (Z3 returns ``unsat``
    or ``unknown``).
    """
    keys = list(range(len(result.assignments)))
    targets = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=target_pi,
        target_rho=target_rho,
        target_p=target_p,
        min_epsilon=min_epsilon,
        cost_fn=None,
        cost_aware=False,
    )
    if targets is None:
        return None

    return [
        (
            assignment,
            {
                "pi": targets[idx].pi,
                "rho": targets[idx].rho,
                "p": targets[idx].p,
            },
        )
        for idx, assignment in enumerate(result.assignments)
    ]


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class AccuracyInferenceOptimizer(BaseOptimizer):
    """Derives per-operator cascade accuracy targets via SMT.

    Given top-level ``precision_target``, ``recall_target``, and
    ``probability_target``, walks the AST, collects the constraints implied
    by the typing rules in Fig. 2 of the paper, solves them with Z3, and
    rewrites each ``SemFilterNode`` / ``SemJoinNode`` (top-level *and*
    nested inside a join's ``right_lf``) with a fresh ``CascadeArgs`` whose
    ``precision_target``, ``recall_target``, and ``failure_probability``
    reflect that operator's share of the obligation.

    Limitations:

    * Union (``T-Union``) and difference (``T-Minus``) are not yet
      supported.
    * The cost term ``c`` from the paper is not modeled; any feasible
      assignment is returned.
    * Sharing the same node object across multiple locations in the
      LazyFrame tree is not handled — each occurrence emits its own
      constraints and only the last write wins on rewrite.

    Requires ``z3-solver`` (``pip install z3-solver``).
    """

    requires_train_data: bool = False

    def __init__(
        self,
        precision_target: float = 0.9,
        recall_target: float = 0.85,
        probability_target: float = 0.95,
        min_epsilon: float = 0.01,
    ) -> None:
        for name, val in (
            ("precision_target", precision_target),
            ("recall_target", recall_target),
            ("probability_target", probability_target),
        ):
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {val!r}")
        if not 0.0 < min_epsilon < 1.0:
            raise ValueError(f"min_epsilon must be in (0, 1), got {min_epsilon!r}")

        self.precision_target = precision_target
        self.recall_target = recall_target
        self.probability_target = probability_target
        self.min_epsilon = min_epsilon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        nodes: list[BaseNode],
        train_data: dict["LazyFrame", pd.DataFrame] | pd.DataFrame | None = None,
    ) -> list[BaseNode]:
        try:
            import z3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AccuracyInferenceOptimizer requires the z3-solver package. "
                "Install it with: pip install z3-solver"
            ) from exc

        logger.info(
            "AccuracyInferenceOptimizer: starting on %d top-level node(s) with "
            "targets pi=%.4f, rho=%.4f, p=%.4f, min_epsilon=%.4f",
            len(nodes),
            self.precision_target,
            self.recall_target,
            self.probability_target,
            self.min_epsilon,
        )
        builder = _ConstraintBuilder()
        result = builder.walk(nodes)

        if not result.assignments:
            logger.info("AccuracyInferenceOptimizer: no semantic operators found")
            return nodes

        logger.info(
            "AccuracyInferenceOptimizer: collected %d operator(s) and %d constraint(s); "
            "invoking Z3",
            len(result.assignments),
            len(result.constraints),
        )
        solved = _solve(
            result,
            self.precision_target,
            self.recall_target,
            self.probability_target,
            self.min_epsilon,
        )

        if solved is None:
            logger.warning(
                "AccuracyInferenceOptimizer: targets infeasible "
                "(pi=%.2f, rho=%.2f, p=%.2f) for %d operators",
                self.precision_target,
                self.recall_target,
                self.probability_target,
                len(result.assignments),
            )
            return nodes

        return self._apply_solved(nodes, solved)

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    def _apply_solved(
        self,
        nodes: list[BaseNode],
        solved: list[tuple[_OpAssignment, dict[str, float]]],
    ) -> list[BaseNode]:
        """Apply solved (pi, rho, p) values back into the AST.

        Groups by ``path`` and delegates the recursion / parent-LazyFrame
        reconstruction to :func:`rewrite_by_path`.
        """
        by_path: dict[PathToLF, list[tuple[_OpAssignment, dict[str, float]]]] = defaultdict(list)
        for assignment, params in solved:
            by_path[assignment.path].append((assignment, params))

        def apply(nodes_at_path: list[BaseNode], path: PathToLF) -> list[BaseNode]:
            for assignment, params in by_path.get(path, ()):
                nodes_at_path[assignment.node_idx] = self._inject(
                    nodes_at_path[assignment.node_idx], params
                )
            return nodes_at_path

        return rewrite_by_path(nodes, by_path.keys(), apply)

    def _inject(self, node: BaseNode, params: dict[str, float]) -> BaseNode:
        cascade_update = {
            "precision_target": params["pi"],
            "recall_target": params["rho"],
            # CascadeArgs encodes failure probability rather than the paper's
            # success probability p; clamp to handle Z3's decimal truncation
            # that can put p marginally above 1.
            "failure_probability": max(0.0, 1.0 - params["p"]),
        }

        existing = getattr(node, "cascade_args", None)
        new_args = (
            existing.model_copy(update=cascade_update)
            if existing is not None
            else CascadeArgs(**cascade_update)
        )

        sig = getattr(node, "signature", lambda: f"<{type(node).__name__}>")
        logger.info(
            "AccuracyInferenceOptimizer: %s -> pi=%.4f, rho=%.4f, p=%.4f",
            sig(),
            params["pi"],
            params["rho"],
            params["p"],
        )

        return node.model_copy(update={"cascade_args": new_args})
