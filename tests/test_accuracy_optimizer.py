"""Unit tests for the cost-aware accuracy optimizer.

The solver in :mod:`lotus.ast.optimizer.accuracy_optimizer` lives behind a
single helper, ``solve_cost_aware_product_targets``. Per Fig. 2 of the
spec (Lesani, *Accuracy Specification and Derivation for Natural Language
Relational Algebra*, Jan 2026), the solver minimizes
``sum_i cost_fn(key_i, pi_i, rho_i, p_i)`` subject to the product-rule
constraints (``Pi_i pi_i >= target_pi`` etc.) with per-axis budget
linearization. When ``cost_fn`` is omitted, the solver falls back to an
equal-split-per-axis uniform allocation.

These tests cover:

* Boundary conditions (empty input, infeasible target, bad
  ``min_epsilon``).
* Uniform (cost-blind) behavior when no ``cost_fn`` is supplied or
  ``cost_aware=False``.
* Cost-aware behavior with explicit ``cost_fn`` callables, including the
  expected "loosen the operator whose cost-function shrinks faster".
* Integration on the AST shape used by
  ``examples/.../04_accuracy_inference.py``.
"""

from __future__ import annotations

import math
from typing import Hashable

import pytest

z3 = pytest.importorskip("z3")
pytest.importorskip("scipy")

import pandas as pd  # noqa: E402

from lotus.ast import LazyFrame  # noqa: E402
from lotus.ast.nodes import SemFilterNode, SemJoinNode  # noqa: E402
from lotus.ast.optimizer.accuracy_optimizer import (  # noqa: E402
    AccuracyTarget,
    _budget_to_target,
    _ConstraintBuilder,
    solve_cost_aware_product_targets,
)

# Z3 truncates decimals to 6 digits and SLSQP converges to ~1e-6 ftol, so
# allow a few thousandths of slack across the board.
NUMERICAL_TOL = 5e-3


# ---------------------------------------------------------------------------
# _budget_to_target
# ---------------------------------------------------------------------------


def test_budget_to_target_zero_budget_is_one():
    assert _budget_to_target(0.0, min_epsilon=0.01) == 1.0


def test_budget_to_target_clamps_to_min_epsilon():
    huge_budget = 100.0
    assert _budget_to_target(huge_budget, min_epsilon=0.01) == 0.01


def test_budget_to_target_continuous_inside_range():
    val = _budget_to_target(0.5, min_epsilon=0.01)
    assert val == pytest.approx(math.exp(-0.5))


# ---------------------------------------------------------------------------
# Helpers used by tests
# ---------------------------------------------------------------------------


def _check_product_meets_global(out, keys, target_pi, target_rho, target_p):
    """Assert the per-operator allocation meets the global product rule."""
    assert math.prod(out[k].pi for k in keys) >= target_pi - NUMERICAL_TOL
    assert math.prod(out[k].rho for k in keys) >= target_rho - NUMERICAL_TOL
    assert math.prod(out[k].p for k in keys) >= target_p - NUMERICAL_TOL


def _stretto_shape_cost_fn(weights):
    """A test ``cost_fn`` whose shape matches Stretto's cascade cost.

    ``cost_fn(key, pi, rho, p) = weights[key] * (pi + rho + p)``

    Like the real ``c_proxy * n + c_silver * n * UnsureFraction(pi, rho)``,
    this is *increasing* in each target axis (tighter targets -> larger
    cost) and *convex* in log-budget space (since pi = exp(-b_pi) is
    convex in b_pi). That gives a unique interior optimum whose properties
    are easy to reason about:

    - Equal weights -> uniform per-axis allocation that saturates the
      product constraint (b_pi_i = -log(target_pi)/N for all i, etc.).
    - Higher weight -> the operator pays more per unit of target
      tightness, so the minimizer assigns it a *looser* (smaller) target
      and shifts the slack onto cheaper operators. This is the legacy
      heuristic's direction and matches the Stretto cascade's expected
      behavior: expensive silver -> looser proxy obligation -> less
      fallback.
    """

    def cost_fn(key: Hashable, pi: float, rho: float, p: float) -> float:
        return float(weights[key]) * (pi + rho + p)

    return cost_fn


# ---------------------------------------------------------------------------
# solve_cost_aware_product_targets -- boundary conditions
# ---------------------------------------------------------------------------


def test_solver_rejects_invalid_min_epsilon():
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="min_epsilon"):
            solve_cost_aware_product_targets(
                operator_keys=[0],
                target_pi=0.9,
                target_rho=0.9,
                target_p=0.9,
                min_epsilon=bad,
            )


def test_solver_returns_none_for_infeasible_target_greater_than_one():
    assert (
        solve_cost_aware_product_targets(
            operator_keys=[0, 1],
            target_pi=1.5,
            target_rho=0.9,
            target_p=0.9,
            min_epsilon=0.01,
        )
        is None
    )


def test_solver_empty_keys_returns_empty_dict_when_feasible():
    out = solve_cost_aware_product_targets(
        operator_keys=[],
        target_pi=0.9,
        target_rho=0.9,
        target_p=0.9,
        min_epsilon=0.01,
    )
    assert out == {}


def test_solver_empty_keys_returns_none_when_target_above_one():
    out = solve_cost_aware_product_targets(
        operator_keys=[],
        target_pi=1.5,
        target_rho=0.9,
        target_p=0.9,
        min_epsilon=0.01,
    )
    assert out is None


# ---------------------------------------------------------------------------
# solve_cost_aware_product_targets -- non-cost-aware / uniform behavior
# ---------------------------------------------------------------------------


def test_solver_single_op_receives_global_targets():
    out = solve_cost_aware_product_targets(
        operator_keys=["only"],
        target_pi=0.81,
        target_rho=0.81,
        target_p=0.81,
        min_epsilon=0.01,
        cost_fn=None,
        cost_aware=False,
    )
    assert out is not None
    target = out["only"]
    assert isinstance(target, AccuracyTarget)
    # Single op carries the whole budget, so its targets equal the globals.
    assert target.pi == pytest.approx(0.81, abs=NUMERICAL_TOL)
    assert target.rho == pytest.approx(0.81, abs=NUMERICAL_TOL)
    assert target.p == pytest.approx(0.81, abs=NUMERICAL_TOL)


def test_solver_without_costs_gives_uniform_combined_budget():
    keys = ["a", "b", "c"]
    PI, RHO, P = 0.81, 0.81, 0.9025
    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=PI,
        target_rho=RHO,
        target_p=P,
        min_epsilon=0.01,
        cost_fn=None,
        cost_aware=False,
    )
    assert out is not None

    # Each op should have the same pi*rho*p = (PI*RHO*P)^(1/N), but the per-
    # metric split is not pinned by the solver.
    expected_combined = (PI * RHO * P) ** (1 / len(keys))
    for k in keys:
        combined = out[k].pi * out[k].rho * out[k].p
        assert combined == pytest.approx(expected_combined, abs=NUMERICAL_TOL)

    _check_product_meets_global(out, keys, PI, RHO, P)


def test_solver_cost_aware_false_ignores_cost_fn():
    # ``cost_aware=False`` should produce the same uniform allocation no
    # matter what cost_fn is passed -- this is the "cost-blind" mode.
    keys = ["cheap", "expensive"]
    cost_fn = _stretto_shape_cost_fn(
        {"cheap": 1.0, "expensive": 100.0}
    )
    common = dict(
        operator_keys=keys,
        target_pi=0.81,
        target_rho=0.81,
        target_p=0.81,
        min_epsilon=0.01,
    )
    no_costs = solve_cost_aware_product_targets(**common, cost_aware=False)
    with_costs = solve_cost_aware_product_targets(
        **common, cost_aware=False, cost_fn=cost_fn
    )
    assert no_costs is not None and with_costs is not None
    for k in keys:
        assert no_costs[k].pi == pytest.approx(with_costs[k].pi, abs=NUMERICAL_TOL)
        assert no_costs[k].rho == pytest.approx(with_costs[k].rho, abs=NUMERICAL_TOL)
        assert no_costs[k].p == pytest.approx(with_costs[k].p, abs=NUMERICAL_TOL)


def test_solver_no_cost_fn_means_uniform_even_with_cost_aware_true():
    # Same allocation should pop out whether ``cost_aware`` is on or off,
    # because there is no cost objective to make any operator special.
    keys = [0, 1, 2]
    common = dict(
        operator_keys=keys,
        target_pi=0.81,
        target_rho=0.81,
        target_p=0.9025,
        min_epsilon=0.01,
    )
    uniform = solve_cost_aware_product_targets(**common, cost_aware=False)
    cost_aware_no_fn = solve_cost_aware_product_targets(
        **common, cost_aware=True, cost_fn=None
    )
    assert uniform is not None and cost_aware_no_fn is not None
    for k in keys:
        assert uniform[k].pi == pytest.approx(
            cost_aware_no_fn[k].pi, abs=NUMERICAL_TOL
        )
        assert uniform[k].rho == pytest.approx(
            cost_aware_no_fn[k].rho, abs=NUMERICAL_TOL
        )
        assert uniform[k].p == pytest.approx(
            cost_aware_no_fn[k].p, abs=NUMERICAL_TOL
        )


# ---------------------------------------------------------------------------
# solve_cost_aware_product_targets -- cost-aware behavior
# ---------------------------------------------------------------------------


def test_cost_aware_with_equal_weights_gives_uniform_combined_budget():
    """If all operators have the same cost function, every operator should
    end up with the same combined budget pi*rho*p.

    The per-axis split within an operator is not pinned: the equal-weight
    quadratic objective is symmetric across operators but not across
    axes, so the Z3-based uniform path and the SciPy cost-aware path may
    legitimately partition (pi, rho, p) differently while landing on the
    same combined value. Both must still satisfy the global product
    constraints."""
    keys = [0, 1, 2]
    PI, RHO, P = 0.81, 0.81, 0.9025
    common = dict(
        operator_keys=keys,
        target_pi=PI,
        target_rho=RHO,
        target_p=P,
        min_epsilon=0.01,
    )
    cost_aware = solve_cost_aware_product_targets(
        **common,
        cost_aware=True,
        cost_fn=_stretto_shape_cost_fn({k: 7.0 for k in keys}),
    )
    assert cost_aware is not None

    expected_combined = (PI * RHO * P) ** (1 / len(keys))
    for k in keys:
        combined = cost_aware[k].pi * cost_aware[k].rho * cost_aware[k].p
        assert combined == pytest.approx(expected_combined, abs=NUMERICAL_TOL)

    _check_product_meets_global(cost_aware, keys, PI, RHO, P)


def test_cost_aware_loosens_expensive_op_and_tightens_cheap_op():
    """Stretto-shape cost: cost(pi, rho, p) = w_key * (pi + rho + p) is
    increasing in each target. Higher-weight operators pay more per unit
    of tightness, so the minimizer gives them *smaller* (looser) targets
    and pushes the strict end of the product budget onto the cheaper op.

    This is the same direction as the legacy weighted-looseness heuristic
    and matches Stretto's actual cascade cost
    (c_proxy*n + c_silver*n*UnsureFraction), where UnsureFraction grows
    with target tightness."""
    keys = ["cheap", "expensive"]
    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=0.64,
        target_rho=0.64,
        target_p=0.64,
        min_epsilon=0.01,
        cost_fn=_stretto_shape_cost_fn(
            {"cheap": 1.0, "expensive": 8.0}
        ),
        cost_aware=True,
    )
    assert out is not None
    cheap, expensive = out["cheap"], out["expensive"]

    # Expensive op gets the looser (smaller) per-operator targets.
    assert expensive.pi < cheap.pi
    assert expensive.rho < cheap.rho
    assert expensive.p < cheap.p

    _check_product_meets_global(out, keys, 0.64, 0.64, 0.64)


def test_cost_aware_orders_targets_by_marginal_cost_for_three_ops():
    keys = ["cheap", "medium", "expensive"]
    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=0.5,
        target_rho=0.5,
        target_p=0.5,
        min_epsilon=0.01,
        cost_fn=_stretto_shape_cost_fn(
            {"cheap": 1.0, "medium": 4.0, "expensive": 16.0}
        ),
        cost_aware=True,
    )
    assert out is not None
    cheap, medium, expensive = out["cheap"], out["medium"], out["expensive"]

    # Monotonic relationship: higher marginal cost -> looser (smaller) target.
    assert cheap.pi >= medium.pi - NUMERICAL_TOL >= expensive.pi - 2 * NUMERICAL_TOL
    assert cheap.rho >= medium.rho - NUMERICAL_TOL >= expensive.rho - 2 * NUMERICAL_TOL
    assert cheap.p >= medium.p - NUMERICAL_TOL >= expensive.p - 2 * NUMERICAL_TOL

    _check_product_meets_global(out, keys, 0.5, 0.5, 0.5)


def test_cost_aware_respects_min_epsilon_floor():
    eps = 0.05
    out = solve_cost_aware_product_targets(
        operator_keys=["a", "b"],
        target_pi=0.5,
        target_rho=0.5,
        target_p=0.5,
        min_epsilon=eps,
        cost_fn=_stretto_shape_cost_fn({"a": 1.0, "b": 1000.0}),
        cost_aware=True,
    )
    assert out is not None
    for k in ("a", "b"):
        assert out[k].pi >= eps - 1e-9
        assert out[k].rho >= eps - 1e-9
        assert out[k].p >= eps - 1e-9


def test_cost_aware_minimizes_step_cost_with_threshold_band_objective():
    """Stretto-faithful cost shape: each operator pays a cost proportional
    to (pi + rho + p), modeling the cascade's
    ``c_silver * n * UnsureFraction(pi, rho)`` -- tighter targets force a
    wider unsure band and therefore higher silver fallback cost. With one
    op carrying a much larger silver cost, the minimizer should give that
    op the loosest targets and concentrate the strict end of the product
    budget on the cheap op."""
    keys = ["expensive", "cheap"]

    def cost_fn(key, pi, rho, p):
        weight = 100.0 if key == "expensive" else 1.0
        return weight * (pi + rho + p)

    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=0.64,
        target_rho=0.64,
        target_p=0.64,
        min_epsilon=0.01,
        cost_fn=cost_fn,
        cost_aware=True,
    )
    assert out is not None
    # The expensive operator should end up with materially looser targets
    # than the cheap operator.
    assert out["expensive"].pi <= out["cheap"].pi + NUMERICAL_TOL
    assert out["expensive"].rho <= out["cheap"].rho + NUMERICAL_TOL
    _check_product_meets_global(out, keys, 0.64, 0.64, 0.64)


# ---------------------------------------------------------------------------
# Integration: cost-aware solving on the pipeline shape used by example
# 04_accuracy_inference.py
# ---------------------------------------------------------------------------


def _build_example_pipeline() -> LazyFrame:
    """Reproduce the pipeline from examples/.../04_accuracy_inference.py.

    The shape that matters for the test is:

        source(papers)
          .sem_filter("...")          # T-Sel, assignment[0] at path=()
          .sem_map("...")              # T-Proj
          .sem_extract([...])          # T-Proj
          .sem_join(                   # T-Prod + T-Sel
              LazyFrame(authors).sem_filter("..."),  # T-Sel,
                                                     # assignment[1] at
                                                     # path=(SemJoin/right_lf,)
              "...")                   # T-Sel for join predicate,
                                       # assignment[2] at path=()
    """
    papers = pd.DataFrame(
        {
            "title": ["a", "b", "c"],
            "abstract": ["x", "y", "z"],
        }
    )
    authors = pd.DataFrame(
        {
            "name": ["alice", "bob", "carol"],
            "bio": ["p", "q", "r"],
        }
    )

    right_lf = LazyFrame(df=authors).sem_filter(
        "{bio} indicates the person publishes in databases"
    )
    return (
        LazyFrame(df=papers)
        .sem_filter("{abstract} discusses query optimization")
        .sem_map("Summarize {abstract} in one sentence")
        .sem_extract(
            input_cols=["title"],
            output_cols={"main_topic": "the main research topic"},
        )
        .sem_join(right_lf, "{title:left} was written by {name:right}")
    )


def _collect_operator_keys(pipeline: LazyFrame) -> list[int]:
    """Walk the example pipeline's AST and return solver operator keys.

    Mirrors what :func:`AccuracyInferenceOptimizer._solve` does: an integer
    key per ``_OpAssignment`` produced by the constraint builder, in
    insertion order (top-level sem_filter, nested right_lf sem_filter,
    then the sem_join's T-Sel).
    """
    builder = _ConstraintBuilder()
    result = builder.walk(pipeline._nodes)
    return list(range(len(result.assignments)))


def test_example_pipeline_has_expected_three_assignments():
    """Sanity check: the example pipeline produces exactly three operator
    assignments -- the two sem_filters and the sem_join's T-Sel."""
    keys = _collect_operator_keys(_build_example_pipeline())
    assert keys == [0, 1, 2], (
        "Pipeline shape changed: AccuracyInferenceOptimizer should still "
        "see three (pi, rho, p) operator assignments."
    )


def test_cost_aware_optimization_on_example_pipeline_orders_targets_by_cost():
    """End-to-end: build the example AST, collect operator keys via the
    real constraint builder, then run the cost-aware solver with a
    realistic skew (the sem_join predicate has the highest marginal
    cost, the nested right_lf filter the lowest)."""
    pipeline = _build_example_pipeline()
    keys = _collect_operator_keys(pipeline)
    assert len(keys) == 3
    TOP_FILTER, RIGHT_LF_FILTER, JOIN_PREDICATE = keys

    PI, RHO, P = 0.81, 0.81, 0.9025
    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=PI,
        target_rho=RHO,
        target_p=P,
        min_epsilon=0.01,
        cost_fn=_stretto_shape_cost_fn(
            {TOP_FILTER: 2.0, RIGHT_LF_FILTER: 1.0, JOIN_PREDICATE: 8.0}
        ),
        cost_aware=True,
    )
    assert out is not None

    join = out[JOIN_PREDICATE]
    top = out[TOP_FILTER]
    right = out[RIGHT_LF_FILTER]

    # Highest marginal-cost op (join) ends up LOOSER (smaller target) than
    # the cheaper ops: every unit of tightness there costs the most, so
    # the minimizer dumps slack onto it.
    assert join.pi <= top.pi + NUMERICAL_TOL
    assert join.rho <= top.rho + NUMERICAL_TOL
    assert join.p <= top.p + NUMERICAL_TOL
    assert join.pi <= right.pi + NUMERICAL_TOL
    assert join.rho <= right.rho + NUMERICAL_TOL
    assert join.p <= right.p + NUMERICAL_TOL

    # The cheapest op (right_lf filter) carries the tightest end of the
    # product budget.
    assert right.pi >= top.pi - NUMERICAL_TOL
    assert right.rho >= top.rho - NUMERICAL_TOL
    assert right.p >= top.p - NUMERICAL_TOL

    _check_product_meets_global(out, keys, PI, RHO, P)


def test_cost_aware_optimization_differs_from_uniform_on_example_pipeline():
    """The cost-aware solve must produce a *different* allocation from the
    cost-blind solve on the same pipeline -- otherwise cost-awareness is
    a no-op for this AST shape."""
    pipeline = _build_example_pipeline()
    keys = _collect_operator_keys(pipeline)

    common = dict(
        operator_keys=keys,
        target_pi=0.81,
        target_rho=0.81,
        target_p=0.9025,
        min_epsilon=0.01,
    )

    uniform = solve_cost_aware_product_targets(**common, cost_aware=False)
    cost_aware = solve_cost_aware_product_targets(
        **common,
        cost_aware=True,
        cost_fn=_stretto_shape_cost_fn(
            {k: w for k, w in zip(keys, (1.0, 1.0, 50.0))}
        ),
    )

    assert uniform is not None and cost_aware is not None

    # At least one (op, metric) entry must change between the two modes.
    any_diff = False
    for k in keys:
        for attr in ("pi", "rho", "p"):
            if abs(getattr(uniform[k], attr) - getattr(cost_aware[k], attr)) > 1e-2:
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, (
        "Cost-aware allocation matched the cost-blind allocation -- the "
        "weight skew did not actually shift the budget."
    )


def test_cost_aware_targets_can_be_injected_into_example_pipeline_cascades():
    """Smoke test: solver outputs can be packaged back into CascadeArgs and
    attached to SemFilter/SemJoin nodes the same way the optimizer does
    after `_solve`. Mirrors `AccuracyInferenceOptimizer._inject` but with
    cost-aware values."""
    from lotus.types import CascadeArgs

    pipeline = _build_example_pipeline()
    builder = _ConstraintBuilder()
    result = builder.walk(pipeline._nodes)
    keys = list(range(len(result.assignments)))

    out = solve_cost_aware_product_targets(
        operator_keys=keys,
        target_pi=0.81,
        target_rho=0.81,
        target_p=0.9025,
        min_epsilon=0.01,
        cost_fn=_stretto_shape_cost_fn({0: 2.0, 1: 1.0, 2: 8.0}),
        cost_aware=True,
    )
    assert out is not None

    cascades = []
    for idx, target in out.items():
        ca = CascadeArgs(
            precision_target=target.pi,
            recall_target=target.rho,
            failure_probability=max(0.0, 1.0 - target.p),
        )
        cascades.append((idx, ca))

    sem_filter_nodes = [
        n for n in pipeline._nodes if isinstance(n, SemFilterNode)
    ]
    sem_join_nodes = [n for n in pipeline._nodes if isinstance(n, SemJoinNode)]
    assert len(sem_filter_nodes) == 1
    assert len(sem_join_nodes) == 1
    nested_filter_nodes = [
        n
        for n in sem_join_nodes[0].right_lf._nodes  # type: ignore[union-attr]
        if isinstance(n, SemFilterNode)
    ]
    assert len(nested_filter_nodes) == 1

    for _, ca in cascades:
        assert 0.0 <= ca.precision_target <= 1.0
        assert 0.0 <= ca.recall_target <= 1.0
