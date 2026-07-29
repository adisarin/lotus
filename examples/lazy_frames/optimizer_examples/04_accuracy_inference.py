"""Optimizer example: AccuracyInferenceOptimizer derives per-operator targets.

Builds a small LazyFrame pipeline that mixes ``sem_filter``, ``sem_map``,
``sem_extract`` and ``sem_join`` -- with a nested ``sem_filter`` inside the
join's ``right_lf`` -- then asks :class:`AccuracyInferenceOptimizer` for a
feasible per-operator (precision, recall, probability) assignment given
top-level guarantees.

The pipeline is never executed: no LM is configured. The example shows
purely the optimizer's behavior:

1. It walks the AST (including nested ``join.right_lf`` pipelines),
   emitting the constraints implied by the typing rules of Lesani 2026,
   Fig. 2 (T-Table, T-Sel, T-Prod, T-Proj).
2. Z3 solves the resulting product-rule system.
3. Each ``SemFilterNode`` / ``SemJoinNode`` is rewritten in place with a
   fresh ``CascadeArgs`` whose ``precision_target``, ``recall_target`` and
   ``failure_probability`` carry that operator's share of the obligation.

A DEBUG log handler is installed so you can match every step of the dry
run -- T-Table, T-Sel for each ``sem_filter``, T-Prod + T-Sel for the
join, T-Proj for ``sem_map`` / ``sem_extract``, and the final solved
targets -- against the rewritten ``cascade_args``.

Usage:
    pip install z3-solver
    python examples/lazy_frames/optimizer_examples/04_accuracy_inference.py
"""

from __future__ import annotations

import logging
import math
import sys

import pandas as pd

from lotus.ast import LazyFrame
from lotus.ast.nodes import SemFilterNode, SemJoinNode
from lotus.ast.optimizer import AccuracyInferenceOptimizer

# Show every DEBUG line emitted by lotus.ast.optimizer.accuracy_optimizer so
# the constraint-building trace is visible alongside the rewritten cascades.
# Route logging to stdout (matching the print()s below) so the trace and the
# trees stay in the order in which they were produced.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s | %(message)s",
    stream=sys.stdout,
)
logging.getLogger("lotus.ast.optimizer.accuracy_optimizer").setLevel(logging.DEBUG)

example_logger = logging.getLogger("accuracy_inference_example")

# ---------------------------------------------------------------------------
# Sample data (never actually executed -- we only need columns to reference)
# ---------------------------------------------------------------------------

papers = pd.DataFrame(
    {
        "title": [
            "Accuracy specification for natural-language relational algebra",
            "Quick recipe for sourdough bread",
            "Survey of probabilistic query optimization",
        ],
        "abstract": [
            "We propose a type system for accuracy in semantic queries.",
            "Combine flour, water, salt, and a starter; bake at 230C.",
            "A survey of recent work on probabilistic plan optimization.",
        ],
    }
)

authors = pd.DataFrame(
    {
        "name": ["Mohsen Lesani", "Bob Baker", "Liana Soares"],
        "bio": [
            "Programming languages researcher working on accuracy types.",
            "Home baker with twenty years of sourdough experience.",
            "Database systems researcher interested in query optimization.",
        ],
    }
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
#
# After the source LazyFrame (T-Table => <1, 1, 1>):
#
#   step 1: sem_filter   -- T-Sel, fresh (pi_1, rho_1, p_1)
#   step 2: sem_map      -- T-Proj, no fresh variables
#   step 3: sem_extract  -- T-Proj, no fresh variables
#   step 4: sem_join     -- walks right_lf first, which contains
#                            its own sem_filter (T-Sel, (pi_2, rho_2, p_2)),
#                            then T-Prod (no assignment) and T-Sel for the
#                            join predicate (pi_3, rho_3, p_3).
#
# Three SemFilter/SemJoin operators total => three (pi, rho, p) triples
# constrained by the product rule.

right_lf = LazyFrame(df=authors).sem_filter(
    "{bio} indicates the person publishes in databases"
)

pipeline = (
    LazyFrame(df=papers)
    .sem_filter("{abstract} discusses query optimization")
    .sem_map("Summarize {abstract} in one sentence")
    .sem_extract(
        input_cols=["title"],
        output_cols={"main_topic": "the main research topic of the paper"},
    )
    .sem_join(right_lf, "{title:left} was written by {name:right}")
)

print("=" * 72)
print("Pipeline before optimization:\n")
pipeline.print_tree()
print("=" * 72)
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Dry run: configure the optimizer and rewrite the AST. No LM is required.
# ---------------------------------------------------------------------------

PI_TARGET, RHO_TARGET, P_TARGET = 0.5, 0.5, 0.5
optimizer = AccuracyInferenceOptimizer(
    precision_target=PI_TARGET,
    recall_target=RHO_TARGET,
    probability_target=P_TARGET,
)

example_logger.info(
    "Top-level accuracy targets: pi=%.4f, rho=%.4f, p=%.4f",
    PI_TARGET,
    RHO_TARGET,
    P_TARGET,
)

optimized = pipeline.optimize(
    [optimizer],
    auto_include_default_optimizers=False,
)
sys.stdout.flush()

print("\n" + "=" * 72)
print("Pipeline after optimization (same shape; cascade_args attached):\n")
optimized.print_tree()
print("=" * 72)

# ---------------------------------------------------------------------------
# Closed-form expectation for the dry run.
# ---------------------------------------------------------------------------
#
# ``_solve`` calls ``solve_cost_aware_product_targets`` with
# ``cost_aware=False`` and ``cost_weights=None``. The Z3 ``Optimize`` lex
# objective:
#
#   1. Maximizes the *total* log-budget across all (operator, metric)
#      pairs subject to ``sum(pi_budget) <= -log(PI)``, ``sum(rho_budget)
#      <= -log(RHO)``, ``sum(p_budget) <= -log(P)``. With uniform cost
#      weights this forces each metric's total to its cap, so the global
#      product ``prod(pi_i) = PI``, ``prod(rho_i) = RHO``, ``prod(p_i) = P``.
#   2. Minimizes the spread of the *combined* per-op budget
#      ``b_i = pi_budget_i + rho_budget_i + p_budget_i``, which makes
#      ``pi_i * rho_i * p_i`` the same on every operator.
#   3. Picks a deterministic tiebreaker.
#
# The per-metric split inside an op is *not* further constrained, so the
# strict closed-form invariant is on the per-op product:
#
#     pi_i * rho_i * p_i  =  (PI * RHO * P) ** (1 / N)    for every op i
#
# rather than on the individual metrics.

N_OPS = 3
expected_combined = (PI_TARGET * RHO_TARGET * P_TARGET) ** (1 / N_OPS)

print(
    "\nClosed-form expectation (uniform combined budget across "
    f"{N_OPS} ops):\n"
    f"    expected pi_i * rho_i * p_i = "
    f"(PI*RHO*P)^(1/{N_OPS}) = {expected_combined:.4f}"
)

# ---------------------------------------------------------------------------
# Inspect the rewritten CascadeArgs and verify the product rule.
# ---------------------------------------------------------------------------


def collect_solved_cascades(
    lf: LazyFrame, prefix: str = ""
) -> list[tuple[str, SemFilterNode | SemJoinNode]]:
    """Walk the optimized LazyFrame (and any join right_lf) collecting the
    SemFilter/SemJoin nodes that ended up with rewritten ``cascade_args``."""
    out: list[tuple[str, SemFilterNode | SemJoinNode]] = []
    for idx, node in enumerate(lf._nodes):
        if (
            isinstance(node, (SemFilterNode, SemJoinNode))
            and node.cascade_args is not None
        ):
            out.append((f"{prefix}#{idx} {node.signature()}", node))
        if isinstance(node, SemJoinNode) and node.right_lf is not None:
            out.extend(
                collect_solved_cascades(node.right_lf, prefix + "    right_lf/")
            )
    return out


solved = collect_solved_cascades(optimized)

print("\nSolved per-operator cascade arguments:")
for label, node in solved:
    args = node.cascade_args
    assert args is not None
    success = 1.0 - args.failure_probability
    print(
        f"  {label}\n"
        f"      precision_target    = {args.precision_target:.4f}\n"
        f"      recall_target       = {args.recall_target:.4f}\n"
        f"      failure_probability = {args.failure_probability:.4f}  "
        f"(=> success p = {success:.4f})"
    )

# Match each solved triple against the closed-form invariant.
# The Z3 decimal serializer truncates to 6 digits, so allow a small
# tolerance.
TOL = 5e-3
print("\nMatching per-operator product pi*rho*p against expected value:")
for label, node in solved:
    args = node.cascade_args
    assert args is not None
    success_p = 1.0 - args.failure_probability
    combined = args.precision_target * args.recall_target * success_p
    diff = abs(combined - expected_combined)
    print(
        f"  {label}\n"
        f"      pi*rho*p          = {combined:.4f}\n"
        f"      expected          = {expected_combined:.4f}\n"
        f"      |combined - exp.| = {diff:.2e}  (tol={TOL})"
    )
    assert diff < TOL, f"combined budget mismatch on {label}"

# Product rule across all operators -- this is the actual safety guarantee
# the optimizer is supposed to deliver.
product_pi = math.prod(node.cascade_args.precision_target for _, node in solved)
product_rho = math.prod(node.cascade_args.recall_target for _, node in solved)
product_p = math.prod(
    1.0 - node.cascade_args.failure_probability for _, node in solved
)

print(
    f"\nProduct rule across {len(solved)} operators:\n"
    f"    product(pi)  = {product_pi:.4f}  >= target {PI_TARGET:.4f}? "
    f"{product_pi >= PI_TARGET - TOL}\n"
    f"    product(rho) = {product_rho:.4f}  >= target {RHO_TARGET:.4f}? "
    f"{product_rho >= RHO_TARGET - TOL}\n"
    f"    product(p)   = {product_p:.4f}   >= target {P_TARGET:.4f}? "
    f"{product_p >= P_TARGET - TOL}"
)

assert product_pi >= PI_TARGET - TOL
assert product_rho >= RHO_TARGET - TOL
assert product_p >= P_TARGET - TOL

# Match expected operator count against what the DEBUG trace announced.
ASSIGNMENT_COUNT_FROM_TRACE = 3
assert len(solved) == ASSIGNMENT_COUNT_FROM_TRACE, (
    "Solved op count does not match the constraint-building trace: "
    f"{len(solved)} != {ASSIGNMENT_COUNT_FROM_TRACE}"
)

print("\nAll checks passed.")
