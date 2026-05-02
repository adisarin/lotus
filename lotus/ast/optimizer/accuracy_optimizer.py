from __future__ import annotations
import logging
from typing import Any
from z3 import Real, Solver, sat, If
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    from lotus.ast.nodes import (
        BaseNode,
        SemFilterNode,
        SemJoinNode,
        SemMapNode,
        SemExtractNode,
        SourceNode,
        PandasFilterNode,
    )
    from lotus.ast.optimizer.base import BaseOptimizer
    from lotus.types import CascadeArgs

    LOTUS_AVAILABLE = True
except ImportError:
    import pandas as pd
    class BaseNode:
        pass

    class SemFilterNode(BaseNode):
        user_instruction: str = ""
        cascade_args = None
        def model_copy(self, **kw):
            return self

    class SemJoinNode(BaseNode):
        join_instruction: str = ""
        cascade_args = None
        right_lf = None
        def model_copy(self, **kw):
            return self

    class SemMapNode(BaseNode):
        pass

    class SemExtractNode(BaseNode):
        pass

    class SourceNode(BaseNode):
        pass

    class PandasFilterNode(BaseNode):
        pass

    class BaseOptimizer:
        requires_train_data: bool = False

    class CascadeArgs:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def model_copy(self, update=None):
            return self

    LOTUS_AVAILABLE = False

class _InferResult:
    __slots__ = ("pi", "rho", "p", "constraints", "variables", "op_map")

    def __init__(self, pi, rho, p, constraints=None, variables=None, op_map=None):
        self.pi = pi
        self.rho = rho
        self.p = p
        self.constraints = constraints or []
        self.variables = variables or []
        self.op_map = op_map or {}

class _ConstraintBuilder:

    def __init__(self):
        self._id = 0

    def _fresh(self, base: str) -> Real:
        self._id += 1
        return Real(f"{base}_{self._id}")

    def traverse(self, nodes: list) -> _InferResult:
        current = _InferResult(pi=1.0, rho=1.0, p=1.0)

        for idx, node in enumerate(nodes):
            if isinstance(node, SourceNode):
                continue

            elif isinstance(node, SemFilterNode):
                current = self._apply_t_sel(current, idx, 
                    label=f"sem_filter")

            elif isinstance(node, SemJoinNode):
                right_result = self._traverse_right_side(node)
                prod_result = self._apply_t_prod(current, right_result)
                current = self._apply_t_sel(prod_result, idx,
                    label=f"sem_join")

            elif isinstance(node, (SemMapNode, SemExtractNode)):
                continue

            elif isinstance(node, PandasFilterNode):
                continue

            else:
                continue

        return current

    def _traverse_right_side(self, join_node) -> _InferResult:
        right_lf = getattr(join_node, 'right_lf', None)

        if right_lf is not None and hasattr(right_lf, '_nodes'):
            return self.traverse(right_lf._nodes)
        else:
            return _InferResult(pi=1.0, rho=1.0, p=1.0)

    def _apply_t_sel(self, child: _InferResult, node_idx: int, label: str = "sel") -> _InferResult:
        piOp = self._fresh(f"pi_{label}")
        rhoOp = self._fresh(f"rho_{label}")
        pOp = self._fresh(f"p_{label}")

        piOut = self._fresh("pi_out")
        rhoOut = self._fresh("rho_out")
        pOut = self._fresh("p_out")

        new_constraints = child.constraints + [
            piOut == child.pi * piOp,
            rhoOut == child.rho * rhoOp,
            pOut == child.p * pOp,
        ]

        new_vars = child.variables + [piOp, rhoOp, pOp, piOut, rhoOut, pOut]
        new_map = dict(child.op_map)
        new_map[node_idx] = {"pi": piOp, "rho": rhoOp, "p": pOp}

        return _InferResult(
            pi=piOut, rho=rhoOut, p=pOut,
            constraints=new_constraints,
            variables=new_vars,
            op_map=new_map,
        )

    def _apply_t_prod(self, left: _InferResult, right: _InferResult) -> _InferResult:
        piOut = self._fresh("pi_prod")
        rhoOut = self._fresh("rho_prod")
        pOut = self._fresh("p_prod")

        merged_constraints = left.constraints + right.constraints + [
            piOut == left.pi * right.pi,
            rhoOut == left.rho * right.rho,
            pOut == left.p * right.p,
        ]

        merged_vars = left.variables + right.variables + [piOut, rhoOut, pOut]
        merged_map = {**left.op_map, **right.op_map}

        return _InferResult(
            pi=piOut, rho=rhoOut, p=pOut,
            constraints=merged_constraints,
            variables=merged_vars,
            op_map=merged_map,
        )

def _solve(result: _InferResult, target_pi: float, target_rho: float, target_p: float, min_eps: float = 0.01):
    solver = Solver()

    for c in result.constraints:
        solver.add(c)

    for v in result.variables:
        solver.add(v >= min_eps, v <= 1.0)

    solver.add(result.pi == target_pi)
    solver.add(result.rho == target_rho)
    solver.add(result.p == target_p)

    if solver.check() != sat:
        return None

    model = solver.model()

    def _eval(var):
        val = model.eval(var, model_completion=True)
        try:
            return float(val.as_fraction())
        except Exception:
            return float(val.as_decimal(6).rstrip("?"))

    solved = {}
    for node_idx, var_dict in result.op_map.items():
        solved[node_idx] = {
            "pi": _eval(var_dict["pi"]),
            "rho": _eval(var_dict["rho"]),
            "p": _eval(var_dict["p"]),
        }

    return solved

class AccuracyInferenceOptimizer(BaseOptimizer):
    requires_train_data: bool = False

    def __init__(
        self,
        precision_target: float = 0.9,
        recall_target: float = 0.85,
        probability_target: float = 0.95,
        min_epsilon: float = 0.01,
    ):
        self.precision_target = precision_target
        self.recall_target = recall_target
        self.probability_target = probability_target
        self.min_epsilon = min_epsilon

    def optimize(
        self,
        nodes: list,
        train_data=None,
    ) -> list:
        """Walk the LOTUS AST, solve for accuracy, inject CascadeArgs."""

        # Step 1: Traverse and extract constraints
        builder = _ConstraintBuilder()
        result = builder.traverse(nodes)

        if not result.op_map:
            logger.info("AccuracyInferenceOptimizer: no semantic operators found")
            return nodes

        # Step 2: Solve with Z3
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
                self.precision_target, self.recall_target,
                self.probability_target, len(result.op_map),
            )
            return nodes

        # Step 3: Inject solved CascadeArgs into each operator node
        updated_nodes = list(nodes)
        for node_idx, params in solved.items():
            node = updated_nodes[node_idx]

            cascade_update = {
                "precision_target": params["pi"],
                "recall_target": params["rho"],
                "failure_probability": 1.0 - params["p"],
            }

            if hasattr(node, 'cascade_args') and node.cascade_args is not None:
                new_args = node.cascade_args.model_copy(update=cascade_update)
            else:
                new_args = CascadeArgs(**cascade_update)

            updated_nodes[node_idx] = node.model_copy(
                update={"cascade_args": new_args}
            )

            sig = getattr(node, 'signature', lambda: f"node[{node_idx}]")
            logger.info(
                "AccuracyInferenceOptimizer: %s -> pi=%.4f, rho=%.4f, p=%.4f",
                sig(), params["pi"], params["rho"], params["p"],
            )

        return updated_nodes

def _demo():
    print("=" * 65)
    print("  AccuracyInferenceOptimizer — Standalone Demo")
    print("=" * 65)

    # --- Demo 1: Chain of 3 semantic filters ---
    print("\n── Demo 1: Chain of 3 sem_filters ──")
    print("   Pipeline: source → sem_filter('AI') → sem_filter('2023') → sem_filter('cited')")
    print("   Target: π=0.80, ρ=0.70, p=0.90\n")

    if LOTUS_AVAILABLE:
        nodes = [SourceNode(), SemFilterNode(user_instruction="about AI"), SemFilterNode(user_instruction="from 2023"), SemFilterNode(user_instruction="highly cited")]
    else:
        nodes = [SourceNode(), SemFilterNode(), SemFilterNode(), SemFilterNode()]

    builder = _ConstraintBuilder()
    result = builder.traverse(nodes)

    solved = _solve(result, 0.80, 0.70, 0.90)
    if solved:
        check_pi, check_rho, check_p = 1.0, 1.0, 1.0
        for idx, params in sorted(solved.items()):
            print(f"   Node {idx} (sem_filter): π={params['pi']:.4f}, "
                  f"ρ={params['rho']:.4f}, p={params['p']:.4f}")
            print(f"     → CascadeArgs(precision_target={params['pi']:.4f}, "
                  f"recall_target={params['rho']:.4f}, "
                  f"failure_probability={1-params['p']:.4f})")
            check_pi *= params["pi"]
            check_rho *= params["rho"]
            check_p *= params["p"]
        print(f"\n   Verification: π={check_pi:.4f} (target 0.80), "
              f"ρ={check_rho:.4f} (target 0.70), p={check_p:.4f} (target 0.90)")
    else:
        print("   UNSAT — targets infeasible")

    # --- Demo 2: Two filters (simpler case) ---
    print("\n── Demo 2: Chain of 2 sem_filters ──")
    print("   Pipeline: source → sem_filter('lawsuits') → sem_filter('2023')")
    print("   Target: π=0.90, ρ=0.85, p=0.95\n")

    nodes2 = [SourceNode(), SemFilterNode(user_instruction="about lawsuits"), SemFilterNode(user_instruction="from 2023")]
    builder2 = _ConstraintBuilder()
    result2 = builder2.traverse(nodes2)
    solved2 = _solve(result2, 0.90, 0.85, 0.95)
    if solved2:
        check_pi, check_rho = 1.0, 1.0
        for idx, params in sorted(solved2.items()):
            print(f"   Node {idx} (sem_filter): π={params['pi']:.4f}, "
                  f"ρ={params['rho']:.4f}, p={params['p']:.4f}")
            check_pi *= params["pi"]
            check_rho *= params["rho"]
        print(f"\n   Verification: π={check_pi:.4f} (target 0.90), "
              f"ρ={check_rho:.4f} (target 0.85)")

    # --- Demo 3: Single filter ---
    print("\n── Demo 3: Single sem_filter ──")
    print("   Pipeline: source → sem_filter('about lawsuits')")
    print("   Target: π=0.90, ρ=0.85, p=0.95\n")

    nodes3 = [SourceNode(), SemFilterNode(user_instruction="about lawsuits")]
    builder3 = _ConstraintBuilder()
    result3 = builder3.traverse(nodes3)
    solved3 = _solve(result3, 0.90, 0.85, 0.95)
    if solved3:
        for idx, params in sorted(solved3.items()):
            print(f"   Node {idx} (sem_filter): π={params['pi']:.4f}, "
                  f"ρ={params['rho']:.4f}, p={params['p']:.4f}")

    # --- Demo 4: Comparison with even-split ---
    print("\n── Demo 4: SMT vs Even-Split (5 operators) ──")
    print("   Target: π=0.75, ρ=0.60, p=0.85\n")

    nodes4 = [SourceNode()] + [SemFilterNode(user_instruction=f"filter_{i}") for i in range(5)]
    builder4 = _ConstraintBuilder()
    result4 = builder4.traverse(nodes4)
    solved4 = _solve(result4, 0.75, 0.60, 0.85)

    n = 5
    even_pi = 0.75 ** (1.0 / n)
    even_rho = 0.60 ** (1.0 / n)

    print(f"   Even-split: every operator gets π={even_pi:.4f}, ρ={even_rho:.4f}")
    print()
    if solved4:
        check_pi = 1.0
        for idx, params in sorted(solved4.items()):
            print(f"   SMT Node {idx}: π={params['pi']:.4f}, ρ={params['rho']:.4f}")
            check_pi *= params["pi"]

if __name__ == "__main__":
    _demo()
