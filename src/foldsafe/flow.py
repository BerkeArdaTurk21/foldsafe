"""Statement-ordered data-flow pass over a module.

The pass walks module-level statements top to bottom, including the bodies of
if/for/while/with/try blocks, and keeps an abstract value for each variable.
Function and class bodies are not analyzed. Code is parsed, never executed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from foldsafe.values import DataValue, Origin, merge_data

# Calls that load a dataset. A file name containing "test" yields TEST rows.
FILE_READERS = {
    "pandas.read_csv",
    "pandas.read_excel",
    "pandas.read_feather",
    "pandas.read_json",
    "pandas.read_parquet",
    "pandas.read_pickle",
    "pandas.read_table",
    "numpy.genfromtxt",
    "numpy.load",
    "numpy.loadtxt",
    "polars.read_csv",
    "polars.read_parquet",
}
IN_MEMORY_SOURCES = {"pandas.DataFrame", "numpy.array", "numpy.asarray"}
DATASET_PREFIXES = ("sklearn.datasets.load_", "sklearn.datasets.fetch_", "sklearn.datasets.make_")

# Calls whose result mixes the rows of every data argument.
ROW_COMBINERS = {
    "pandas.concat",
    "numpy.concatenate",
    "numpy.hstack",
    "numpy.vstack",
    "numpy.column_stack",
}


@dataclass
class FlowState:
    """What the pass knows after the last statement."""

    imports: dict[str, str] = field(default_factory=dict)  # local name -> qualified name
    env: dict[str, DataValue] = field(default_factory=dict)


class FlowAnalyzer:
    def __init__(self) -> None:
        self.state = FlowState()

    # ------------------------------------------------------------------ names

    def qualified_name(self, node: ast.expr) -> str | None:
        """Resolve `pd.read_csv` to `pandas.read_csv` using the module's imports."""
        if isinstance(node, ast.Name):
            return self.state.imports.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualified_name(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def record_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self.state.imports[alias.asname] = alias.name
                else:
                    top = alias.name.split(".", 1)[0]
                    self.state.imports[top] = top
            return
        if node.module is None or node.level:
            return  # relative imports point at the user's own code
        for alias in node.names:
            self.state.imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    # ------------------------------------------------------------ expressions

    def eval_expr(self, node: ast.expr | None) -> DataValue | None:
        """Abstract value of an expression; None when it is not tracked data."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.state.env.get(node.id)
        if isinstance(node, ast.Call):
            return self.eval_call(node)
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            # df.values, df["col"], df.iloc[:10] keep the rows of df.
            return self.eval_expr(node.value)
        if isinstance(node, ast.BinOp):
            return merge_data(self.data_operands([node.left, node.right]))
        if isinstance(node, ast.UnaryOp):
            return self.eval_expr(node.operand)
        if isinstance(node, (ast.List, ast.Tuple)):
            return merge_data(self.data_operands(node.elts))
        return None

    def eval_call(self, node: ast.Call) -> DataValue | None:
        name = self.qualified_name(node.func)
        if name in FILE_READERS:
            return DataValue(Origin.TEST if self.names_test_file(node) else Origin.FULL)
        if name in IN_MEMORY_SOURCES or (name and name.startswith(DATASET_PREFIXES)):
            return DataValue(Origin.FULL)
        if name in ROW_COMBINERS:
            return merge_data(self.data_operands(node.args))
        if isinstance(node.func, ast.Attribute):
            receiver = self.eval_expr(node.func.value)
            if receiver is not None:
                # A method on data (drop, fillna, median, to_numpy, ...) keeps its rows.
                return merge_data([receiver, *self.data_operands(node.args)])
        return None

    def data_operands(self, nodes: list[ast.expr]) -> list[DataValue]:
        values = (self.eval_expr(node) for node in nodes)
        return [value for value in values if value is not None]

    @staticmethod
    def names_test_file(node: ast.Call) -> bool:
        first = node.args[0] if node.args else None
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and "test" in first.value.lower()
        )

    # ------------------------------------------------------------- statements

    def run(self, tree: ast.Module) -> FlowState:
        self.visit_block(tree.body)
        return self.state

    def visit_block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit_statement(statement)

    def visit_statement(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.record_import(node)
        elif isinstance(node, ast.Assign):
            value = self.eval_expr(node.value)
            for target in node.targets:
                self.bind(target, value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            self.bind(node.target, self.eval_expr(node.value))
        elif isinstance(node, ast.AugAssign):
            self.bind(node.target, merge_data(self.data_operands([node.target, node.value])))
        elif isinstance(node, ast.Expr):
            self.eval_expr(node.value)
        elif isinstance(node, (ast.If, ast.While)):
            self.visit_block(node.body)
            self.visit_block(node.orelse)
        elif isinstance(node, ast.For):
            self.bind(node.target, self.eval_expr(node.iter))
            self.visit_block(node.body)
            self.visit_block(node.orelse)
        elif isinstance(node, ast.With):
            self.visit_block(node.body)
        elif isinstance(node, ast.Try):
            self.visit_block(node.body)
            for handler in node.handlers:
                self.visit_block(handler.body)
            self.visit_block(node.orelse)
            self.visit_block(node.finalbody)
        # FunctionDef, ClassDef and the rest are out of scope for v0.1.

    def bind(self, target: ast.expr, value: DataValue | None) -> None:
        if isinstance(target, ast.Name):
            if value is None:
                self.state.env.pop(target.id, None)
            else:
                self.state.env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            # `X, y = load_iris(return_X_y=True)`: every part comes from the same rows.
            for element in target.elts:
                self.bind(element, value)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, value)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            # `df["new"] = ...` adds a column: df keeps its rows and gains the value's taints.
            current = self.state.env.get(target.value.id)
            if current is not None and value is not None:
                self.state.env[target.value.id] = DataValue(
                    current.origin, current.taints | value.taints
                )


def run_flow(tree: ast.Module) -> FlowState:
    return FlowAnalyzer().run(tree)
