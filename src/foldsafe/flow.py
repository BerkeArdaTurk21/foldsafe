"""Statement-ordered data-flow pass over a module.

The pass walks module-level statements top to bottom, including the bodies of
if/for/while/with/try blocks, and keeps an abstract value for each variable.
Along the way it records events (splits, cross-validation, fits) for the rule checkers.
Function and class bodies are analyzed where they are defined (see visit_scope), and a call
to a function defined in the same file is followed into its body with the call's arguments
(see call_function). Methods, lambdas and helpers from other files are not followed.
Code is parsed, never executed.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from foldsafe.catalog import FUNCTIONAL_TRANSFORMERS, LEAKY_KINDS, estimator_kind
from foldsafe.events import (
    CVEvent,
    Event,
    FitEvent,
    HeldOutTransformEvent,
    SplitEvent,
    StatEvent,
)
from foldsafe.values import (
    DataTuple,
    DataValue,
    Enumerated,
    EstimatorValue,
    FoldIndices,
    IndexValue,
    Origin,
    SearchValue,
    SplitterValue,
    Taint,
    TrackedValue,
    Value,
    merge_data,
)

# Values a variable keeps; anything else makes the pass forget the name.
TRACKED = (DataValue, EstimatorValue, SplitterValue, SearchValue, IndexValue)

FIT_METHODS = {"fit", "fit_transform", "fit_resample"}

# Other numpy and pandas functions keep the rows of their arguments: np.log1p, pd.get_dummies.
ARRAY_LIBRARIES = ("numpy.", "pandas.")

# "test" as a word in a path or name: test.csv, X_test.csv, TEST_PATH, but not latest.csv.
TEST_WORD = re.compile(r"(?<![a-z])test(ing)?(?![a-z])")

# A held-out split part with one of these name parts is for tuning or calibration, not final
# evaluation: fitting a calibrator on X_cal or refitting on X_train + X_val is legitimate.
VALIDATION_NAME_PARTS = {"val", "valid", "validation", "cal", "calib", "calibration", "dev"}

# pandas/numpy methods that summarize the rows they are called on.
STAT_METHODS = {
    "mean",
    "median",
    "std",
    "var",
    "mode",
    "quantile",
    "min",
    "max",
    "sum",
    "value_counts",
}
# groupby(...).transform("mean") and .agg("mean") summarize too.
AGGREGATING_METHODS = {"transform", "agg", "aggregate"}
# Statistic methods whose first positional argument is the axis: df.sum(1).
AXIS_FIRST_METHODS = {"mean", "median", "std", "var", "mode", "min", "max", "sum"}
# axis values that compute along each row instead of down each column.
ROW_WISE_AXES = (1, "columns")

# Calls that load a dataset. A file name containing "test" yields TEST_FILE rows.
FILE_READERS = {
    "pandas.read_csv",
    "pandas.read_excel",
    "pandas.read_feather",
    "pandas.read_json",
    "pandas.read_parquet",
    "pandas.read_pickle",
    "pandas.read_table",
    "pandas.read_fwf",
    "pandas.read_hdf",
    "pandas.read_orc",
    "pandas.read_sql",
    "pandas.read_sql_query",
    "pandas.read_sql_table",
    "pandas.read_stata",
    "numpy.genfromtxt",
    "numpy.load",
    "numpy.loadtxt",
    "polars.read_csv",
    "polars.read_parquet",
}
IN_MEMORY_SOURCES = {"pandas.DataFrame", "numpy.array", "numpy.asarray"}

# A call into the project's own code (from titanic.data import load_raw) that is given a
# data file path loads that file: load_raw(DATA_DIR / "train.csv").
DATA_FILE = re.compile(
    r"\.(csv|tsv|parquet|pq|feather|arrow|xlsx|xls|orc|dta|sav)(\.(gz|bz2|xz|zip))?$",
    re.IGNORECASE,
)
# Modules whose functions take a data file path but do not return its rows.
NON_DATA_MODULES = {
    "builtins",
    "glob",
    "io",
    "joblib",
    "json",
    "kaggle",
    "kagglehub",
    "keras",
    "logging",
    "matplotlib",
    "os",
    "pathlib",
    "pickle",
    "plotly",
    "requests",
    "seaborn",
    "shutil",
    "subprocess",
    "sys",
    "tarfile",
    "tensorflow",
    "torch",
    "urllib",
    "zipfile",
}

# Calls into functions defined in the same file are followed this many levels deep.
MAX_CALL_DEPTH = 3

Function = ast.FunctionDef | ast.AsyncFunctionDef
DATASET_PREFIXES = ("sklearn.datasets.load_", "sklearn.datasets.fetch_", "sklearn.datasets.make_")

# Calls whose result mixes the rows of every data argument.
ROW_COMBINERS = {
    "pandas.concat",
    "numpy.concatenate",
    "numpy.hstack",
    "numpy.vstack",
    "numpy.column_stack",
}

MODEL_SELECTION = "sklearn.model_selection"
TRAIN_TEST_SPLIT = f"{MODEL_SELECTION}.train_test_split"

# Splitter class -> whether it shuffles rows when `shuffle` is not given.
SPLITTERS = {
    f"{MODEL_SELECTION}.{name}": shuffles_by_default
    for name, shuffles_by_default in {
        "KFold": False,
        "StratifiedKFold": False,
        "GroupKFold": False,
        "StratifiedGroupKFold": False,
        "TimeSeriesSplit": False,
        "LeaveOneOut": False,
        "LeavePOut": False,
        "LeaveOneGroupOut": False,
        "PredefinedSplit": False,
        "RepeatedKFold": True,
        "RepeatedStratifiedKFold": True,
        "ShuffleSplit": True,
        "StratifiedShuffleSplit": True,
        "GroupShuffleSplit": True,
    }.items()
}

SEARCHES = {
    f"{MODEL_SELECTION}.{name}"
    for name in (
        "GridSearchCV",
        "RandomizedSearchCV",
        "HalvingGridSearchCV",
        "HalvingRandomSearchCV",
    )
}

# Functions that cross-validate an estimator; X is their second argument.
CV_FUNCTIONS = {
    f"{MODEL_SELECTION}.{name}"
    for name in (
        "cross_val_score",
        "cross_validate",
        "cross_val_predict",
        "learning_curve",
        "validation_curve",
        "permutation_test_score",
    )
}


@dataclass
class FlowState:
    """What the pass knows after the last statement."""

    imports: dict[str, str] = field(default_factory=dict)  # local name -> qualified name
    env: dict[str, TrackedValue] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    describe_line: Callable[[int], str] = lambda line: f"line {line}"  # how messages name a line
    functions: dict[str, Function] = field(default_factory=dict)  # defined in this file


class FlowAnalyzer:
    def __init__(self, state: FlowState | None = None, call_stack: tuple[str, ...] = ()) -> None:
        self.state = state or FlowState()
        self.storing = False  # the current statement assigns its result
        self.call_stack = call_stack  # functions being followed, outermost first
        self.returns: list[Value | None] | None = None  # set while following a call

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
                    self.imported(alias.asname, alias.name)
                else:
                    top = alias.name.split(".", 1)[0]
                    self.imported(top, top)
            return
        if node.module is None or node.level:
            return  # relative imports point at the user's own code
        for alias in node.names:
            self.imported(alias.asname or alias.name, f"{node.module}.{alias.name}")

    def imported(self, local: str, qualified: str) -> None:
        self.state.imports[local] = qualified
        self.state.functions.pop(local, None)  # the import replaces a function of that name

    # ------------------------------------------------------------ expressions

    def eval_expr(self, node: ast.expr | None) -> Value | None:
        """Abstract value of an expression; None when it is not tracked."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.state.env.get(node.id)
        if isinstance(node, ast.Call):
            return self.eval_call(node)
        if isinstance(node, ast.Subscript):
            return self.eval_subscript(node)
        if isinstance(node, ast.Attribute):
            # df.values, df.iloc keep the rows of df.
            return self.as_data(self.eval_expr(node.value))
        if isinstance(node, ast.BinOp):
            return merge_data(self.data_operands([node.left, node.right]))
        if isinstance(node, ast.UnaryOp):
            return self.as_data(self.eval_expr(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return merge_data(self.data_operands(node.elts))
        if isinstance(node, ast.IfExp):
            self.eval_expr(node.test)
            return merge_data(self.data_operands([node.body, node.orelse]))
        if isinstance(node, ast.Dict):
            return merge_data(self.data_operands(node.values))
        self.evaluate_children(node)
        return None

    def eval_subscript(self, node: ast.Subscript) -> DataValue | None:
        base = self.as_data(self.eval_expr(node.value))
        index = self.eval_expr(node.slice)
        if base is not None and isinstance(index, IndexValue):
            # X.iloc[train_idx] holds the training rows of X.
            return DataValue(index.side, base.taints)
        if base is not None and base.origin is Origin.COMBINED and self.selects_rows(node):
            # combined.iloc[:n] may be only the train rows: no longer known to be mixed.
            return DataValue(Origin.FULL, base.taints)
        return base  # df["col"], df.iloc[:10] keep the rows of df

    @staticmethod
    def selects_rows(node: ast.Subscript) -> bool:
        by_position = isinstance(node.value, ast.Attribute) and node.value.attr in ("iloc", "loc")
        return by_position or isinstance(node.slice, ast.Slice)

    def eval_call(self, node: ast.Call) -> Value | None:
        name = self.qualified_name(node.func)
        if (
            name is None
            and isinstance(node.func, ast.Name)
            and node.func.id in self.state.functions
        ):
            return self.call_function(node, self.state.functions[node.func.id])
        if name in FILE_READERS:
            path = node.args[0] if node.args else None
            return DataValue(Origin.TEST_FILE if self.mentions_test(path) else Origin.FULL)
        if name in IN_MEMORY_SOURCES:
            # pd.DataFrame(scaled, columns=X.columns) wraps its first argument: same rows, same
            # taints. The keywords add no rows, even when they read from another frame.
            data = self.as_data(self.eval_expr(self.argument(node, 0, "data")))
            self.evaluate_keywords(node)
            return data or DataValue(Origin.FULL)
        if name and name.startswith(DATASET_PREFIXES):
            self.evaluate_arguments(node)
            return DataValue(Origin.FULL)
        if name in ROW_COMBINERS:
            return merge_data(self.data_operands(node.args))
        if name in FUNCTIONAL_TRANSFORMERS:
            return self.functional_transform(node, name)
        if name and name.startswith(ARRAY_LIBRARIES):
            return self.array_function(node, name)
        if name == TRAIN_TEST_SPLIT:
            return self.train_test_split(node)
        if name in CV_FUNCTIONS:
            self.cross_validation(node, name, data_node=self.argument(node, 1, "X"))
            return None
        if name in SPLITTERS:
            self.evaluate_arguments(node)
            return SplitterValue(name, self.keyword_flag(node, "shuffle", SPLITTERS[name]))
        if name in SEARCHES:
            self.evaluate_arguments(node)
            return SearchValue(name, self.cv_shuffles(node))
        kind = estimator_kind(name) if name else None
        if kind is not None:
            return self.construct_estimator(node, name, kind)
        if name is None and isinstance(node.func, ast.Attribute):
            return self.method_call(node, node.func)
        if isinstance(node.func, ast.Name) and node.func.id == "enumerate" and node.args:
            item = self.eval_expr(node.args[0])
            return Enumerated(item) if isinstance(item, FoldIndices) else None
        self.evaluate_arguments(node)  # print(cross_val_score(...)) still records the event
        path = self.data_file_argument(node) if name else None
        if path is not None and name.split(".", 1)[0] not in NON_DATA_MODULES:
            return DataValue(Origin.TEST_FILE if self.mentions_test(path) else Origin.FULL)
        return None

    # ---------------------------------------------------- calls into own code

    def call_function(self, node: ast.Call, function: Function) -> Value | None:
        """Follow a call to a function defined in this file into its body.

        The parameters get the call's argument values, the body sees the variables known
        at the call, its events are recorded, and its return value is the call's value.
        Recursion and calls deeper than MAX_CALL_DEPTH are not followed.
        """
        starred = any(isinstance(argument, ast.Starred) for argument in node.args)
        positional = [self.eval_expr(argument) for argument in node.args]
        named = {kw.arg: self.eval_expr(kw.value) for kw in node.keywords if kw.arg}
        if function.name in self.call_stack or len(self.call_stack) >= MAX_CALL_DEPTH:
            return None
        bound = self.bind_parameters(function.args, [] if starred else positional, named)
        parameters = self.parameter_names(function.args)
        env = {name: value for name, value in self.state.env.items() if name not in parameters}
        env.update({name: value for name, value in bound.items() if isinstance(value, TRACKED)})
        local = FlowState(
            dict(self.state.imports),
            env,
            self.state.events,
            self.state.describe_line,
            dict(self.state.functions),
        )
        callee = FlowAnalyzer(local, (*self.call_stack, function.name))
        callee.returns = []
        callee.visit_block(function.body)
        return self.combine_returns(callee.returns)

    def bind_parameters(
        self,
        arguments: ast.arguments,
        positional: list[Value | None],
        named: dict[str, Value | None],
    ) -> dict[str, Value | None]:
        """Parameter name -> value: from the call, else from the default, else unknown."""
        values: dict[str, Value | None] = {}
        ordered = [*arguments.posonlyargs, *arguments.args]
        first_default = len(ordered) - len(arguments.defaults)
        for index, parameter in enumerate(ordered):
            if index < len(positional):
                values[parameter.arg] = positional[index]
            elif parameter.arg in named:
                values[parameter.arg] = named[parameter.arg]
            elif index >= first_default:
                values[parameter.arg] = self.eval_expr(arguments.defaults[index - first_default])
        for parameter, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
            if parameter.arg in named:
                values[parameter.arg] = named[parameter.arg]
            elif default is not None:
                values[parameter.arg] = self.eval_expr(default)
        return values

    @staticmethod
    def combine_returns(returns: list[Value | None]) -> Value | None:
        """One value for all return statements; rows of different origins are not guessed."""
        values = [value for value in returns if value is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        if all(isinstance(value, DataValue) for value in values):
            return merge_same_origin(values)
        lengths = {len(value.parts) for value in values if isinstance(value, DataTuple)}
        if all(isinstance(value, DataTuple) for value in values) and len(lengths) == 1:
            columns = zip(*(value.parts for value in values), strict=True)
            return DataTuple(tuple(merge_same_origin(column) for column in columns))
        return None

    def eval_return(self, node: ast.expr | None) -> Value | None:
        """`return train, test` keeps the two parts apart, like train_test_split's result."""
        if isinstance(node, ast.Tuple) and not any(isinstance(e, ast.Starred) for e in node.elts):
            return DataTuple(tuple(self.as_data(self.eval_expr(element)) for element in node.elts))
        return self.eval_expr(node)

    def data_file_argument(self, node: ast.Call) -> ast.expr | None:
        """The argument that names a data file (train.csv, sales.xlsx), if any."""
        for argument in self.argument_nodes(node):
            for part in ast.walk(argument):
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and DATA_FILE.search(part.value)
                ):
                    return argument
        return None

    def array_function(self, node: ast.Call, name: str) -> DataValue | None:
        """np.log1p(X), pd.get_dummies(X), np.mean(X): the result comes from the same rows."""
        self.evaluate_keywords(node)
        data = merge_data(self.data_operands(node.args))
        statistic = name.rsplit(".", 1)[-1]
        if data is not None and statistic in STAT_METHODS and not self.row_wise(node, 1):
            self.record(StatEvent(node.lineno, statistic, data, self.storing))
            return self.statistic_of(data, replace(data, summary=True), node.lineno)
        return data

    def functional_transform(self, node: ast.Call, name: str) -> DataValue | None:
        """sklearn.preprocessing.scale(X) is a fit_transform in one call."""
        x = self.as_data(self.eval_expr(self.argument(node, 0, "X")))
        self.evaluate_keywords(node)
        self.record(FitEvent(node.lineno, name, "preprocess", "fit_transform", x, False))
        estimator = self.fitted(EstimatorValue(name, "preprocess"), x, node.lineno, False)
        return self.transformed(x, estimator)

    def method_call(self, node: ast.Call, func: ast.Attribute) -> Value | None:
        receiver = self.eval_expr(func.value)
        if isinstance(receiver, SplitterValue) and func.attr == "split":
            inputs = self.data_operands(node.args)
            self.record(SplitEvent(node.lineno, "cv_split", tuple(inputs), receiver.shuffled))
            return FoldIndices()
        if isinstance(receiver, SearchValue) and func.attr == "fit":
            self.cross_validation(
                node, f"{receiver.cls}.fit", self.argument(node, 0, "X"), receiver.shuffled
            )
            return None
        if isinstance(receiver, EstimatorValue) or (receiver is None and func.attr in FIT_METHODS):
            return self.estimator_method(node, func, receiver)
        if isinstance(receiver, DataValue):
            statistic = self.statistic_name(node, func.attr)
            # A method on data (drop, fillna, to_numpy, ...) keeps its rows.
            result = merge_data([receiver, *self.data_operands(self.argument_nodes(node))])
            if statistic is None:
                return result
            self.record(StatEvent(node.lineno, statistic, receiver, self.storing))
            summary = replace(result, origin=receiver.origin, summary=True)
            return self.statistic_of(receiver, summary, node.lineno)
        self.evaluate_arguments(node)
        return None

    @staticmethod
    def statistic_of(rows: DataValue, result: DataValue, line: int) -> DataValue:
        """A statistic of all rows carries a "stat" mark: wherever it is kept, so are they."""
        if rows.origin is not Origin.FULL:
            return result
        return replace(result, taints=result.taints | {Taint(line, "stat")})

    @staticmethod
    def row_wise(node: ast.Call, axis_position: int | None) -> bool:
        """sum(axis=1), mean(axis="columns"), np.max(X, 1): each row uses its own values only."""
        axis = FlowAnalyzer.keyword(node, "axis")
        if axis is None and axis_position is not None and len(node.args) > axis_position:
            axis = node.args[axis_position]
        return isinstance(axis, ast.Constant) and axis.value in ROW_WISE_AXES

    @classmethod
    def statistic_name(cls, node: ast.Call, method: str) -> str | None:
        if method in STAT_METHODS:
            # df.mean(1) passes the axis first; quantile(0.5) and value_counts() do not.
            axis_position = 0 if method in AXIS_FIRST_METHODS else None
            return None if cls.row_wise(node, axis_position) else method
        first = node.args[0] if node.args else None
        if (
            method in AGGREGATING_METHODS
            and isinstance(first, ast.Constant)
            and first.value in STAT_METHODS
        ):
            return f"{method}:{first.value}"
        return None

    # ------------------------------------------------------------- estimators

    def construct_estimator(self, node: ast.Call, name: str, kind: str) -> EstimatorValue:
        if kind == "pipeline":
            kind = self.pipeline_kind(self.pipeline_steps(node, name))
        else:
            self.evaluate_arguments(node)
        return EstimatorValue(name, kind)

    def pipeline_steps(self, node: ast.Call, name: str) -> list[Value | None]:
        if name.endswith("make_pipeline"):
            step_nodes = list(node.args)
        else:
            steps = self.argument(node, 0, "steps")
            elements = steps.elts if isinstance(steps, (ast.List, ast.Tuple)) else []
            # Pipeline([("scale", StandardScaler()), ...]): the estimator is the last item.
            step_nodes = [el.elts[-1] for el in elements if isinstance(el, ast.Tuple) and el.elts]
        return [self.eval_expr(step) for step in step_nodes]

    @staticmethod
    def pipeline_kind(steps: list[Value | None]) -> str:
        """A pipeline ending in a transformer behaves like one; otherwise it is a safe container."""
        last = steps[-1] if steps else None
        if not isinstance(last, EstimatorValue) or last.kind not in ("preprocess", "select"):
            return "pipeline"
        kinds = {step.kind for step in steps if isinstance(step, EstimatorValue)}
        return "select" if "select" in kinds else "preprocess"

    def estimator_method(
        self, node: ast.Call, func: ast.Attribute, estimator: EstimatorValue | None
    ) -> Value | None:
        method = func.attr
        x_node = self.argument(node, 0, "X")
        y_node = self.argument(node, 1, "y")
        x = self.as_data(self.eval_expr(x_node))
        y = self.as_data(self.eval_expr(y_node))
        for other in self.argument_nodes(node):
            if other is not x_node and other is not y_node:
                self.eval_expr(other)
        if estimator is None:
            if x is None:
                return None  # .fit() on something unknown, with nothing we track
            estimator = self.unlisted_estimator(func, method, y_node is not None)
        if method in FIT_METHODS:
            uses_y = y_node is not None
            event = FitEvent(node.lineno, estimator.cls, estimator.kind, method, x, uses_y)
            self.record(event)
            estimator = self.fitted(estimator, x, node.lineno, uses_y)
            if isinstance(func.value, ast.Name):
                self.state.env[func.value.id] = estimator  # fit() changes the object in place
        if method == "fit":
            return estimator  # sklearn's fit returns self
        if method in ("fit_transform", "transform"):
            output = self.transformed(x, estimator)
            if self.is_held_out(x) and estimator.fit_taint is not None:
                self.record(HeldOutTransformEvent(node.lineno, output))
            return output
        if method == "fit_resample":
            return DataTuple((self.transformed(x, estimator), self.transformed(y, estimator)))
        return None  # predict, score, ... produce no tracked rows

    @classmethod
    def unlisted_estimator(cls, func: ast.Attribute, method: str, uses_y: bool) -> EstimatorValue:
        """A custom or third-party class that is not in the catalog but is fitted on data.

        It is treated as a transformer: a plain model leaves no trace, because the taint
        only reaches data through transform() or fit_transform().
        """
        kind = "resample" if method == "fit_resample" else "select" if uses_y else "preprocess"
        return EstimatorValue(cls.receiver_label(func.value), kind)

    @classmethod
    def receiver_label(cls, node: ast.expr) -> str:
        """Readable name for messages: MyEncoder().fit -> MyEncoder, self.scaler.fit -> scaler."""
        if isinstance(node, ast.Call):
            return cls.receiver_label(node.func)
        if isinstance(node, ast.Attribute):
            return node.attr
        return node.id if isinstance(node, ast.Name) else "estimator"

    @staticmethod
    def is_held_out(data: DataValue | None) -> bool:
        return data is not None and data.origin in (Origin.TEST, Origin.VALIDATION)

    @staticmethod
    def fitted(
        estimator: EstimatorValue, data: DataValue | None, line: int, uses_y: bool
    ) -> EstimatorValue:
        origin = data.origin if data is not None else None
        # A model's fit taint only ever reaches data through its transform output.
        model_kind = "select" if uses_y else "preprocess"
        taint_kind = model_kind if estimator.kind == "model" else estimator.kind
        taint = None
        if origin is Origin.FULL and taint_kind in LEAKY_KINDS:
            taint = Taint(line, taint_kind)
        return EstimatorValue(estimator.cls, estimator.kind, origin, taint)

    @staticmethod
    def transformed(data: DataValue | None, estimator: EstimatorValue) -> DataValue | None:
        """Output of a transformer: the input rows, plus the fit's taint if it saw all rows."""
        if data is None:
            return None
        if estimator.fit_taint is None:
            return data
        return DataValue(data.origin, data.taints | {estimator.fit_taint})

    def train_test_split(self, node: ast.Call) -> DataTuple:
        inputs = [self.as_data(self.eval_expr(arg)) for arg in node.args]
        self.evaluate_keywords(node)
        tracked = tuple(value for value in inputs if value is not None)
        shuffled = self.keyword_flag(node, "shuffle", True)
        self.record(SplitEvent(node.lineno, "train_test_split", tracked, shuffled))
        parts: list[DataValue | None] = []
        for value in inputs:
            if value is None:
                parts.extend([None, None])
            else:
                parts.extend(
                    [DataValue(Origin.TRAIN, value.taints), DataValue(Origin.TEST, value.taints)]
                )
        return DataTuple(tuple(parts))

    def cross_validation(
        self,
        node: ast.Call,
        consumer: str,
        data_node: ast.expr | None,
        shuffled: bool | None = None,
    ) -> None:
        data = self.as_data(self.eval_expr(data_node))
        for other in self.argument_nodes(node):
            if other is not data_node:
                self.eval_expr(other)  # nested searches, estimators, y
        if shuffled is None:
            shuffled = self.cv_shuffles(node)
        self.record(CVEvent(node.lineno, consumer, data, shuffled))

    def cv_shuffles(self, node: ast.Call) -> bool:
        cv = self.eval_expr(self.keyword(node, "cv"))
        return isinstance(cv, SplitterValue) and cv.shuffled

    # ---------------------------------------------------------------- helpers

    def record(self, event: Event) -> None:
        self.state.events.append(event)

    @staticmethod
    def as_data(value: Value | None) -> DataValue | None:
        return value if isinstance(value, DataValue) else None

    def data_operands(self, nodes: list[ast.expr]) -> list[DataValue]:
        values = (self.as_data(self.eval_expr(node)) for node in nodes)
        return [value for value in values if value is not None]

    @staticmethod
    def argument_nodes(node: ast.Call) -> list[ast.expr]:
        return [*node.args, *(keyword.value for keyword in node.keywords)]

    def evaluate_arguments(self, node: ast.Call) -> None:
        for argument in self.argument_nodes(node):
            self.eval_expr(argument)

    def evaluate_keywords(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            self.eval_expr(keyword.value)

    def evaluate_children(self, node: ast.AST) -> None:
        """Visit sub-expressions of an untracked expression so nested calls still count."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.eval_expr(child)

    @staticmethod
    def argument(node: ast.Call, position: int, name: str) -> ast.expr | None:
        if len(node.args) > position:
            return node.args[position]
        return FlowAnalyzer.keyword(node, name)

    @staticmethod
    def keyword(node: ast.Call, name: str) -> ast.expr | None:
        return next((kw.value for kw in node.keywords if kw.arg == name), None)

    @staticmethod
    def keyword_flag(node: ast.Call, name: str, default: bool) -> bool:
        value = FlowAnalyzer.keyword(node, name)
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            return value.value
        return default

    @staticmethod
    def mentions_test(path: ast.expr | None) -> bool:
        """True for test.csv, DATA_DIR / "test.csv" and TEST_PATH; False for latest_prices.csv."""
        if path is None:
            return False
        for part in ast.walk(path):
            text = None
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text = part.value
            elif isinstance(part, ast.Name):
                text = part.id
            elif isinstance(part, ast.Attribute):
                text = part.attr
            if text is not None and TEST_WORD.search(text.lower()):
                return True
        return False

    # ------------------------------------------------------------- statements

    def run(self, tree: ast.Module) -> FlowState:
        self.visit_block(tree.body)
        return self.state

    def visit_block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit_statement(statement)

    def visit_statement(self, node: ast.stmt) -> None:
        self.storing = isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
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
        elif isinstance(node, ast.Return):
            value = self.eval_return(node.value)  # return cross_val_score(...) records the event
            if self.returns is not None:
                self.returns.append(value)
        elif isinstance(node, ast.Expr):
            self.eval_expr(node.value)
        elif isinstance(node, (ast.If, ast.While)):
            self.eval_expr(node.test)
            self.visit_block(node.body)
            self.visit_block(node.orelse)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self.bind(node.target, self.eval_expr(node.iter))
            self.visit_block(node.body)
            self.visit_block(node.orelse)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            self.visit_block(node.body)
        elif isinstance(node, ast.Try):
            self.visit_block(node.body)
            for handler in node.handlers:
                self.visit_block(handler.body)
            self.visit_block(node.orelse)
            self.visit_block(node.finalbody)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                self.visit_block(case.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.state.functions[node.name] = node
            self.visit_scope(node.body, self.parameter_names(node.args))
        elif isinstance(node, ast.ClassDef):
            self.state.env.pop(node.name, None)
            self.state.functions.pop(node.name, None)
            self.visit_scope(node.body, set())

    def visit_scope(self, body: list[ast.stmt], parameters: set[str]) -> None:
        """Analyze a function or class body on its own, where it is defined.

        It sees the module's imports and the variables known so far, minus names its
        parameters shadow (their values are unknown). Its assignments stay local; its
        events are kept. A function that is called is followed again at each call, with
        the call's arguments (call_function); this pass covers the functions never called.
        """
        env = {name: value for name, value in self.state.env.items() if name not in parameters}
        local = FlowState(
            dict(self.state.imports), env, self.state.events, functions=dict(self.state.functions)
        )
        FlowAnalyzer(local, self.call_stack).visit_block(body)

    @staticmethod
    def parameter_names(arguments: ast.arguments) -> set[str]:
        names = {
            arg.arg for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        }
        names.update(arg.arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None)
        return names

    def bind(self, target: ast.expr, value: Value | None) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            self.bind_sequence(target.elts, value)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, value)
        elif isinstance(target, ast.Name):
            self.state.functions.pop(target.id, None)  # the name no longer means the function
            if isinstance(value, TRACKED):
                self.state.env[target.id] = value
            else:
                self.state.env.pop(target.id, None)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            # `df["new"] = ...` adds a column: df keeps its rows and gains the value's taints.
            current = self.state.env.get(target.value.id)
            if isinstance(current, DataValue) and isinstance(value, DataValue):
                self.state.env[target.value.id] = DataValue(
                    current.origin, current.taints | value.taints
                )

    @staticmethod
    def named_validation(target: ast.expr, part: DataValue | None) -> DataValue | None:
        """`X_train, X_cal = train_test_split(X)`: the held-out part is a validation set by name."""
        if part is None or part.origin is not Origin.TEST or not isinstance(target, ast.Name):
            return part
        if VALIDATION_NAME_PARTS.intersection(target.id.lower().split("_")):
            return DataValue(Origin.VALIDATION, part.taints)
        return part

    def bind_sequence(self, targets: list[ast.expr], value: Value | None) -> None:
        if isinstance(value, DataTuple) and len(targets) == len(value.parts):
            for target, part in zip(targets, value.parts, strict=True):
                self.bind(target, self.named_validation(target, part))
        elif isinstance(value, FoldIndices) and len(targets) == 2:
            self.bind(targets[0], IndexValue(Origin.TRAIN))
            self.bind(targets[1], IndexValue(Origin.TEST))
        elif isinstance(value, Enumerated) and len(targets) == 2:
            self.bind(targets[0], None)
            self.bind(targets[1], value.item)
        else:
            # `X, y = load_iris(return_X_y=True)`: every part comes from the same rows.
            part = value if isinstance(value, DataValue) else None
            for target in targets:
                self.bind(target, part)


def merge_same_origin(values: Iterable[DataValue | None]) -> DataValue | None:
    """Merge the taints of values from the same rows; None when the origins differ."""
    present = list(values)
    if not present or any(value is None for value in present):
        return None
    if len({value.origin for value in present}) != 1:
        return None
    return merge_data(present)


def run_flow(tree: ast.Module, state: FlowState | None = None) -> FlowState:
    return FlowAnalyzer(state).run(tree)
