"""Estimator catalog: which classes learn from the rows (and labels) they are fitted on."""

from __future__ import annotations


def _qualify(module: str, names: str) -> list[str]:
    return [f"{module}.{name}" for name in names.split()]


# Kinds:
#   preprocess  learns statistics from X (scalers, imputers, encoders, decompositions)
#   select      learns from y as well (feature selectors, target encoders)
#   resample    creates or removes rows from X and y (imblearn samplers)
#   pipeline    a container; its kind depends on its steps (see FlowAnalyzer.pipeline_kind)
#   model       any other estimator; fitting it is never a leak on its own
ESTIMATOR_KINDS: dict[str, str] = {}

for module, names in {
    "sklearn.preprocessing": (
        "StandardScaler MinMaxScaler MaxAbsScaler RobustScaler QuantileTransformer "
        "PowerTransformer KBinsDiscretizer OneHotEncoder OrdinalEncoder"
    ),
    "sklearn.impute": "SimpleImputer KNNImputer IterativeImputer",
    "sklearn.decomposition": "PCA IncrementalPCA KernelPCA TruncatedSVD FastICA NMF",
    "sklearn.feature_selection": "VarianceThreshold",
    "sklearn.feature_extraction.text": "CountVectorizer TfidfVectorizer TfidfTransformer",
    "sklearn.compose": "ColumnTransformer make_column_transformer",
}.items():
    ESTIMATOR_KINDS.update({qualified: "preprocess" for qualified in _qualify(module, names)})

for module, names in {
    "sklearn.feature_selection": (
        "SelectKBest SelectPercentile SelectFpr SelectFdr SelectFwe GenericUnivariateSelect "
        "RFE RFECV SelectFromModel SequentialFeatureSelector"
    ),
    "sklearn.preprocessing": "TargetEncoder",
}.items():
    ESTIMATOR_KINDS.update({qualified: "select" for qualified in _qualify(module, names)})

for module, names in {
    "imblearn.over_sampling": "SMOTE SMOTENC SMOTEN ADASYN BorderlineSMOTE SVMSMOTE "
    "KMeansSMOTE RandomOverSampler",
    "imblearn.under_sampling": "RandomUnderSampler NearMiss TomekLinks EditedNearestNeighbours "
    "ClusterCentroids",
    "imblearn.combine": "SMOTEENN SMOTETomek",
}.items():
    ESTIMATOR_KINDS.update({qualified: "resample" for qualified in _qualify(module, names)})

for module, names in {
    "sklearn.pipeline": "Pipeline make_pipeline",
    "imblearn.pipeline": "Pipeline make_pipeline",
}.items():
    ESTIMATOR_KINDS.update({qualified: "pipeline" for qualified in _qualify(module, names)})

# Packages whose capitalized, unlisted classes are treated as models.
MODEL_PACKAGES = ("sklearn.", "xgboost.", "lightgbm.", "catboost.", "imblearn.")

# Taint kinds a fit on full data leaves on its output.
LEAKY_KINDS = {"preprocess", "select", "resample"}


def estimator_kind(qualified_name: str) -> str | None:
    """Kind of the class or factory a call constructs; None when it is not an estimator."""
    if qualified_name in ESTIMATOR_KINDS:
        return ESTIMATOR_KINDS[qualified_name]
    last = qualified_name.rsplit(".", 1)[-1]
    if qualified_name.startswith(MODEL_PACKAGES) and last[:1].isupper():
        return "model"
    return None
