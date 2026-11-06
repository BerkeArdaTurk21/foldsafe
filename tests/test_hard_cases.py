"""Hard cases: reassignment, control flow blocks and function or class bodies."""

from __future__ import annotations

import textwrap

from foldsafe.analyzer import analyze_source
from foldsafe.findings import Finding

HEADER = """\
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("train.csv")
X = df.drop(columns=["target"])
y = df["target"]
"""
FIRST_BODY_LINE = HEADER.count("\n") + 1


def rule_lines(body: str) -> list[tuple[str, int]]:
    """(rule, body line) pairs; body lines count from 1 after a leading newline."""
    found: list[Finding] = analyze_source(HEADER + textwrap.dedent(body), "script.py")
    return [(finding.rule, finding.line - FIRST_BODY_LINE) for finding in found]


# ------------------------------------------------------------- reassignment


def test_reassigning_x_keeps_the_taint():
    body = """
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_reassigning_after_the_split_is_fine():
    body = """
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """
    assert rule_lines(body) == []


# ------------------------------------------------------------------- blocks


def test_fit_inside_if_branch_is_followed():
    body = """
        use_scaling = True
        if use_scaling:
            X = StandardScaler().fit_transform(X)
        scores = cross_val_score(LogisticRegression(), X, y)
        """
    assert rule_lines(body) == [("FS003", 4)]


def test_fit_inside_loop_and_with_and_try_blocks():
    body = """
        for seed in range(3):
            with open("log.txt", "a") as log:
                try:
                    X_s = StandardScaler().fit_transform(X)
                    X_a, X_b = train_test_split(X_s, random_state=seed)
                except ValueError:
                    pass
        """
    assert rule_lines(body) == [("FS001", 4)]


# ---------------------------------------------------------- function bodies


def test_leak_inside_main_function_is_found():
    body = """
        def main():
            data = pd.read_csv("train.csv")
            features = data.drop(columns=["target"])
            features = StandardScaler().fit_transform(features)
            X_train, X_test = train_test_split(features)

        if __name__ == "__main__":
            main()
        """
    assert rule_lines(body) == [("FS001", 4)]


def test_function_sees_module_variables_defined_before_it():
    body = """
        X_scaled = StandardScaler().fit_transform(X)

        def evaluate():
            return cross_val_score(LogisticRegression(), X_scaled, y)
        """
    assert rule_lines(body) == [("FS003", 4)]


def test_parameters_shadow_module_variables():
    body = """
        X_scaled = StandardScaler().fit_transform(X)

        def evaluate(X_scaled, y):
            return cross_val_score(LogisticRegression(), X_scaled, y)
        """
    assert rule_lines(body) == []


def test_local_assignments_do_not_leak_into_module_scope():
    body = """
        def prepare():
            X = StandardScaler().fit_transform(df)
            return X

        X_train, X_test = train_test_split(X)
        """
    assert rule_lines(body) == []


def test_methods_nested_and_async_functions_are_analyzed():
    body = """
        class Experiment:
            def run(self):
                features = StandardScaler().fit_transform(X)
                return train_test_split(features)

        def outer():
            def inner():
                return cross_val_score(LogisticRegression(), StandardScaler().fit_transform(X), y)

        async def job():
            Z = StandardScaler().fit_transform(X)
            parts = train_test_split(Z)
        """
    assert rule_lines(body) == [("FS001", 3), ("FS003", 8), ("FS001", 11)]


# ------------------------------------------- found by trying real code (day 10)


def test_unlisted_classes_count_as_transformers():
    body = """
        from my_project.features import GroupEncoder
        X_encoded = GroupEncoder().fit_transform(X, y)
        X_a, X_b = train_test_split(X_encoded)

        class Experiment:
            def run(self):
                features = self.scaler.fit_transform(X)
                return cross_val_score(LogisticRegression(), features, y)
        """
    assert rule_lines(body) == [("FS002", 2), ("FS003", 8)]


def test_unlisted_class_fitted_then_transforming_keeps_the_fit_line():
    body = """
        from my_project.features import GroupEncoder
        encoder = GroupEncoder()
        encoder.fit(X)
        X_encoded = encoder.transform(X)
        X_a, X_b = train_test_split(X_encoded)
        """
    assert rule_lines(body) == [("FS001", 3)]


def test_unlisted_model_leaves_no_trace():
    body = """
        from my_project.models import build_model
        model = build_model()
        model.fit(X, y)
        predictions = model.predict(X)
        X_a, X_b = train_test_split(X)
        """
    assert rule_lines(body) == []


def test_fit_on_test_rows_names_an_unlisted_class():
    source = HEADER + textwrap.dedent(
        """
        from my_project.features import GroupEncoder
        X_train, X_test = train_test_split(X)
        encoder = GroupEncoder().fit(X_test)
        """
    )
    (finding,) = analyze_source(source, "script.py")
    assert finding.rule == "FS004"
    assert "GroupEncoder" in finding.message


def test_validation_names_are_not_test_rows():
    body = """
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        calibrator = LogisticRegression().fit(X_val, y_val)
        X_tr, X_dev = train_test_split(X)
        scaler = StandardScaler().fit(X_dev)
        X_a, X_calib_set = train_test_split(X)
        other = StandardScaler().fit(X_calib_set)
        """
    assert rule_lines(body) == []


def test_refit_on_train_plus_validation_is_fine_but_test_is_not():
    body = """
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        ok = LogisticRegression().fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        X_a, X_test = train_test_split(X)
        bad = LogisticRegression().fit(pd.concat([X_a, X_test]), y)
        """
    assert rule_lines(body) == [("FS004", 4)]


def test_names_that_only_contain_val_are_not_validation():
    body = """
        X_a, interval = train_test_split(X)
        scaler = StandardScaler().fit(interval)
        """
    assert rule_lines(body) == [("FS004", 2)]


def test_rows_selected_from_combined_data_are_not_called_mixed():
    body = """
        test = pd.read_csv("test.csv")
        both = pd.concat([X, test])
        fitted_on_mixed = StandardScaler().fit(both)
        fitted_on_slice = StandardScaler().fit(both.iloc[: len(X)])
        fitted_on_columns = StandardScaler().fit(both[["a", "b"]])
        """
    assert rule_lines(body) == [("FS004", 3), ("FS004", 5)]


# ---------------------------------------- found while reviewing coverage (day 11)


def test_dataframe_wrapper_keeps_rows_and_taints():
    body = """
        S = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        X_a, X_b = train_test_split(S)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_dataframe_columns_keyword_does_not_mix_rows():
    body = """
        X_train, X_test = train_test_split(X)
        wrapped = pd.DataFrame(X_test, columns=X_train.columns)
        total = wrapped.sum()
        kept_on_test = wrapped["a"].mean()
        """
    assert rule_lines(body) == []


def test_numpy_and_pandas_functions_keep_rows_and_taints():
    body = """
        import numpy as np
        logged = np.log1p(StandardScaler().fit_transform(X))
        dummies = pd.get_dummies(logged)
        X_a, X_b = train_test_split(dummies)
        """
    assert rule_lines(body) == [("FS001", 2)]


def test_numpy_statistic_of_combined_rows_is_reported_when_kept():
    body = """
        import numpy as np
        test = pd.read_csv("test.csv")
        both = pd.concat([X, test])
        fare_mean = np.mean(both["Fare"])
        print(np.median(both["Fare"]))
        """
    assert rule_lines(body) == [("FS004", 4)]


def test_function_form_scalers_count_as_fits():
    body = """
        from sklearn.preprocessing import robust_scale, scale
        X_a, X_test = train_test_split(scale(X))
        wrong = robust_scale(X_test)
        """
    assert rule_lines(body) == [("FS001", 2), ("FS004", 3)]


def test_test_file_names_are_matched_as_words():
    body = """
        from pathlib import Path
        DATA = Path("data")
        TEST_PATH = "x.csv"
        by_path = pd.read_csv(DATA / "test.csv")
        by_name = pd.read_csv(TEST_PATH)
        by_fstring = pd.read_csv(f"{DATA}/Test_Set.csv")
        by_attribute = pd.read_csv(DATA.test_file)
        not_test = pd.read_csv("latest_prices.csv")
        also_not = pd.read_csv("data/contest.csv")
        for frame in (by_path, by_name, by_fstring, by_attribute):
            pass
        a = StandardScaler().fit(by_path)
        b = StandardScaler().fit(by_name)
        c = StandardScaler().fit(by_fstring)
        d = StandardScaler().fit(by_attribute)
        e = StandardScaler().fit(not_test)
        f = StandardScaler().fit(also_not)
        """
    assert rule_lines(body) == [("FS004", line) for line in (12, 13, 14, 15)]


# ---------------------------------------------------- remaining uncovered paths


def test_annotated_and_augmented_assignments_are_followed():
    body = """
        S: pd.DataFrame = StandardScaler().fit_transform(X)
        Z = df["a"]
        Z += S
        X_a, X_b = train_test_split(Z)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_unary_minus_keeps_the_taint():
    body = """
        S = -StandardScaler().fit_transform(X)
        X_a, X_b = train_test_split(S)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_fit_with_extra_arguments_and_nested_calls():
    body = """
        S = StandardScaler().fit_transform(X)
        LogisticRegression().fit(X, y, sample_weight=cross_val_score(LogisticRegression(), S, y))
        """
    assert rule_lines(body) == [("FS003", 2)]


def test_statsmodels_style_fit_without_data_arguments_is_ignored():
    body = """
        import statsmodels.api as sm
        results = sm.OLS(y, X).fit()
        print(results.summary())
        """
    assert rule_lines(body) == []


def test_transform_of_untracked_data_is_untracked():
    body = """
        scaler = StandardScaler().fit(X)
        out = scaler.transform(unknown_rows)
        X_a, X_b = train_test_split(out)
        """
    assert rule_lines(body) == []


def test_starred_targets_share_the_value():
    body = """
        S = StandardScaler().fit_transform(X)
        first, *rest = S
        X_a, X_b = train_test_split(first)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_values_stored_on_objects_are_not_followed():
    body = """
        class Holder:
            pass

        holder = Holder()
        holder.X = StandardScaler().fit_transform(X)
        X_a, X_b = train_test_split(holder.X)
        """
    assert rule_lines(body) == []


def test_one_finding_for_two_test_fits_on_one_line():
    body = """
        X_a, X_test = train_test_split(X)
        a = StandardScaler().fit(X_test); b = StandardScaler().fit(X_test)
        """
    assert rule_lines(body) == [("FS004", 2)]


# ------------------------------------------------ found by reviewing M2 (day 12)


def test_fit_on_all_rows_after_the_split_is_found_when_test_rows_are_transformed():
    body = """
        X_train, X_test = train_test_split(X)
        scaler = StandardScaler().fit(X)
        a = scaler.transform(X_train)
        b = scaler.transform(X_test)
        c = scaler.transform(X_test)
        """
    assert rule_lines(body) == [("FS001", 2)]


def test_fit_on_all_rows_before_the_split_is_found_when_validation_rows_are_transformed():
    body = """
        scaler = StandardScaler().fit(X)
        X_train, X_val = train_test_split(X)
        b = scaler.transform(X_val)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_full_fit_transforming_only_training_rows_or_a_test_file_is_not_reported():
    body = """
        test = pd.read_csv("test.csv")
        X_train, X_held = train_test_split(X)
        scaler = StandardScaler().fit(X)
        a = scaler.transform(X_train)
        b = scaler.transform(test)
        """
    assert rule_lines(body) == []


def test_fit_on_a_test_file_is_still_reported():
    body = """
        test = pd.read_csv("test.csv")
        scaler = StandardScaler().fit(test)
        """
    assert rule_lines(body) == [("FS004", 2)]


def test_statistics_do_not_mix_rows():
    body = """
        import numpy as np
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        error = np.abs(y_test - y_train.mean()).mean()
        filled = X_test.fillna(X_train.median())
        total = filled.sum()
        scale_all = np.mean(y_train) + np.std(y_test)
        """
    assert rule_lines(body) == []


def test_statistic_taints_still_reach_the_frame():
    body = """
        S = StandardScaler().fit_transform(X)
        filled = df.fillna(S.mean())
        X_a, X_b = train_test_split(filled)
        """
    assert rule_lines(body) == [("FS001", 1)]


def test_models_used_as_transformers_leave_a_taint():
    body = """
        from sklearn.cluster import KMeans
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        distances = KMeans(3).fit_transform(X)
        X_a, X_b = train_test_split(distances)
        projected = LDA().fit(X, y).transform(X)
        X_c, X_d = train_test_split(projected)
        """
    assert rule_lines(body) == [("FS001", 3), ("FS002", 5)]


def test_stateless_transformers_leave_no_taint():
    body = """
        from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
        poly = PolynomialFeatures(2).fit_transform(X)
        labels = LabelEncoder().fit_transform(y)
        X_a, X_b, y_a, y_b = train_test_split(poly, labels)
        """
    assert rule_lines(body) == []


def test_conditional_expressions_dicts_match_and_more_readers():
    body = """
        S = StandardScaler().fit_transform(X) if len(X) else X
        X_a, X_b = train_test_split(S)
        Z = StandardScaler().fit_transform(X)
        frame = pd.DataFrame({"z": Z})
        X_c, X_d = train_test_split(frame)
        match "x":
            case "x":
                M = StandardScaler().fit_transform(X)
                X_e, X_f = train_test_split(M)
        rows = pd.read_sql("select 1", None)
        Q = StandardScaler().fit_transform(rows)
        X_g, X_h = train_test_split(Q)
        """
    assert rule_lines(body) == [("FS001", line) for line in (1, 3, 8, 11)]
