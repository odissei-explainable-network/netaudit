"""Train a prediction model to predict target from embeddings and/or covariates.
"""

import argparse
import logging
import os
import time

from functools import partial

import hydra
import json
import numpy as np
import optuna
import polars as pl
import polars.selectors as cs

from gensim.models import KeyedVectors

from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, r2_score, roc_auc_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from xgboost import XGBClassifier, XGBRegressor

from predict_income_helper import CustomJSONEncoder

logger = logging.getLogger(__name__)


def convert_keyed_vectors_to_polars(keyed_vectors, keys, skip_missing=True):
    requested_keys = set(keys)
    available_keys = set(keyed_vectors.index_to_key)

    missing_keys = requested_keys - available_keys

    if missing_keys and not skip_missing:
        raise KeyError(f"Keys not found in keyed vectors: {missing_keys}")
    else:
        logger.info("Missing keys: %s", missing_keys)

    valid_keys = requested_keys & available_keys

    vectors = keyed_vectors[valid_keys]

    vector_dict = {
        f"dim_{i}": vectors[:, i] for i in range(vectors.shape[1])
    }

    vector_dict["RINPERSOON"] = list(valid_keys)

    df = pl.DataFrame(vector_dict)

    return df


def get_scoring_funs(is_classification, **kwargs):
    if is_classification:
        return {
            "F1_micro": make_scorer(f1_score, average="micro", **kwargs),
            "F1_macro": make_scorer(f1_score, average="macro", **kwargs),
            "F1_weighted": make_scorer(f1_score, average="weighted", **kwargs),
            "Precision_macro": make_scorer(precision_score, average="macro", **kwargs),
            "Recall_macro": make_scorer(recall_score, average="macro", **kwargs),
            "Accuracy": make_scorer(accuracy_score),
            "AUC_macro": make_scorer(roc_auc_score, average="macro", multi_class="ovr", response_method="predict_proba"),
            "AUC_micro": make_scorer(roc_auc_score, average="micro", multi_class="ovr", response_method="predict_proba"),
            "AUC_weighted": make_scorer(roc_auc_score, average="weighted", multi_class="ovr", response_method="predict_proba"),
            "Average_precision_macro": make_scorer(average_precision_score, average="macro", response_method="predict_proba"),
            "Average_precision_micro": make_scorer(average_precision_score, average="micro", response_method="predict_proba"),
            "Average_precision_weighted": make_scorer(average_precision_score, average="weighted", response_method="predict_proba"),
            
        }
    else:
        return {
            "R2": make_scorer(r2_score),
            "MSE": make_scorer(mean_squared_error),
            "MAE": make_scorer(mean_absolute_error),
        }
    

def get_estimator_class(estimator, is_classification):
    if is_classification:
        return {
            "lm": LogisticRegression,
            "knn": KNeighborsClassifier,
            "xgb": XGBClassifier,
            "dummy": DummyClassifier
        }[estimator]
    else:
        return {
            "lm": Ridge,
            "knn": KNeighborsRegressor,
            "xgb": XGBRegressor,
            "dummy": DummyRegressor
        }[estimator]


def get_estimator_params(trial, estimator, is_classification):
    params = {}

    if estimator == "lm":
        if is_classification:
            params["multi_class"] = "multinomial"
            params["penalty"] = "l2"
            params["C"] = trial.suggest_float("C", low=0.001, high=100, log=True)
            # params["solver"] = "saga"
        else:
            params["max_iter"] = 5000
            params["alpha"] = trial.suggest_float("alpha", low=0.001, high=100, log=True)

    if estimator == "knn":
        params["n_neighbors"] = trial.suggest_int("n_neighbors", low=2, high=100)

    if estimator == "xgb":
        params["tree_method"] = "hist"
        params["eta"] = trial.suggest_float("eta", low=0, high=1)
        params["max_depth"] = trial.suggest_int("max_depth", low=2, high=50)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", low=1, high=100)
        params["subsample"] = trial.suggest_float("subsample", low=0.2, high=1)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", low=0.2, high=1)
        params["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", low=0.2, high=1)
        params["lambda"] = trial.suggest_float("lambda", low=0.001, high=100, log=True)

    return params  


def create_estimator_pipeline(columns, estimator, is_classification, **kwargs):
    estimator_class = get_estimator_class(estimator, is_classification)

    estimator_obj = estimator_class(**kwargs)

    numeric_columns = [i for i, col in enumerate(columns) if not col.endswith("_cat")]
    categorical_columns = [i for i, col in enumerate(columns) if col.endswith("_cat")]

    numeric_preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_preprocessor = make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore"))

    preprocessor = make_column_transformer(
        (numeric_preprocessor, numeric_columns),
        (categorical_preprocessor, categorical_columns)
    )

    estimator_pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator_obj)])

    return estimator_pipeline


def create_objective(trial, X, y, inner_cv, columns, estimator, is_classification, scoring):
    params = get_estimator_params(trial, estimator, is_classification)

    estimator_pipeline = create_estimator_pipeline(columns, estimator, is_classification, **params)

    score = cross_val_score(estimator_pipeline, X=X, y=y, cv=inner_cv, scoring=scoring, n_jobs=-1)

    return score.mean()


def run_optimization_study(X, y, inner_cv, columns, estimator, is_classification, scoring, num_trials=100, seed=2024):
    direction = "maximize"
    
    sampler = optuna.samplers.TPESampler(seed=seed)

    logger.info("Creating optimization study for estimator %s with direction %s", estimator, direction)
    study = optuna.create_study(direction=direction, sampler=sampler)

    optim_fun = partial(create_objective, X=X, y=y, inner_cv=inner_cv, columns=columns, estimator=estimator, is_classification=is_classification, scoring=scoring)

    study.optimize(optim_fun, n_trials=num_trials)

    return study


def preprocess_training_data(args):
    logger.info("Loading liss data from %s", args.liss_filename)

    df_liss = pl.read_csv(args.liss_filename, separator=",", schema_overrides={"RINPERSOON": pl.Int64})

    if args.target != "cv24p013_trust_government":
        logger.info("Excluding %s non-elligble voters", df_liss.filter(pl.col("cv24p053_voted").eq(2)).shape[0])
        df_liss = df_liss.filter(pl.col("cv24p053_voted").is_in([1, 2])).drop("cv24p053_voted")

    logger.info("Loading embeddings from %s", args.embeddings_filename)
    
    wv_embeddings = KeyedVectors.load(args.embeddings_filename)

    logger.info("Converting embeddings to polars dataframe")
    df_embeddings = convert_keyed_vectors_to_polars(wv_embeddings, df_liss["RINPERSOON"])

    df_features = df_liss.join(df_embeddings, on="RINPERSOON", how="left", validate="1:1")

    if args.target == "cv24p013_trust_government":
        logger.info("Removing %s cases with missing target", df_features.filter(pl.col("cv24p013_trust_government").is_null()).shape[0])
        df_features = df_features.filter(pl.col("cv24p013_trust_government").is_not_null())

    # df_features.write_csv("debug.csv")
    # df_features = pl.read_csv("debug.csv")

    y = df_features[args.target].to_numpy()

    df_features = df_features.drop([args.target, "RINPERSOON"])

    if args.target == "cv24p307_voting_behavior":
        logger.info("Dropping feature cv24p307_voting_behavior_populist")
        df_features = df_features.drop("cv24p307_voting_behavior_populist")
    elif args.target == "cv24p307_voting_behavior_populist":
        logger.info("Dropping feature cv24p307_voting_behavior")
        df_features = df_features.drop("cv24p307_voting_behavior")
    else:
        logger.info("Dropping features cv24p307_voting_behavior and cv24p307_voting_behavior_populist")
        df_features = df_features.drop(["cv24p307_voting_behavior", "cv24p307_voting_behavior_populist"])

    dim_col_selector = cs.starts_with("dim_")
    cbs_col_selector = cs.starts_with("cbs_")

    if args.model == "embeddings":
        logger.info("Selecting embeddings as only features")
        df_features = df_features.select(dim_col_selector)
    elif args.model == "covariates":
        logger.info("Selecting covariates as only features")
        df_features = df_features.drop(dim_col_selector)
    elif args.model == "embeddings_cbs":
        logger.info("Selecting embeddings and cbs covariates as only features")
        df_features = df_features.select(dim_col_selector | cbs_col_selector)
    else:
        logger.info("Selecting all features")

    logger.info("Training estimator with target %s and features %s", args.target, df_features.columns)

    X = df_features.to_numpy()

    return df_features, X, y


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a prediction model to predict target from embeddings and/or covariates.",
        epilog="""python train_prediction_model.py \
            --embeddings-filename /data/models/flat_2021/embeddings.kv \
            --estimator xgb
"""
    )
    parser.add_argument("--embeddings-filename", required=True, help="Filename of node embeddings in gensim embeddings vector format with '.kv' ending.")
    parser.add_argument("--estimator", default="lm", choices=["dummy", "lm", "knn", "xgb"], help="Name of prediction algorithm.")
    parser.add_argument("--model", default="embeddings", choices=["embeddings", "covariates", "embeddings_covariates", "embeddings_cbs"], help="Feature set to use for prediction.")
    parser.add_argument("--target", default="cv24p307_voting_behavior", choices=["cv24p307_voting_behavior", "cv24p307_voting_behavior_populist", "cv24p013_trust_government"], help="Target variable to predict.")
    parser.add_argument("--num-folds", default=5, type=int, help="Number of folds in inner and outer cross-validation loops.")
    parser.add_argument("--num-trials", default=100, type=int, help="Number of iterations in hyperparameters optimization.")
    parser.add_argument("--liss-filename", default="\data\processed\liss_preprocessed.csv", help="Path to preprocessed LISS data file with covariates.")
    parser.add_argument("--seed", default=42, help="Seed for reproducibility.")

    args = parser.parse_args(args)

    return args

@hydra.main(version_base=None, config_path="../conf", config_name="config_estimator")
def main(args):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", f"train_estimator_{os.path.basename(os.path.dirname(args.embeddings_filename))}_{args.estimator}_{args.model}_{args.target}.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
        encoding="utf-8"
    )
    df_features, X, y = preprocess_training_data(args)
    
    logger.info("Training estimator with target shape %s and features shape %s", y.shape, X.shape)

    is_classification = args.target != "cv24p013_trust_government"

    if is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    scoring = get_scoring_funs(args.target != "cv24p013_trust_government", zero_division=0)

    outer_cv = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    inner_cv = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed + 1)

    result = {f"{key}_train": [] for key in scoring} | {f"{key}_test": [] for key in scoring}

    trial_dfs = []
    best_params = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info("Tuning hyparameters on fold %s", i)
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        study = run_optimization_study(
            X_train,
            y_train,
            inner_cv,
            columns=df_features.columns,
            estimator=args.estimator,
            is_classification=is_classification,
            scoring=scoring[
                {"cv24p307_voting_behavior": "AUC_weighted",
                 "cv24p307_voting_behavior_populist": "AUC_macro",
                 "cv24p013_trust_government": "R2"}[args.target]
            ],
            num_trials=args.num_trials,
            seed=args.seed
        )

        trial_dfs.append(pl.from_pandas(study.trials_dataframe()).with_columns(pl.lit(i).alias("fold")))
        best_params.append(study.best_params)
        best_estimator = create_estimator_pipeline(df_features.columns, args.estimator, is_classification, **study.best_params)

        logger.info("Testing best hyperparameter set on fold %s", i)
        best_estimator.fit(X_train, y_train)

        for key, val in scoring.items():
            result[f"{key}_train"].append(val(best_estimator, X_train, y_train))
            result[f"{key}_test"].append(val(best_estimator, X_test, y_test))

    output_dirname = os.path.join("data", "estimators", args.target, args.estimator, args.model)

    os.makedirs(output_dirname, exist_ok=True)

    output_filename = os.path.join(output_dirname, os.path.basename(os.path.dirname(args.embeddings_filename)))

    if os.path.splitext(args.embeddings_filename)[0].split("_")[-1] == "dine":
        output_filename += "_dine"

    logger.info("Saving results to %s", output_filename)
    with open(output_filename + "_scores.json", "w") as file:
        json.dump(result, file, cls=CustomJSONEncoder, indent=4)

    with open(output_filename + "_params.json", "w") as file:
        json.dump(best_params, file, cls=CustomJSONEncoder, indent=4)

    pl.concat(trial_dfs).drop("duration").write_csv(output_filename + "_trials.csv")    


if __name__ == '__main__':
    main()
	# main(args=["--embeddings-filename", "H:/data/models/collapsed_2022_len10_window5_walks5_dim8/model.model", "--estimator", "lm", "--model", "embeddings_covariates", "--target", "cv24p307_voting_behavior_populist"])
        