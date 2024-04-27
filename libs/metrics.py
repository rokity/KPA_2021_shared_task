from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
from libs.kpa_functions import get_predictions, evaluate_predictions
import polars as pl

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def compute_metrics(
    mode: str,
    df: pl.DataFrame,
    filename: str,
    lbl_df: pl.DataFrame,
    arg_df: pl.DataFrame,
    kp_df: pl.DataFrame,
):
    # ----------------------------- Save predictions in json format -----------------------------
    # save all model predictions in json file
    # for each argumennt save the matching score with the corresponding key points
    args = {}
    for arg_, kp_, score in zip(df["arg_id"], df["key_point_id"], df["predictions"]):
        args[arg_] = {}
    for arg_, kp_, score in zip(df["arg_id"], df["key_point_id"], df["predictions"]):
        args[arg_][kp_] = score
    with open(filename, "w") as fp:
        fp.write(json.dumps(args))
        fp.close()

    merged_df = get_predictions(
        filename, lbl_df, arg_df, kp_df
    )  # DF CON PREDICTION (ARG, KP, SCORE, LABEL)
    # choose metrics to evaluate quality of model's prediction

    # ----------------------------- Metric to analyze TR performance -----------------------------
    if mode == "train":
        merged_df.write_csv("prediction_results_TRAINING.csv")
        # compute Classification Report (Accuracy, Precision, Recall, F1) and Confusion Matrix
        merged_df = merged_df.with_columns(
            pl.when(merged_df["score"] < 0.5)
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("score")
        )  # put threshold to 0.5
        merged_df = merged_df.cast({"label": pl.Int16})
        # merged_df = merged_df.cast({"score": pl.Int16})
        cr = classification_report(
            merged_df["label"], merged_df["score"], zero_division=0,output_dict=True
        )
        cm = confusion_matrix(merged_df["label"], merged_df["score"])
        return {"classification_report_training":cr,"confusion_matrix_train":cm}
    # ----------------------------- Metric to analyze VL and TS performance -----------------------------
    else:  # mode=="test" or "eval"
        merged_df.write_csv("prediction_results_TEST.csv")
        # compute mAP Strict and mAP Relaxed
        mAP_strict, mAP_relaxed = evaluate_predictions(merged_df)
        # compute Accuracy, Precision, Recall, F1, Confusion Matrix
        merged_df = merged_df.drop_nulls()  # not consider undecided label
        merged_df = merged_df.with_columns(
            pl.when(merged_df["score"] < 0.5)
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("score")
        )  # put threshold to 0.5
        merged_df = merged_df.cast({"label": pl.Int16})
        cr = classification_report(
            merged_df["label"], merged_df["score"], zero_division=0,output_dict=True
        )
        cm = confusion_matrix(merged_df["label"], merged_df["score"])

        return {"classification_report_training":dict(cr),"confusion_matrix_train":cm,"mAP_strict_test":mAP_strict,"mAP_relaxed_test":mAP_relaxed}
