import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score
import numpy as np
import os
import json

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def get_ap(df: pl.DataFrame, label_column: str, top_percentile: float = 0.5):
    top = int(len(df) * top_percentile)
    df = df.sort("score", descending=True).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.filter(pl.col("key_point_id") == "c").with_columns(score=0.99)
    ap = average_precision_score(
        y_true=df.select(label_column), y_score=df.select("score")
    )
    # multiply by the number of positives in top 50% and devide by the number of max positives within the top 50%, which is the number of top 50% instances
    positives_in_top_predictions = sum(df.select(label_column))
    max_num_of_positives = len(df)
    ap_retrieval = ap * positives_in_top_predictions / max_num_of_positives
    return ap_retrieval.to_numpy()


def calc_mean_average_precision(df: pl.DataFrame, label_column: str):
    precisions = [
        get_ap(group, label_column) for _, group in df.group_by(by=["topic", "stance"])
    ]

    return np.mean(precisions)


def evaluate_predictions(merged_df: pl.DataFrame):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    logging.debug(
        {"evaluation": f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}"}
    )
    return mAP_strict, mAP_relaxed


def load_kpm_data(
    gold_data_dir: str, subset: str, submitted_kp_file: str = None, n_rows=None
):
    # print("\nֿ** loading task data:")
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file = submitted_kp_file
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pl.read_csv(arguments_file, n_rows=n_rows)
    key_points_df = pl.read_csv(key_points_file, n_rows=n_rows)
    labels_file_df = pl.read_csv(labels_file, n_rows=n_rows)

    for desc, group in arguments_df.group_by(["stance", "topic"]):
        stance = desc[0]
        topic = desc[1]
        key_points = key_points_df.filter(
            (pl.col("stance") == stance) & (pl.col("topic") == topic)
        )
        # logging.info(f"\t{desc}: loaded {len(group)} arguments and {len(key_points)} key points")
    return arguments_df, key_points_df, labels_file_df


def get_predictions(
    predictions_file: str,
    labels_df: pl.DataFrame,
    arg_df: pl.DataFrame,
    kp_df: pl.DataFrame,
):
    # print("\nֿ** loading predictions:")
    arg_df = arg_df.select("arg_id", "topic", "stance")
    predictions_df = load_predictions(
        predictions_file, kp_df.select("key_point_id").unique()
    )
    # make sure each arg_id has a prediction
    predictions_df = arg_df.join(predictions_df, on="arg_id", how="left")
    # print(predictions_df[predictions_df.isna().any(axis=1)])
    # handle arguements with no matching key point
    predictions_df = predictions_df.with_columns(
        pl.when(predictions_df["key_point_id"].is_null())
        .then(pl.lit("dummy_id"))
        .otherwise(predictions_df["key_point_id"])
        .alias("key_point_id")
    )
    predictions_df = predictions_df.with_columns(
        pl.when(predictions_df["score"].is_null())
        .then(pl.lit(0))
        .otherwise(predictions_df["score"])
        .alias("score")
    )

    # merge each argument with the gold labels
    merged_df = predictions_df.join(
        labels_df, on=["arg_id", "key_point_id"], how="left"
    )
    # Filtra i record con 'key_point_id' uguale a 'dummy_id' e assegna 0 alla colonna 'label'
    merged_df = merged_df.with_columns(
        pl.when(merged_df["key_point_id"] == "dummy_id")
        .then(pl.lit(0))
        .otherwise(merged_df["label"])
        .alias("label")
    )

    # Sostituisci i valori mancanti in 'label' con 0 per 'label_strict' e con 1 per 'label_relaxed'
    merged_df = merged_df.with_columns(
        pl.when(merged_df["label"].is_null())
        .then(pl.lit(0))
        .otherwise(merged_df["label"])
        .alias("label_strict")
    )

    merged_df = merged_df.with_columns(
        pl.when(merged_df["label"].is_null())
        .then(pl.lit(1))
        .otherwise(merged_df["label"])
        .alias("label_relaxed")
    )
    return merged_df


"""
this method chooses the best key point for each argument
and generates a dataframe with the matches and scores
"""


def load_predictions(predictions_dir: str, correct_kp_list: list):
    arg = []
    kp = []
    scores = []
    invalid_keypoints = set()
    correct_kp_list = correct_kp_list["key_point_id"].to_list()
    with open(predictions_dir, "r") as f_in:
        res = json.load(f_in)
        for arg_id, kps in res.items():
            valid_kps = {
                key: value for key, value in kps.items() if key in correct_kp_list
            }
            invalid = {
                key: value for key, value in kps.items() if key not in correct_kp_list
            }
            for invalid_kp, _ in invalid.items():
                if invalid_kp not in invalid_keypoints:
                    # print(f"key point {invalid_kp} doesn't appear in the key points file and will be ignored")
                    invalid_keypoints.add(invalid_kp)
            if valid_kps:
                best_kp = max(valid_kps.items(), key=lambda x: x[1])
                arg.append(arg_id)
                kp.append(best_kp[0])
                scores.append(best_kp[1])
        return pl.DataFrame({"arg_id": arg, "key_point_id": kp, "score": scores})
