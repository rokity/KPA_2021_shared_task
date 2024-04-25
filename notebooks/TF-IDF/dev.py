import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy.linalg import norm
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.metrics.pairwise import linear_kernel
from  tqdm import tqdm
import warnings
from pandas import DataFrame
import json
from sklearn.metrics import accuracy_score,f1_score

"""
this method chooses the best key point for each argument
and generates a dataframe with the matches and scores
"""
def load_predictions(predictions_dir, correct_kp_list):
    arg =[]
    kp = []
    scores = []
    invalid_keypoints = set()
    with open(predictions_dir, "r") as f_in:
        res = json.load(f_in)
        for arg_id, kps in res.items():
            valid_kps = {key: value for key, value in kps.items() if key in correct_kp_list}
            invalid = {key: value for key, value in kps.items() if key not in correct_kp_list}
            for invalid_kp, _ in invalid.items():
                if invalid_kp not in invalid_keypoints:
                    #print(f"key point {invalid_kp} doesn't appear in the key points file and will be ignored")
                    invalid_keypoints.add(invalid_kp)
            if valid_kps:
                best_kp = max(valid_kps.items(), key=lambda x: x[1])
                arg.append(arg_id)
                kp.append(best_kp[0])
                scores.append(best_kp[1])
        #print(f"\tloaded predictions for {len(arg)} arguments")
        return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})

def load_kpm_data(gold_data_dir, subset, submitted_kp_file=None):
    #print("\nֿ** loading task data:")
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    if not submitted_kp_file:
        key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    else:
        key_points_file=submitted_kp_file
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")


    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)


    for desc, group in arguments_df.groupby(["stance", "topic"]):
        stance = desc[0]
        topic = desc[1]
        key_points = key_points_df[(key_points_df["stance"] == stance) & (key_points_df["topic"] == topic)]
        #print(f"\t{desc}: loaded {len(group)} arguments and {len(key_points)} key points")
    return arguments_df, key_points_df, labels_file_df


def get_predictions(predictions_file, labels_df, arg_df, kp_df):
    #print("\nֿ** loading predictions:")
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(predictions_file, kp_df["key_point_id"].unique())

    #make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")
    #print(predictions_df[predictions_df.isna().any(axis=1)])
    #handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)

    return merged_df

def data_clean(df:DataFrame) -> DataFrame :
    df=df.drop_duplicates()
    df=df.dropna()
    return df
    

def calculate_tf_idf(df,column_name):
    
    
    tf_idf =  vec.fit_transform(df[column_name])
    # tf_idf=pd.DataFrame(tf_idf.toarray(), columns=vec.get_feature_names())
    
    # compute and print the cosine similarity matrix
    
    return tf_idf

if __name__ == "__main__":
    # retrieve data and preprocess by using TfidfVectorizer
    gold_data_dir = 'kpm_data/'
    # predictions_file = sys.argv[2]
    vec = TfidfVectorizer(smooth_idf=True,use_idf=True,stop_words='english',norm='l2')
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="train")
    arg_df=data_clean(arg_df)
    kp_df=data_clean(kp_df)
    labels_df=data_clean(labels_df)
    arg_tf_idf =  vec.fit_transform(arg_df["argument"])
    kp_tf_idf = vec.fit_transform(kp_df["key_point"])
    # kp_tf_idf = vec.transform(kp_df["key_point"])

    # cosine_sim_arg = cosine_similarity(arg_tf_idf, arg_tf_idf)
    # cosine_sim_kp = cosine_similarity(kp_tf_idf, kp_tf_idf)
    results = pd.DataFrame(columns=["argument", "key_point", "prediction","label"])
    
    # calculate cosine similarity for each label by retrieving the corresponding argument and keypoint
    warnings.simplefilter("ignore")
    with tqdm(total=labels_df.shape[0]) as pbar:    
        for index, row in labels_df.iterrows():
            arg = arg_df[arg_df["arg_id"] == row["arg_id"]]["argument"].values[0]
            kp = kp_df[kp_df["key_point_id"] == row["key_point_id"]]["key_point"].values[0]
            arg_tf_idf = vec.transform([arg])
            kp_tf_idf = vec.transform([kp])
            cosine_sim_arg = linear_kernel(arg_tf_idf, arg_tf_idf)
            cosine_sim_kp = linear_kernel(kp_tf_idf, kp_tf_idf)
            cosine_sim = cosine_sim_arg * cosine_sim_kp
            prediction = cosine_sim[0][0]
            pbar.update(1)
            new_row = {"argument": arg, "key_point": kp, "prediction": prediction, "label": row["label"]}
            results=results.append(new_row, ignore_index=True)

    # results["prediction"] = results[["prediction"]].round(0)
     
    # results.to_csv("results.csv", index=False)
    results=results.astype({"label":int})
    results=results.astype({"prediction":int})
    print(results.dtypes)
    print(results.columns)
    print("accuracy : ",accuracy_score(results["label"], results["prediction"]))
    print("f1_score : ",f1_score(results["label"], results["prediction"], average='macro'))
    
    
    # for argument in range(len(arg_df)):
    #     for key_point in range(len(kp_df)):
    #         item = labels_df[(labels_df["arg_id"] == arg_df["arg_id"].iloc[argument]) &
    #                          (labels_df["key_point_id"] == kp_df['key_point_id'].iloc[key_point])]
    #         if(not item.empty):
    #             # get vector this argument and key point

    #             # kp_tf_cos = kp_tf_idf[key_point].resize(
    #             #     arg_tf_idf[argument].shape[0], arg_tf_idf[argument].shape[1])
    #             cosine_sim = linear_kernel(arg_tf_idf[argument], kp_tf_idf[key_point])
    #             results.append([arg_df["argument"].iloc[argument],
    #                             kp_df["key_point"].iloc[key_point],
    #                             cosine_sim,
    #                             item["label"].iloc[0]])
    #             print("Cosine", cosine_sim)
    #             # print("Key Point Cosine",kp_tf_idf[key_point].mean())
    #             print(f" label : {item['label'].values[0]}")
    #             print(f" arg_id : {item['arg_id']}")
    #             print(f" key_point_id : {item['key_point_id']}")
    #             print("----------------------------------------------------")
    # results.to_csv("results_traing_tf_idf_cos.csv", index=False)



        