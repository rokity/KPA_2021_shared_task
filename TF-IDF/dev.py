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


def load_kpa_data(gold_data_dir, subset):

    # load arguments set, keypoint set and label set from given directory
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    # read arguments set, keypoint set and label set in csv format as pandas dataframes
    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)
    
    # label set will be ignored
    # argument set anf keypoint set will be combined as follow

    # define required list to store  pairs informations
    argument, keypoint, argument_id, keypoint_id, stance, topic = [], [], [],[], [], []

    # for each argument and for each key point under the same topic and stance 
    # ---->  create a pair  with all relative info (such as topic, stance, id, text, etc... )
    for arg, arg_id,topic_arg,stance_arg  in zip(arguments_df['argument'],arguments_df['arg_id'],arguments_df['topic'],arguments_df['stance']):
      for kp,kp_id,topic_kp,stance_kp in zip(key_points_df['key_point'],key_points_df['key_point_id'],key_points_df['topic'],key_points_df['stance']):
        if (topic_arg == topic_kp and stance_arg == stance_kp):
          
          argument.append(arg)
          argument_id.append(arg_id)
          keypoint.append(kp)
          keypoint_id.append(kp_id)
          topic.append(topic_arg)
          stance.append(stance_arg)

    # use all the generated pair to create a final dataset (a dataframe)
    dataset_df = pd.DataFrame({'arg_id':argument_id,
                               'key_point_id':keypoint_id,
                               'argument':argument,
                               'keypoint':keypoint,
                               'topic' : topic,
                               'stance': stance})
    # add a supplemntar column to store the concatenation of argument and topic
    dataset_df["arg_topic"] = dataset_df["argument"] + " " + dataset_df["topic"]

    # return final dataset
    return  arguments_df, key_points_df, labels_file_df




def calculate_tf_idf(df,column_name):
    
    
    tf_idf =  vec.fit_transform(df[column_name])
    # tf_idf=pd.DataFrame(tf_idf.toarray(), columns=vec.get_feature_names())
    
    # compute and print the cosine similarity matrix
    
    return tf_idf

if __name__ == "__main__":
    # retrieve data and preprocess by using TfidfVectorizer
    gold_data_dir = 'kpm_data/'
    # predictions_file = sys.argv[2]
    vec = TfidfVectorizer(stop_words='english',norm='l2')
    arg_df, kp_df, labels_df = load_kpa_data(gold_data_dir, subset="train")
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

    # results["prediction"] = results["prediction"].round(0).astype(int)
    # map prediction if > 0.5 to 1 else 0 
    results["prediction"] = results["prediction"].apply(lambda x: 1 if x > 0.5 else 0)

    # results["prediction"] = results["prediction"].round(0).astype(int)
    # accurancy = accuracy_score(results["label"], results["prediction"])
    # print(accurancy)
    print(results[results["label"] != results["prediction"]].shape[0])
    results.to_csv("results.csv", index=False)
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



        