import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
import polars as pl
from .kpa_functions import load_kpm_data

nltk.download("stopwords")


class FeatureEngineering:
    def __init__(self: object, tokenizer_name: str):
        self.arguments: pl.DataFrame = (
            None  # dataframe of arguments: [arg_id, argument, topic, stance]
        )
        self.keypoints: pl.DataFrame = (
            None  # dataframe of keypoints: [kp_id, key_point, topic, stance]
        )
        self.labels: pl.DataFrame = (
            None  # dataframe of labels:    [arg_id, kp_id, label]
        )

        self.merged_dataset: pl.DataFrame = None  # dataframe of merged information: [arg_id, argument, kp_id, key_points, topic, stance, label]
        self.tokenized_dataset = None  # tensor dataset: [ids, token_types_info, attention_mask, stance, label]
        self.preds: pl.DataFrame = None  # dataframe dataset to store model predictions, also pairs with undecided label are reported here
        self.tokenized_preds = None  # tensor dataset: [ids, token_types_info, attention_mask, stance, label]

        self.lemmatizer = WordNetLemmatizer()  # lemmatizer object loaded from nltk
        self.stemmer = PorterStemmer()  # Stemmer object loaded from nltk
        self.stop_words = set(
            stopwords.words("english")
        )  # set of stopwords for english language

        # tokenizer object loaded from hugging face pretrained tokenizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def preprocess_data(self):
        # used preprocessing technique: lower case, remove punctuations.
        # execute data preprocessing on arguments
        self.arguments = self.one_hot_encoding(self.arguments, "stance")
        self.arguments = self.col_to_lower_case(self.arguments, "argument")
        self.arguments = self.remove_punctuations(self.arguments, "argument")
        self.arguments = self.col_to_lower_case(self.arguments, "topic")
        self.arguments = self.remove_punctuations(self.arguments, "topic")

        self.keypoints = self.one_hot_encoding(self.keypoints, "stance")
        self.keypoints = self.col_to_lower_case(self.keypoints, "key_point")
        self.keypoints = self.remove_punctuations(self.keypoints, "key_point")
        self.keypoints = self.col_to_lower_case(self.keypoints, "topic")
        self.keypoints = self.remove_punctuations(self.keypoints, "topic")
        return

    # -------------------- Create a dataset merging information from arguments, keypoints and labels ----------------------------------------

    def get_merged_dataset(self):
        # create a merged dataset with all <argument, keypoint> pairs for which an annotated label (0 or 1) exist
        # do not consider <argument, keypoint> pairs with undecided label

        self.merged_dataset = self.labels.join(self.arguments, on="arg_id")
        self.merged_dataset = self.merged_dataset.join(
            self.keypoints, on="key_point_id"
        )
        self.merged_dataset = self.merged_dataset.drop(["topic", "stance"])
        self.merged_dataset = self.merged_dataset.rename(
            {"topic_right": "topic", "stance_right": "stance"}
        )
        return

    # -------------------- Tokenize dataset to properly feed it to the model -----------------------------------------------------------------

    def get_tokenized_dataset(self):
        input_ids = []
        input_tti = []
        input_mask = []
        input_stance = []
        input_label = []

        # apply tokenization on all pairs <(argument+topic), keypoint> for
        for i in range(len(self.merged_dataset)):
            encoded_input = self.tokenizer(
                self.merged_dataset["argument"][i] + self.merged_dataset["topic"][i],
                self.merged_dataset["key_point"][i],
                add_special_tokens=True,
                max_length=80,
                padding="max_length",
            )

            input_ids.append(encoded_input["input_ids"])
            if self.tokenizer_name.startswith("bert-") == True:
                input_tti.append(encoded_input["token_type_ids"])
            input_mask.append(encoded_input["attention_mask"])
            input_stance.append(self.merged_dataset["stance"][i])
            input_label.append(self.merged_dataset["label"][i])

        # trasnform to tensors
        input_ids = torch.tensor(input_ids).squeeze()
        input_mask = torch.tensor(input_mask).squeeze()
        input_stance = torch.tensor(input_stance).squeeze()
        input_label = torch.tensor(input_label).squeeze()

        # use token type id only if the used tokenizer is the bert tokenizer
        # create the final tokeinzed dataset made up with tensors
        if self.tokenizer_name.startswith("bert-") == True:
            input_tti = torch.tensor(input_tti).squeeze()
            self.tokenized_dataset = TensorDataset(
                input_ids, input_tti, input_mask, input_stance, input_label
            )
        else:
            self.tokenized_dataset = TensorDataset(
                input_ids, input_mask, input_stance, input_label
            )

        return

    def get_tokenized_preds(self):
        # do the same tokenization procedure reported above
        # here use dataset composed by all <argument keypoint> pairs, also the pairs labelled with undecided

        input_ids = []
        input_tti = []
        input_mask = []
        input_stance = []

        for i in range(len(self.preds)):
            encoded_input = self.tokenizer(
                self.preds["argument"][i] + self.preds["topic"][i],
                self.preds["key_point"][i],
                add_special_tokens=True,
                max_length=80,
                padding="max_length",
            )

            input_ids.append(encoded_input["input_ids"])
            if self.tokenizer_name.startswith("bert-") == True:
                input_tti.append(encoded_input["token_type_ids"])
            input_mask.append(encoded_input["attention_mask"])
            input_stance.append(self.preds["stance"][i])

        input_ids = torch.tensor(input_ids).squeeze()
        input_mask = torch.tensor(input_mask).squeeze()
        input_stance = torch.tensor(input_stance).squeeze()

        if self.tokenizer_name.startswith("bert-") == True:
            input_tti = torch.tensor(input_tti).squeeze()
            self.tokenized_preds = TensorDataset(
                input_ids, input_tti, input_mask, input_stance
            )
        else:
            self.tokenized_preds = TensorDataset(input_ids, input_mask, input_stance)

        return

    def get_preds(self):
        arg_pred = []
        key_point_pred = []

        arg_id_pred = []
        key_point_id_pred = []
        stance = []
        topic = []

        # crate a dataset to use to evualate the model (not for training)
        # this dataset is composed by all <argument, keypoint> pairs and include also pairs labelled as undecided
        # the pair labelled with undecided have NaN label

        for arg, arg_id, topic_arg, stance_arg in zip(
            self.arguments["argument"],
            self.arguments["arg_id"],
            self.arguments["topic"],
            self.arguments["stance"],
        ):
            for kp, kp_id, topic_kp, stance_kp in zip(
                self.keypoints["key_point"],
                self.keypoints["key_point_id"],
                self.keypoints["topic"],
                self.keypoints["stance"],
            ):
                if topic_arg == topic_kp and stance_arg == stance_kp:
                    arg_pred.append(arg)
                    arg_id_pred.append(arg_id)
                    key_point_pred.append(kp)
                    key_point_id_pred.append(kp_id)
                    topic.append(topic_arg)
                    stance.append(stance_arg)

        self.preds = pl.DataFrame(
            {
                "arg_id": arg_id_pred,
                "key_point_id": key_point_id_pred,
                "argument": arg_pred,
                "key_point": key_point_pred,
                "topic": topic,
                "stance": stance,
            }
        )

        return

    # -------------------- execute all preprocessing --------------------

    def get_data(self, data_directory, mode,n_rows=None) -> dict:
        # Load dataset using the official ArgMining function
        self.arguments, self.keypoints, self.labels = load_kpm_data(
            data_directory, mode,n_rows=n_rows
        )
        # Execute data preprocessing and cleaning
        self.preprocess_data()
        # Create a dataset with pairs <argument, keypoint> obtained merging information from argumets, keypoints and labels
        self.get_merged_dataset()
        # tokenize the already created dataset
        self.get_tokenized_dataset()
        if(mode!="train"):
            # Create a dataset with pairs <argument, keypoint> obtained merging information from argumets, keypoints
            self.get_preds()
            # tokenize the already created dataset
            self.get_tokenized_preds()
        
        
        # return a dataset dictionary with each processed information
        data_dict = {}
        data_dict["arguments_df"] = self.arguments  # arguments set
        data_dict["keypoints_df"] = self.keypoints  # keypoints set
        data_dict["labels_df"] = self.labels  # labels set

        data_dict["merged_dataset_df"] = (
            self.merged_dataset
        )  # dataset 1: <argument, keypoint, label> set with label (0,1)
        data_dict["tokenized_dataset_tensor"] = (
            self.tokenized_dataset
        )  # tokenized dataset 1
        data_dict["preds"] = (
            self.preds
        )  # dataset 2: <argument, keypoint, label> set with label (0,1, undecided)
        data_dict["tokenized_preds"] = self.tokenized_preds  # tokenized dataset 2

        return data_dict

    def remove_punctuations(self, df: pl.DataFrame, key: str) -> pl.DataFrame:
        df.with_columns(pl.col(key).str.replace(r"[^\w\s]", " "))  # remove punctuations
        return df

    def one_hot_encoding(self, df: pl.DataFrame, key: str) -> pl.DataFrame:
        df.with_columns(
            pl.when(pl.col(key) == -1).then(0).otherwise(1)
        )  # one hot encoder over stance
        return df

    def col_to_lower_case(self, df: pl.DataFrame, key: str) -> pl.DataFrame:
        df.with_columns(pl.col(key).str.to_lowercase())  # trasform to lower case
        return df

    def remove_stopwords(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(col)
            .arr.eval(pl.when(~pl.element().is_in(self.stopwords)).then(pl.element()))
            .arr.eval(pl.element().drop_nulls())
        )
        return df

    def lemmatize_words(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(col).arr.eval(True).then(self.lemmatizer.lemmatize(pl.element()))
        )
        return df

    def stemming_words(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(col).arr.eval(True).then(self.stemmer.stem(pl.element()))
        )
        return df
