from libs.feature_engineering import FeatureEngineering
from libs.download_data import download_data
from libs.trainer import Trainer
import json
import polars as pl
import os
import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_data():
    download_data()

    used_tokenizer = "microsoft/deberta-base"
    dataset_directory = "KPA_2021_shared_task/kpm_data"
    testset_directory = "KPA_2021_shared_task/test_data"

    dataset_parser = FeatureEngineering(tokenizer_name=used_tokenizer)
    tr_data_dict = dataset_parser.get_data(
        data_directory=dataset_directory, mode="train"
    )
    vl_data_dict = dataset_parser.get_data(
        data_directory=dataset_directory, mode="dev"
    )
    ts_data_dict = dataset_parser.get_data(
        data_directory=testset_directory, mode="test"
    )
    return tr_data_dict, vl_data_dict, ts_data_dict


def train(tr: pl.DataFrame, vl: pl.DataFrame, ts: pl.DataFrame):
    used_model = "microsoft/deberta-base"
    hyperparameter_value = {
        "epochs": 2,
        "learning_rate": 1e-05,
        "batch_size": 32,
        "drop_out": 0.2,
        "unit_1": 10,
    }

    model_trainer = Trainer(
        model_name=used_model,
        model_checkpoint=None,
        train_data_dict=tr,
        val_data_dict=vl,
        param=hyperparameter_value,
    )
    model, history = model_trainer.fit()
    logging.debug({"results": history})

    with open("model_selection_result.txt", "a") as file_result:
        file_result.write(str(hyperparameter_value))
        file_result.write(json.dumps(history))
        file_result.write("\n")
        file_result.close()


if __name__ == "__main__":
    train(*load_data())
