from libs.feature_engineering import FeatureEngineering
from libs.download_data import download_data
from libs.trainer import Trainer
from libs.hyperparameter import HyperParameters
import json
import polars as pl
import os
import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class ModelSelection:

    def __init__(self,hyperparameters:HyperParameters,tokenizer:str,model:str) -> None:
        self.hyperparameters=hyperparameters
        self.tokenizer_name = tokenizer
        self.model_name = model

    def load_data(self):
        download_data()

        used_tokenizer = self.tokenizer_name  #"microsoft/deberta-base"
        dataset_directory = "KPA_2021_shared_task/kpm_data"
        testset_directory = "KPA_2021_shared_task/test_data"

        dataset_parser = FeatureEngineering(tokenizer_name=used_tokenizer)
        tr_data_dict = dataset_parser.get_data(
            data_directory=dataset_directory, mode="train",n_rows=200
        )
        vl_data_dict = dataset_parser.get_data(
            data_directory=dataset_directory, mode="dev",n_rows=200
        )
        ts_data_dict = dataset_parser.get_data(
            data_directory=testset_directory, mode="test",n_rows=200
        )
        return tr_data_dict, vl_data_dict, ts_data_dict


    def train(self,tr: pl.DataFrame, vl: pl.DataFrame, ts: pl.DataFrame):
        used_model = self.model_name # "microsoft/deberta-base"
        self.hyperparameters.explodeCombination()

        model_trainer = Trainer(
            model_name=used_model,
            model_checkpoint=None,
            train_data_dict=tr,
            val_data_dict=vl,
            param=self.hyperparameters,
        )
        model, history = model_trainer.fit()
        logging.debug({"results": history})

        with open("model_selection_result.txt", "a") as file_result:
            file_result.write(str({"model_name":self.model_name,"tokenizer_name":self.tokenizer_name}))
            file_result.write(str(self.hyperparameters))
            file_result.write(json.dumps(history.__str__()))
            file_result.write("\n")
            file_result.close()




if __name__ == "__main__":
    hyperparameters=HyperParameters([2],[1e-05],[32],[0.2],[10])
    __model_name="microsoft/deberta-base"
    model_selection= ModelSelection(hyperparameters,__model_name,__model_name)
    tr,vl,ts=model_selection.load_data()
    model_selection.train(tr,vl,ts)

