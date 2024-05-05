from feature_engineering import FeatureEngineering
from download_data import download_data
from trainer import Trainer
from hyperparameter import HyperParameters
import json
import polars as pl
import os
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ModelSelection:
    def __init__(
        self,
        hyperparameters: HyperParameters,
        tokenizer: str,
        model: str,
        output_file: str,
    ) -> None:
        self.hyperparameters = hyperparameters
        self.tokenizer_name = tokenizer
        self.model_name = model
        self.output_file = output_file

    def load_data(self):
        download_data()

        used_tokenizer = self.tokenizer_name  # "microsoft/deberta-base"
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

    def write_result(self, configuration: dict, history: dict):
        with open(self.output_file, "a") as file_result:
            file_result.write(
                str(
                    {
                        "model_name": self.model_name,
                        "tokenizer_name": self.tokenizer_name,
                    }
                )
            )
            file_result.write(str(configuration))
            file_result.write("\n")
            file_result.write(json.dumps(history.__str__()))
            file_result.write("\n")
            file_result.close()

    def train(self, tr: pl.DataFrame, vl: pl.DataFrame, ts: pl.DataFrame):
        used_model = self.model_name  # "microsoft/deberta-base"
        self.hyperparameters.explodeCombination()
        all_model_loss = []
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        for configuration in self.hyperparameters.hyperparams:
            configuration = self.hyperparameters.to_dict(configuration)
            model_trainer = Trainer(
                model_name=used_model,
                model_checkpoint=None,
                train_data_dict=tr,
                val_data_dict=vl,
                param=configuration,
            )
            model, history = model_trainer.fit()
            logging.debug({"results": history})
            all_model_loss.append(history["vl_loss"][-1])
            self.write_result(configuration, history)

        best_model_loss = min(all_model_loss)
        best_loss_idx = [
            idx for idx, val in enumerate(all_model_loss) if val == best_model_loss
        ]
        best_param = self.hyperparameters.hyperparams[best_loss_idx]
        return best_param, best_model_loss
