from libs.classification.hyperparameter import HyperParameters
from libs.classification.model_selection import ModelSelection
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


if __name__ == "__main__":
    hyperparameters = HyperParameters([2], [1e-05], [32], [0.2], [10])
    model_name = "google-bert/bert-base-uncased"
    fileprefix = model_name.lower().replace("-", "").replace("/", "")
    model_selection = ModelSelection(
        hyperparameters, model_name, model_name, f"{fileprefix}_result.txt"
    )
    tr, vl, ts = model_selection.load_data()
    params, loss = model_selection.train(tr, vl, ts)
    logging.debug({"best_params": params, "best_loss": loss})
