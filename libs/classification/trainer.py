import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from libs.classification.neuralnetworklayers import NeuralNetworkLayers
from torch import device, cuda, load, reshape, no_grad
import torch.nn as nn
import numpy as np
from libs.classification.metrics import compute_metrics
import polars as pl
from tqdm import tqdm
from libs.classification.hyperparameter import HyperParameters
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


class Trainer:
    def __init__(
        self, model_name, model_checkpoint, train_data_dict, val_data_dict, param
    ):
        # inizialize model and device
        self.model_name = model_name
        # get hypeparameter values
        self.epochs = param["epochs"]
        self.batch_size = param["batch_size"]
        self.learning_rate = param["learning_rate"]
        self.dropout = param["drop_out"]
        self.n_out_unit_1 = param["unit_1"]

        if model_checkpoint is None:
            #  build model architecture
            self.model = NeuralNetworkLayers(
                self.model_name, self.dropout, self.n_out_unit_1
            )
            self.device = device("cuda") if cuda.is_available() else device("cpu")
            # self.device = device("mps") 
        else:
            # load model from checkpoint
            self.model = load(model_checkpoint)
            self.device = device("cuda") if cuda.is_available() else device("cpu")
            # self.device = device("mps")
        logging.debug({"device": self.device})

        self.model.to(self.device)  # pass model to gpu
        # define TRAINING DATA
        self.tr_arguments = train_data_dict["arguments_df"]
        self.tr_keypoints = train_data_dict["keypoints_df"]
        self.tr_labels = train_data_dict["labels_df"]
        self.tr_merged = train_data_dict["merged_dataset_df"]
        self.tr_dataset = train_data_dict["tokenized_dataset_tensor"]

        # define VALIDATION DATA
        if val_data_dict is not None:
            self.vl_arguments = val_data_dict["arguments_df"]
            self.vl_keypoints = val_data_dict["keypoints_df"]
            self.vl_labels = val_data_dict["labels_df"]
            self.vl_merged = val_data_dict["merged_dataset_df"]
            self.vl_dataset = val_data_dict["tokenized_dataset_tensor"]
            self.vl_merged_pred = val_data_dict["preds"]
            self.vl_dataset_pred = val_data_dict["tokenized_preds"]

        self.tr_dataloader = None  # define null dataloader object for training
        self.vl_dataloader = None  # define null dataloader object for validation
        self.vl_dataloader_pred = None  # define null dataloader object for validation also for undediced label

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        self.scheduler = LinearLR(self.optimizer)

        self.tr_loss, self.tr_cr = None, None
        self.vl_loss, self.vl_cr, self.vl_map_strict, self.vl_map_relaxed = (
            None,
            None,
            None,
            None,
        )

        self.history = {
            "tr_loss": [],
            "tr_cr": [],
            "vl_loss": [],
            "vl_cr": [],
            "vl_map_strict": [],
            "vl_map_relaxed": [],
        }

    # ---------------------------------------- FINETUNING on TR set  ------------------------------
    def training(self):
        self.model.train()  # set model state in training mode
        logging.info("Training started!!")
        running_loss = 0  # accumulate computed loss for each batch
        running_acc = 0  # accumulate computed accuracy for each batch
        epoch_preds = []

        # iterate over batches
        for batch, dl in tqdm(enumerate(self.tr_dataloader)):
            if self.model_name.startswith("bert-") is True:
                # define input features in the current batches and move it to the employed device
                ids, tti, mask, stance, label = dl
                ids = ids.to(self.device)  # feature 1 = ids
                tti = tti.to(self.device)  # feature 2 = token types ids
                mask = mask.to(self.device)  # feature 3 = attention mask
                stance = stance.to(self.device)  # feature 4 = stance
                label = label.to(self.device)  # target = label
            else:
                ids, mask, stance, label = dl
                ids = ids.to(self.device)  # feature 1 = ids
                mask = mask.to(self.device)  # feature 3 = attention mask
                stance = stance.to(self.device)  # feature 4 = stance
                label = label.to(self.device)

            self.optimizer.zero_grad()  # clear previously computed gradients
            # ---------- FORWARD ----------
            # compute model's output (this is matching score for each sample in the current batch)
            if self.model_name.startswith("bert-") is True:
                output = self.model(ids, mask, stance, tti)
            else:
                output = self.model(ids, mask, stance)

            # compute loss for the current batches
            label = reshape(label, (label.shape[0], 1)).float()
            loss = self.loss_function(output, label)

            # ---------- BACKWARD ----------
            loss.backward()  # backpropagate loss
            # clip gradient to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()  # update model parameter
            self.scheduler.step()  # update learning rate value

            # ---------- METRICS ----------
            # accumulate loss of each batch
            running_loss += loss.item()
            # accumulate predictions (matching score) of each batch
            pred = output.detach().cpu().numpy()
            pred = np.hstack(pred)
            epoch_preds.append(pred)

        # compute loss, classification report, map strict, map relaxed for the whole training epoch
        logging.info("Computing Loss and Metrics")
        epoch_preds = np.concatenate(epoch_preds, axis=0)
        self.tr_merged = self.tr_merged.with_columns(predictions=pl.Series(epoch_preds))
        self.tr_cr = compute_metrics(
            "train",
            self.tr_merged,
            "predictions_tr.p.",
            self.tr_labels,
            self.tr_arguments,
            self.tr_keypoints,
        )

        self.tr_loss = running_loss / len(self.tr_dataloader)

        return

    # ---------------------------------------- EVALUATION on VL set  ------------------------------
    def evaluation(self):
        epoch_preds = []
        epoch_preds_map = []

        self.model.eval()
        logging.info("Evaluating...")
        running_loss = 0  # accumulate computed loss for each batch

        with no_grad():  # disable gradient calculation
            # use dataset with only pairs havinng (0,1) label to compute vl loss

            for batch, dl in tqdm(enumerate(self.vl_dataloader)):
                if self.model_name.startswith("bert-") is True:
                    ids, tti, mask, stance, label = dl
                    ids = ids.to(self.device)
                    tti = tti.to(self.device)
                    mask = mask.to(self.device)
                    stance = stance.to(self.device)
                    label = label.to(self.device)

                    output = self.model(ids, mask, stance, tti)
                else:
                    ids, mask, stance, label = dl
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)
                    stance = stance.to(self.device)
                    label = label.to(self.device)

                    output = self.model(ids, mask, stance)

                label = reshape(label, (label.shape[0], 1)).float()
                loss = self.loss_function(output, label)
                running_loss += loss.item()

                pred = output.detach().cpu().numpy()
                pred = np.hstack(pred)
                epoch_preds.append(pred)

            # this is NOT a good procedure but here we need it to not rewrite all code structures and still compute loss and others metrics
            # the use dataset with only pairs havinng (0,1, undecided) label to compute all others metrics
            for batch, dl in enumerate(self.vl_dataloader_pred):
                if self.model_name.startswith("bert-") is True:
                    ids, tti, mask, stance = dl
                    ids = ids.to(self.device)
                    tti = tti.to(self.device)
                    mask = mask.to(self.device)
                    stance = stance.to(self.device)

                    output = self.model(ids, mask, stance, tti)
                else:
                    ids, mask, stance = dl
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)
                    stance = stance.to(self.device)

                    output = self.model(ids, mask, stance)

                pred = output.detach().cpu().numpy()
                pred = np.hstack(pred)
                epoch_preds_map.append(pred)

        # compute map and classification report
        epoch_preds = np.concatenate(epoch_preds, axis=0)
        self.vl_merged = self.vl_merged.with_columns(predictions=pl.Series(epoch_preds))
        self.vl_cr = compute_metrics(
            "train",
            self.vl_merged,
            "predictions_tr.p.",
            self.vl_labels,
            self.vl_arguments,
            self.vl_keypoints,
        )
        self.vl_loss = running_loss / len(self.vl_dataloader)
        epoch_preds_map = np.concatenate(epoch_preds_map, axis=0)
        self.vl_merged_pred = self.vl_merged_pred.with_columns(
            predictions=pl.Series(epoch_preds_map)
        )
        _, _, self.vl_map_strict, self.vl_map_relaxed = compute_metrics(
            "test",
            self.vl_merged_pred,
            "predictions_vl.p.",
            self.vl_labels,
            self.vl_arguments,
            self.vl_keypoints,
        )
        return

    # ------------------------------------- RUNS fine tuning cycle ----------------------------------------------
    def fit(self, retrain=False):
        # for each training epoch
        for epoch in range(self.epochs):
            logging.info(
                "Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " started..."
            )

            # shuffle TR set and create batches
            self.tr_merged, self.tr_dataset = shuffle(self.tr_merged, self.tr_dataset)
            self.tr_dataloader = DataLoader(self.tr_dataset, batch_size=self.batch_size)

            if retrain is False:
                # if this is not the final retrain of the model

                # shuffle Vl set and create batches
                self.vl_merged, self.vl_dataset = shuffle(
                    self.vl_merged, self.vl_dataset
                )
                self.vl_dataloader = DataLoader(
                    self.vl_dataset, batch_size=self.batch_size
                )

                # shuffle VL set with also undecided labels and create batches
                self.vl_merged_pred, self.vl_dataset_pred = shuffle(
                    self.vl_merged_pred, self.vl_dataset_pred
                )
                self.vl_dataloader_pred = DataLoader(
                    self.vl_dataset_pred, batch_size=self.batch_size
                )

                self.training()  # train model on TR set
                self.evaluation()  # monitor model behaviour by evaluate them on VL set
                # note: we monitor model behaviour on both VL pairs with (0,1) labels as well as on VL pairs on (0,1,undecided) labels

                # return training and evaluation history
                self.history["tr_loss"].append(self.tr_loss)
                self.history["tr_cr"].append(self.tr_cr)
                self.history["vl_loss"].append(self.vl_loss)
                self.history["vl_cr"].append(self.vl_cr)
                self.history["vl_map_strict"].append(self.vl_map_strict)
                self.history["vl_map_relaxed"].append(self.vl_map_relaxed)

            else:
                # if this is the final retrain of the model we don't have a VL set
                # just retrain the model on TR+VL set
                self.training()

                # return training history
                self.history["tr_loss"].append(self.tr_loss)
                self.history["tr_cr"].append(self.tr_cr)
            logging.info(
                "Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " complete!!!"
            )

        return self.model, self.history
