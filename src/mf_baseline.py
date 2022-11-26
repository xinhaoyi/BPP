import os
import sys

import numpy as np
from data_loader import Database
from matrix_factorisation import MFEngine
from sklearn import metrics

from utils.utils import instance_bpr_loader, predict_full

sys.path.append("../src/")


class MF_train:
    """MF_train Class."""

    def __init__(self, args):
        """Initialize MF_train Class."""
        self.config = args
        self.data = Database(args["dataset"], args["task"])
        self.train_set = self.data.train
        self.test_set = self.data.test
        self.valid_set = self.data.valid
        self.n_entity = max(list(self.train_set["entity"])) + 1
        self.n_reaction = max(list(self.train_set["reaction"])) + 1
        self.config["n_entity"] = self.n_entity
        self.config["n_reaction"] = self.n_reaction
        self.best_model = None

    def train(self):
        """Train the model."""

        train_loader = instance_bpr_loader(
            data=self.train_set,
            batch_size=self.config["batch_size"],
            device=self.config["device_str"],
            n_entity=self.n_entity,
            n_reaction=self.n_reaction,
        )

        self.engine = MFEngine(self.config)
        self.model_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["save_name"]
        )
        best_valid_performance = 0
        best_epoch = 0
        epoch_bar = range(self.config["max_epoch"])
        for epoch in epoch_bar:
            print("Epoch", epoch)
            self.engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            validation_set = self.valid_set
            n_samples = len(validation_set)
            predictions = predict_full(validation_set, self.engine)
            NDCG = self.evaluate(predictions, n_samples)
            if NDCG > best_valid_performance:
                best_valid_performance = NDCG
                best_epoch = epoch
                self.best_model = self.engine
            print("NDCG", NDCG)

        print("BEST performenace on validation set is %f" % NDCG)
        print("BEST performance happens at epoch", best_epoch)
        return best_valid_performance

    def test(self):
        """Evaluate the performance for the testing sets based on the best performing model."""
        best_model = self.best_model
        test_set = self.test_set
        predictions = predict_full(test_set, best_model)
        n_samples = len(test_set)
        NDCG = self.evaluate(predictions, n_samples)
        print("Test performance is ", NDCG)
        return NDCG

    def evaluate(self, predictions, n_samples):

        predictions = predictions.reshape(
            n_samples, int(predictions.shape[0] / n_samples)
        )
        ground_truth = np.zeros(int(predictions.shape[1]))
        ground_truth[0] = 1
        new = []
        for i in range(n_samples):
            new.append(list(ground_truth))

        ground_truth = np.array(new)

        NDCG = metrics.ndcg_score(ground_truth, predictions)

        return NDCG


name = "Disease"
task = "input link prediction dataset"
args = {
    "batch_size": 64,
    "learning_rate": 0.01,
    "emb_dim": 64,
    "dataset": name,
    "task": task,
}
args["num_negative"] = 100
args["device_str"] = "cpu"
args["model"] = "MF"
args["model_save_dir"] = "model_checkpoint"
args["optimizer"] = "adam"
args["lr"] = 0.001
args["run_dir"] = "runs/"
args["save_name"] = "mf.model"
args["max_epoch"] = 100
MF_disease = MF_train(args)
MF_disease.train()
MF_disease.test()
