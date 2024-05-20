#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# utils for use in the examples and tutorials
import os
import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from replace_bg_dataset import ReplaceBGDataset
from flsim.clients.base_client import Client
from flsim.trainers.sync_trainer import SyncTrainer

SEED = 2137
random.seed(SEED)
np.random.seed(SEED)


class MyDataLoader(IFLDataLoader):

    def __init__(
        self,
        train_datasets: List[Dataset],
        eval_datasets: List[Dataset],
        test_datasets: List[Dataset],
        batch_size: int,
        drop_last: bool = False,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.test_datasets = test_datasets
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_datasets, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_datasets, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_datasets, drop_last=False)

    def _batchify(
        self,
        datasets: List[Dataset],
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        for key, dataset in enumerate(datasets):
            batch = {
                "features": batchify(dataset, self.batch_size, drop_last),
            }
            yield batch


class MyUserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split: float = 0.0):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split
        user_features = list(user_data["features"])
        total = sum(len(batch) for batch in user_features)

        for features in user_features:
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += MyUserData.get_num_examples(features)
                self._eval_batches.append(MyUserData.fl_training_batch(features))
            else:
                self._num_train_batches += 1
                self._num_train_examples += MyUserData.get_num_examples(features)
                self._train_batches.append(MyUserData.fl_training_batch(features))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            yield batch

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            yield batch

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"features": torch.stack(features)}


class MyDataProvider(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader

        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: MyUserData(user_data, eval_split=eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


def build_data_provider(
    local_batch_size, num_training_clients, drop_last: bool = False
):

    external_mean = [160.87544032, 0.21893523, 1.49783614]
    external_std = [63.60143682, 1.14457581, 8.92042825]
    input_length = 12
    pred_length = 4
    path = "/home/taiello/projects/glucose-prediction/data/raw/patients"

    patients = os.listdir(path)
    np.random.seed(SEED)
    patients = [int(p.replace(".csv", "")) for p in patients if ".csv" in p]
    patients_training = np.random.choice(patients, num_training_clients, replace=False)
    # patients_training = patients[:10]
    patients_testing = list(set(patients) - set(patients_training))  # patients[50:60]
    # patients_testing = patients[10:20]
    train_datasets = []
    for p in patients_training:
        patient_df = pd.read_csv(os.path.join(path, f"{p}.csv"))
        dataset = ReplaceBGDataset(
            patient_df, input_length + pred_length, external_mean, external_std
        )
        train_datasets.append(dataset)
    avg_num_examples = np.mean([len(d) for d in train_datasets])
    std_num_examples = np.std([len(d) for d in train_datasets])
    min_num_examples = np.min([len(d) for d in train_datasets])

    test_datasets = []
    for p in patients_testing:
        patient_df = pd.read_csv(os.path.join(path, f"{p}.csv"))
        dataset = ReplaceBGDataset(
            patient_df, input_length + pred_length, external_mean, external_std
        )
        test_datasets.append(dataset)
    fl_data_loader = MyDataLoader(
        train_datasets, test_datasets, test_datasets, local_batch_size, drop_last
    )
    my_data_provider = MyDataProvider(fl_data_loader)
    print(f"Clients in total: {my_data_provider.num_train_users()}")
    return my_data_provider, avg_num_examples, std_num_examples, min_num_examples


class CNN_LSTM(nn.Module):
    def __init__(self, input_len, single_pred=True, d_in=3):
        super().__init__()
        # kernel size 7
        if single_pred:
            predict_channels = [0]
        else:
            predict_channels = list(range(d_in))
        self.input_len = input_len

        self.predict_channels = predict_channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
        )

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Tanh(),
            nn.Linear(6, len(predict_channels)),
        )

    def _forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc_layers(x[:, -1, :])
        return x

    def forward(self, whole_example):
        """
        Args:
            whole_example: (N, l, d_in)
            input_len: int
        Returns:
            (N, l, d_in) where self.predict_channels on position [input_len: ] has been changed by the prediction
        """
        whole_example_clone = whole_example.clone().detach()
        total_len = whole_example_clone.shape[1]
        input_len = self.input_len
        assert input_len < total_len

        while True:
            if input_len == total_len:
                return whole_example_clone
            x = whole_example[:, :input_len, :]
            y_hat = self._forward(x)
            whole_example_clone[:, input_len, self.predict_channels] = y_hat[
                :, self.predict_channels
            ]
            input_len += 1


class MyFLModel(IFLModel):
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        pred_length=4,
    ):
        self.model = model
        self.device = device
        self.pred_length = pred_length

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]
        labels = features[:, -self.pred_length :, 0] * 63.60143682 + 160.87544032
        if self.device is not None:
            features = features.to(self.device)

        output = (
            self.model(features)[:, -self.pred_length :, 0] * 63.60143682 + 160.87544032
        )

        if self.device is not None:
            output, labels = (
                output.to(self.device),
                labels.to(self.device),
            )

        loss = torch.nn.L1Loss()(output, labels)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        labels = labels.detach().cpu()

        del features

        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=labels,
            model_inputs=[],
        )

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        return MyUserData.fl_training_batch(features)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return MyUserData.get_num_examples(batch["features"])


class MyMetricsReporter(FLMetricsReporter):
    RMSE = "RMSE"

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = 0.0,
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_rmse = eval_metrics.get(self.RMSE, float("inf"))
        best_rmse = best_metrics.get(self.RMSE, float("inf"))
        return current_rmse < best_rmse

    def compute_scores(self) -> Dict[str, Any]:

        predictions = []
        targets = []

        for i in range(len(self.predictions_list)):
            predictions.extend(self.predictions_list[i].tolist())
            targets.extend(self.targets_list[i].tolist())

        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)

        final_rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        return {self.RMSE: final_rmse}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        rmse = scores[self.RMSE]
        return {self.RMSE: rmse}
