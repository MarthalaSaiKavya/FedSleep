import flwr as fl
import sys
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
from typing import Tuple, Union, List
import openml
import csv
import os
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from flwr.common import NDArrays
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import datetime
import warnings

N_CLIENTS = 40

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        # Load test data here to avoid the overhead of doing it in `evaluate` itself
        _, (X_test, y_test) = utils.load_sleep_data()
        data_len = len(X_test)
        start_idx = int((server_round / N_CLIENTS) * data_len)
        end_idx = int(((server_round + 1) / N_CLIENTS) * data_len)

        X_test = X_test.reshape(-1, X_test.shape[-1])
        X_test = np.array(X_test)

        X_test_partition = np.concatenate([X_test[:start_idx], X_test[end_idx:]])
        X_test_partition = X_test[start_idx:end_idx]
        y_test_partition = np.concatenate([y_test[:start_idx], y_test[end_idx:]]).flatten()
        y_test_partition = y_test[start_idx:end_idx]

        utils.set_model_params(model, parameters)
        loss = log_loss(y_test_partition, model.predict_proba(X_test_partition))
        accuracy = model.score(X_test_partition, y_test_partition)
        return loss, {"accuracy": accuracy}


    return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
    penalty="l2",
    max_iter=5,  # Increase the number of local epochs
    warm_start=True,
    )
    
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=4,
        evaluate_fn = get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=6),
    )
