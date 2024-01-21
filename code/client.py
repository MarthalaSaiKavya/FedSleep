import warnings
import flwr as fl
import numpy as np
import argparse
import csv
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer

# Import load_sleep_data and other necessary functions from utils
import utils
if __name__ == "__main__":
    N_CLIENTS = 40
    server_round2 = 1
    performances = []

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.node_id

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # Adjust local epoch as needed
        warm_start=True,
    )

    utils.set_initial_params(model)

    class SleepClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                (X_train_original, y_train_original), _ = utils.load_sleep_data()

                # Split the on edge data: 80% train, 20% test
                X_train = X_train_original[: int(0.8 * len(X_train_original))]
                y_train = y_train_original[: int(0.8 * len(y_train_original))]

                server_round1 = int(config['server_round'])
                #print({"server_round": server_round1})
                data_len = len(X_train)
                #print({"data_len": data_len})
                start_idx = int(((int(partition_id) + int(server_round1) - 1) / N_CLIENTS) * data_len)
                #print({"start_idx": start_idx})
                end_idx = int(((int(partition_id) + int(server_round1)) / N_CLIENTS) * data_len)
                #print({"end_idx": end_idx})

                X_train_partition = X_train[start_idx:end_idx]
                y_train_partition = y_train[start_idx:end_idx]

                if len(X_train_partition) == 0:
                    print(f"Empty partition for client {partition_id} in round {config['server_round']}")

                model.fit(X_train_partition, y_train_partition)

            # Record loss and accuracy during training
            loss = log_loss(y_train_partition, model.predict_proba(X_train_partition))
            accuracy = model.score(X_train_partition, y_train_partition)

            print(f"Training finished for round {config['server_round']}")
            
            # Get the updated model parameters
            updated_params = utils.get_model_parameters(model)
            
            return updated_params, len(X_train_partition), {}

        
        def evaluate(self, parameters, config):
            global server_round2, performances
            (X_train_eval, y_train_eval), _ = utils.load_sleep_data()

            # Split the on edge data: 80% train, 20% test
            X_test = X_train_eval[int(0.8 * len(X_train_eval)) :]
            y_test = y_train_eval[int(0.8 * len(y_train_eval)) :]
            
            #print({"server_round": server_round2})
            data_len2 = len(X_test)
            #print({"data_len": data_len2})
            start_idx = int(((int(partition_id) + int(server_round2) - 1) / N_CLIENTS) * data_len2)
            #print({"start_idx": start_idx})
            end_idx = int(((int(partition_id) + int(server_round2)) / N_CLIENTS) * data_len2)
            #print({"end_idx": end_idx})

            X_test_partition = X_test[start_idx:end_idx]
            y_test_partition = y_test[start_idx:end_idx]

            utils.set_model_params(model, parameters)
            loss = log_loss(y_test_partition, model.predict_proba(X_test_partition))
            accuracy = model.score(X_test_partition, y_test_partition)
            print({"accuracy": accuracy}, {"loss": loss})

            performances.append([partition_id, loss, accuracy])

            server_round2 = server_round2 + 1

            return loss, len(X_test_partition), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=SleepClient())

    csv_file_path = '../images/result_data.csv'

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(["Partition_ID", "Loss", "Accuracy"])

        # Append each performance entry to the CSV file
        for entry in performances:
            writer.writerow(entry)

