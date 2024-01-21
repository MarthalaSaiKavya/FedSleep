from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv
import os

def generate_image(arr, name):
	plt.cla()
	for client_id, row in enumerate(arr):
		plt.plot(np.arange(len(row)), row, label='client_id_'+str(client_id))
	plt.legend()
	if 'acc' in name:
		plt.ylim([-.01, 1.01])
	plt.savefig(name)

if __name__ == '__main__':
    df = pd.read_csv('../images/result_data.csv')
    
    # Convert Partition_ID to numeric, coercing non-numeric values to NaN
    df['Partition_ID'] = pd.to_numeric(df['Partition_ID'], errors='coerce')
    
    # Drop rows with NaN in Partition_ID (non-integer values)
    df = df.dropna(subset=['Partition_ID'])

    # Get unique integer Partition_ID values
    unique_partition_ids = df['Partition_ID'].unique()

    # Map unique Partition_ID values to client names
    partition_id_mapping = {partition_id: f'client_{i+1}' for i, partition_id in enumerate(unique_partition_ids)}

    # Replace Partition_ID values with client names
    df['Partition_ID'] = df['Partition_ID'].map(partition_id_mapping)

    # Create a new DataFrame for the result with the specified columns
    result_df = pd.DataFrame(columns=[
        'client_1_loss', 'client_1_accuracy',
        'client_2_loss', 'client_2_accuracy',
        'client_3_loss', 'client_3_accuracy',
        'client_4_loss', 'client_4_accuracy'
    ])

    for client_id in range(1, 5):
        loss_col = f'client_{client_id}_loss'
        acc_col = f'client_{client_id}_accuracy'

        # Filter data for the current client_id
        client_data = df[df['Partition_ID'] == f'client_{client_id}']

        # Add data to the result DataFrame
        result_df[loss_col] = client_data['Loss'].reset_index(drop=True)
        result_df[acc_col] = client_data['Accuracy'].reset_index(drop=True)

    # Save the result DataFrame to a new CSV file
    result_df.to_csv('../images/history.csv', index=False)
    history = result_df.values
    
    loss = [history.T[i] for i in range(0, history.shape[1], 2)]
    acc = [history.T[i] for i in range(1, history.shape[1], 2)]
    
    generate_image(loss, '../images/loss.png')
    generate_image(acc, '../images/acc.png')


