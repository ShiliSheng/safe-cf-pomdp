import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

path = './OpenTraj/datasets/ETH/seq_eth/'
LSTM_file = path + '/model_weights.pth'
input_size = 2  # Assuming 2 features (x, y) per timestep
hidden_size = 64
output_size = 2  # Assuming 2 features in the output

history_length = 5
prediction_length = 2

# Assuming you have a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Assuming you want to predict from the last timestep
        return output

class Predictor():
    def __init__(self, history_length, prediction_legnth, input_size):
        # Create an instance of the model
        pass

    def load_trained_model(self, LSTM_file):
        self.model = LSTMModel(input_size, hidden_size, output_size * prediction_length)        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the saved model weights
        self.model.load_state_dict(torch.load(LSTM_file, map_location=device))
        # Move the model to the GPU if available
        self.model.to(device)

    def predict(self, test):
        model = self.model
        x = torch.tensor(test, dtype=torch.float32).clone().detach().unsqueeze(0)  # Add batch dimension
        # Ensure the model is in evaluation mode
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            x = x.to(device)
            model_output = model(x)
        model_output = model_output.view(-1, prediction_length, output_size)
        predicted_values = model_output[0, -prediction_length:, :].cpu().numpy()  # Move back to CPU for further processing if needed
        return predicted_values

    def conformal_prediction(self, horizon):
        ACP = [()] * horizon
        return ACP

if __name__ == "__main__":

    file = path + "obsmat.txt"
    csv_columns = ["frame_id", "id", "x", "z", "y", "vel_x", "vel_z", "vel_y"]
    # read from csv => fill traj table
    raw_dataset = pd.read_csv(file, sep=r"\s+", header=None, names=csv_columns)
    raw_dataset["timestamp"]= raw_dataset.frame_id
    start_frame = raw_dataset.frame_id.min()
    d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
    fps = d_frame[0] * 2.5  # 2.5 is the common annotation
    for i in raw_dataset.index:
        raw_dataset.loc[i, "timestamp"] = (raw_dataset.loc[i, "frame_id"] - start_frame) / fps
    raw_dataset = raw_dataset.loc[:, ["timestamp", "id", "x", "y"]]

    train_id, test_id = train_test_split(raw_dataset.id.unique(), test_size= 1 / 10, random_state=42)
    train_id, test_id = set(train_id), set(test_id)

    min_lenth = 10
    total_sequence_length = history_length + prediction_length

    train_dataset = []
    test_dataset = []
    for pid, group in raw_dataset.groupby("id"):
        timestamps = group["timestamp"].values
        x_values = group["x"].values
        y_values = group["y"].values

        for i in range(len(group) - total_sequence_length + 1):
            # Extract the sequence and target
            sequence = np.column_stack((x_values[i:i+history_length], y_values[i:i+history_length]))
            target = np.column_stack((x_values[i+history_length:i+total_sequence_length], y_values[i+history_length:i+total_sequence_length]))
            if pid in train_id:
                train_dataset.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))
            else:
                test_dataset.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))

    # Convert to PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = 2  # Assuming 2 features (x, y) per timestep
    hidden_size = 64
    output_size = 2  # Assuming 2 features in the output
    # Create an instance of the model
    model = LSTMModel(input_size, hidden_size, output_size * prediction_length)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    # Assuming 'train_loader' is your DataLoader
    for epoch in range(num_epochs):
        for batch_input, batch_target in train_loader:
            # Forward pass
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)

            output = model(batch_input)
            output = output.view(-1, prediction_length, output_size)
            # Compute the loss
            loss = criterion(output, batch_target)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print the loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # Training complete

    torch.save(model.state_dict(), LSTM_file)

    pred = Predictor()
    test_a = [[7.4008, 5.3609],
         [6.9278, 5.4221],
         [6.6198, 5.3929],
         [6.1320, 5.3706],
         [5.5926, 5.3497]]
    res = pred.predict(test_a)
    print(res)