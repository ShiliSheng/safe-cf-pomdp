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
import pickle
import time
from itertools import chain
import random
from datetime import datetime
from preprocess import load_dataset 

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
    def __init__(self, history_length, prediction_length, input_size = 2, 
                 hidden_size = 64, output_size = 2, training_epochs = 100) -> None:
        self.input_size = input_size # Assuming 2 features (x, y) per timestep
        self.hidden_size = hidden_size
        self.output_size = output_size  # Assuming 2 features in the output
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.num_epochs = training_epochs
        self.model = None # model untrained
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
           
    def train(self, train_dataset_path):
        train_dataset = load_dataset(train_dataset_path, 'train_dataset', self.history_length, self.prediction_length)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        history_length = self.history_length
        prediction_length = self.prediction_length
        input_size = self.input_size  
        hidden_size = self.hidden_size
        output_size = self.output_size
        device = self.device

        # Create an instance of the model
        model = LSTMModel(input_size, hidden_size, output_size * prediction_length)
        model.to(device)
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = self.num_epochs
        # Assuming 'train_loader' is your DataLoader
        for epoch in range(num_epochs):
            for batch_input, batch_target in train_loader:
                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                output = model(batch_input)
                output = output.view(-1, prediction_length, output_size)
                loss = criterion(output, batch_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Print the loss after each epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Training complete

        model_metadata = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size ,
        'prediction_length': prediction_length,
        'history_length': history_length,
        'state_dict': model.state_dict()
        }
        output_LSTM_file = train_dataset_path + 'history-{}-prediction-{}.pth'.format(history_length, prediction_length)
        torch.save(model_metadata, output_LSTM_file)
        self.model = model
        print("model trained, loaded and saved", output_LSTM_file)

    def validate(self, train_dataset_path):
        validation_dataset = load_dataset(train_dataset_path, 'validation_dataset', self.history_length, self.prediction_length)
        validation_dataset = DataLoader(validation_dataset, batch_size=64, shuffle=False)
        if not self.model:
            model_file = train_dataset_path + 'history-{}-prediction-{}.pth'.format(self.history_length, self.prediction_length)
            if not os.path.exists(model_file):
                print("model not trained. Training")
                self.train(train_dataset_path)
            else:
                self.load_model(model_file)
        self.model.eval()
        total_mse = 0.0
        num_batches = len(validation_dataset)
        criterion = torch.nn.MSELoss()

        for batch_input, batch_target in validation_dataset:
            # batch_input: batch_size * history_length * feature_size
            # batch_target: batch_size * prediction_length * feature_size
            # print(batch_input, batch_target)
            with torch.no_grad():  # Disable gradient computation during evaluation
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
                output = self.model(batch_input) # batch_size * (prediction_length * feature_size)
                output = output.view(-1, self.prediction_length, self.output_size)# batch_size * prediction_length * feature_size
                loss = criterion(output, batch_target)
                total_mse += loss.item()
        average_mse = total_mse / num_batches
        print(f"Average Mean Squared Error (MSE) on the validation set: {average_mse}")

    def load_model(self, train_dataset_path):
        # Create an instance of the model
        # history_length = 0, prediction_legnth = 0, input_size = 0)
        model_file = train_dataset_path + 'history-{}-prediction-{}.pth'.format(self.history_length, self.prediction_length)
        if not os.path.exists(model_file):
            self.train(train_dataset_path)
            self.validate(train_dataset_path)
        loaded_model = torch.load(model_file)
        print(model_file, "model loaded")
        self.input_size = loaded_model['input_size']
        self.hidden_size = loaded_model['hidden_size']
        self.output_size  = loaded_model['output_size']
        self.prediction_length = loaded_model['prediction_length']
        self.history_length = loaded_model['history_length']
        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size * self.prediction_length)
        self.model.load_state_dict(loaded_model['state_dict'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def predict_case(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            model_output = self.model(x)
        model_output = model_output.view(-1, self.prediction_length, self.output_size)
        predicted_values = model_output[0, -self.prediction_length:, :].cpu().numpy()  # Move back to CPU for further processing if needed
        return predicted_values

    def get_observations_from_moving_agents(self, dynamic_agents, cur_time, starting_index = 2):
        history_length = self.history_length
        res = []
        for i in range(starting_index, len(dynamic_agents.columns), 2):
            values = dynamic_agents.loc[cur_time - history_length + 1: cur_time , [dynamic_agents.columns[i], dynamic_agents.columns[i+1]]]
            values = np.array(values)
            res.append(values)
        res = np.array(res)
        res = torch.tensor(res, dtype=torch.float32)
        return res

    def predit_in_batch(self, dynamic_agents, cur_time):
        obs = self.get_observations_from_moving_agents(dynamic_agents, cur_time)
        estimation = self._predict(obs)
        reshaped =  [list(chain(*group)) for group in zip(*estimation)]
        return reshaped
    
    def predict(self, dynamic_agents, cur_time, starting_index = 1):
        history_length = self.history_length
        estimation = []
        for i in range(starting_index, len(dynamic_agents.columns), 2):
            values = dynamic_agents.loc[cur_time - history_length + 1: cur_time, [dynamic_agents.columns[i], dynamic_agents.columns[i+1]]]
            test = torch.tensor(np.array(values), dtype=torch.float32).clone().detach().unsqueeze(0)  # Add batch dimension
            # print(values.values)
            p = self.predict_case(test)
            estimation.append(p)
        reshaped =  [list(chain(*group)) for group in zip(*estimation)]
        # [
        #  [time1_agent1, ..  time1_agentn]   
        #  [...           ..        ..    ]    
        #  [timeH_agent1, ..  timeH_agentn]   
        # ]
        return reshaped 

    def compute_prediction_error(self, truth, prediction):
        N = len(truth) // 2
        cf_distance = 0
        for i in range(0, N):
            distance = ((truth[i * 2] - prediction[i* 2]) ** 2 + (truth[2 * i + 1] - prediction[2 * i + 1]) ** 2) ** (0.5)
            cf_distance = max(cf_distance, distance)
        return cf_distance

    def compute_prediction_error_abs(self, truth, prediction):
        N = len(truth) // 2
        cf_distance = 0
        for i in range(0, N):
            distance_x = abs(truth[i * 2] - prediction[i* 2]) 
            distance_y = abs(truth[i * 2 + 1] - prediction[i* 2] + 1) 
            cf_distance = max(cf_distance, distance_x)
            cf_distance = max(cf_distance, distance_y)
        return cf_distance

    def compute_prediction_error_by_feature(self, truth, prediction):
        N = len(truth) // 2
        cf_distance_x = 0
        cf_distance_y = 0
        for i in range(0, N):
            distance_x = abs(truth[i * 2] - prediction[i* 2]) 
            distance_y = abs(truth[i * 2 + 1] - prediction[i* 2] + 1) 
            cf_distance_x = max(cf_distance_x, distance_x)
            cf_distance_y = max(cf_distance_y, distance_y)
        return cf_distance_x, cf_distance_y
    
    def create_online_dataset(self, test_dataset, num_agents_tracked):
        num_agents_total = len(test_dataset.columns) // 2 
        tracked_agent = random.sample([i for i in range(num_agents_total)], num_agents_tracked)
        tracked_agent_col = []
        for i in tracked_agent:
            tracked_agent_col.append(2 * i)
            tracked_agent_col.append(2 * i + 1)
        tracked_agent_col.sort()
        starting_row = random.randint(0, 40);
        dynamic_agents = test_dataset.iloc[starting_row:, tracked_agent_col]
        dynamic_agents = dynamic_agents.set_index(pd.RangeIndex(start=-10, stop=-10 + len(dynamic_agents)))
        return dynamic_agents
    
if __name__ == "__main__":
    path = './test_data/'
    scene = "/SDD-deathCircle-video1/"
    history_length = 4
    prediction_length = 3
    prediction_model = Predictor(history_length, prediction_length)
    
    # prediction_model.train(path + scene)
    # prediction_model.validate(path + scene)
    prediction_model.load_model(path + scene)

    H = prediction_model.prediction_length
    test_dataset =  pd.read_csv(path + scene + "dynamic_agents.csv", index_col=0)
    num_agents_tracked = 2
    starting_col_index = 0
    max_steps = 300
    estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 

    dynamic_agents = prediction_model.create_online_dataset(test_dataset, num_agents_tracked)
    for cur_time in range(10, 50):
        estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )
        for i, row in enumerate(estimation):
            estimation_moving_agents[cur_time][i+1] = row
            Y_est = row
            Y_cur = dynamic_agents.loc[cur_time + i + 1,:].values[starting_col_index:]
            estimation_error = prediction_model.compute_prediction_error(Y_cur, Y_est)#
            # print(estimation_error, cur_time, i+1, )
            # print(row)
            print(Y_cur)
