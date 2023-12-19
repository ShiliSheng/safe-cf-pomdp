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
    def __init__(self) -> None:
        self.input_size = 2  # Assuming 2 features (x, y) per timestep
        self.hidden_size = 64
        self.output_size = 2  # Assuming 2 features in the output
        self.history_length = 4
        self.prediction_length = 2

    def create_test_dataset(self, df, test_data_file, subgroup_id):
        new_df = pd.DataFrame()
        total_length = 300
        for pid, group in df.groupby("id"):
            trajectory_length = len(group)
            direction = 1
            cnt = 0
            cool_down = random.randint(self.history_length, 2 * self.history_length)
            indices = group.index
            index = 0
            while cnt < total_length:
                x = (group.loc[indices[index], 'x'])
                y = (group.loc[indices[index], 'y'])
                new_df.loc[cnt, str(pid) + "x"] = x
                new_df.loc[cnt, str(pid) + "y"] = y
                cnt += 1
                index += direction
                if cool_down and ( index == trajectory_length or index == -1):
                    index -= direction
                    cool_down -= 1
                if index == trajectory_length or index == -1:
                    direction *= -1
                    index += direction
                    cool_down = random.randint(self.history_length, 2 * self.history_length)

        new_df = new_df.set_index(pd.RangeIndex(start=-10, stop=-10 + len(new_df)))
        new_df.to_csv(test_data_file + "_" + str(subgroup_id) + ".csv")
        # reshaped_dataset = test_raw_dataset.pivot_table(index='timestamp',
        #                                             columns='id',
        #                                             values=['x', 'y'],
        #                                             aggfunc='first')

        # # Flatten the multi-level column index
        # reshaped_dataset.columns = [f'{col[1]}_{col[0]}' for col in reshaped_dataset.columns]
        # # Reset the index to have 'timestamp' as a regular column
        # reshaped_dataset = reshaped_dataset.fillna(-1)
        # reshaped_dataset = reshaped_dataset.reindex(sorted(reshaped_dataset.columns), axis=1)
        # reshaped_dataset = reshaped_dataset.reset_index()
        # reshaped_dataset = reshaped_dataset.set_index(pd.RangeIndex(start=-10, stop=-10 + len(reshaped_dataset)))
        # new_df.to_csv(test_data_file + "_" + str(subgroup_id) + ".csv")

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
           
    def preprocess(self, raw_data_file, train_data_file, validation_data_file, test_data_file):
        csv_columns = ["frame_id", "id", "x", "z", "y", "vel_x", "vel_z", "vel_y"]
        # read from csv => fill traj table
        raw_dataset = pd.read_csv(raw_data_file, sep=r"\s+", header=None, names=csv_columns)
        raw_dataset["timestamp"]= raw_dataset.frame_id
        start_frame = raw_dataset.frame_id.min()
        d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
        fps = d_frame[0] * 2.5  # 2.5 is the common annotation
        for i in raw_dataset.index:
            raw_dataset.loc[i, "timestamp"] = (raw_dataset.loc[i, "frame_id"] - start_frame) / fps
        ####### raw_dataset['pos'] = raw_dataset.apply(lambda row: [row['x'], row['y']], axis=1)
        raw_dataset = raw_dataset.loc[:, ["timestamp", "id", "x", "y"]]

        min_length = 10
        trajectory_lengths = raw_dataset.groupby('id')['x'].apply(len)
        filtered = trajectory_lengths[trajectory_lengths >= min_length].index.to_list()
        raw_dataset = raw_dataset[raw_dataset.id.isin(filtered)]
        raw_dataset.id.nunique()
        for col in ["x", "y"]:
            raw_dataset[col] = (raw_dataset[col] - raw_dataset[col].min()) #/ (raw_dataset[col].max() - raw_dataset[col].min())
        raw_dataset["id"] = raw_dataset["id"].astype(int)

        # print("preprocessed",raw_dataset.id.nunique())
        # print(raw_dataset.x.min(), raw_dataset.x.max(), raw_dataset.y.min(), raw_dataset.y.max())

        train_id, test_id = train_test_split(raw_dataset.id.unique(), test_size= 1 / 10, random_state=42)
        train_id, validation_id = train_test_split(train_id, test_size = 0.2, random_state = 42)
        train_id, validation_id = set(train_id), set(validation_id)
        test_id = set(test_id)

        train_raw_dataset = raw_dataset[raw_dataset.id.isin(train_id)]
        validation_raw_dataset = raw_dataset[raw_dataset.id.isin(validation_id)]
        test_raw_datasete = raw_dataset[raw_dataset.id.isin(test_id)]

        print("len", len(train_id), len(validation_id), len(test_id))
        self.create_test_dataset(test_raw_datasete, test_data_file, 0)
        # random_seed = 42
        # random.seed(random_seed)
        # n_subgroup = 10
        # # k = len(test_id) // 3 * 2
        # k = 10
        # test_id = list(test_id)
        # for i in range(n_subgroup):
        #     sub_group = set(random.sample(test_id, k))
        #     test_raw_dataset = raw_dataset[raw_dataset.id.isin(sub_group)]
        #     self.create_test_dataset(test_raw_dataset, test_data_file, i)
        n_test = len(test_id)
        # n_subgroup = 4
        # group = n_test // n_subgroup
        # for i in range(n_subgroup):
        #     sub_group = set(test_id[i * group: (i + 1)* group])
        #     test_raw_dataset = raw_dataset[raw_dataset.id.isin(sub_group)]
        #     self.create_test_dataset(test_raw_dataset, test_data_file, i)
        # train_raw_dataset.to_csv(path + "raw_train.csv")
        # validation_raw_dataset.to_csv(path + "raw_validation.csv")
        # test_raw_dataset.to_csv(path + "raw_test.csv")

        ##########
        history_length = self.history_length
        prediction_length = self.prediction_length
        
        total_sequence_length = history_length + prediction_length

        train_dataset = []
        for pid, group in train_raw_dataset.groupby("id"):
            timestamps = group["timestamp"].values
            x_values = group["x"].values
            y_values = group["y"].values
            for i in range(len(group) - total_sequence_length + 1):
                # Extract the sequence and target
                sequence = np.column_stack((x_values[i:i+history_length], y_values[i:i+history_length]))
                target = np.column_stack((x_values[i+history_length:i+total_sequence_length], y_values[i+history_length:i+total_sequence_length]))
                train_dataset.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))

        validation = []
        for pid, group in validation_raw_dataset.groupby("id"):
            timestamps = group["timestamp"].values
            x_values = group["x"].values
            y_values = group["y"].values
            for i in range(len(group) - total_sequence_length + 1):
                # Extract the sequence and target
                sequence = np.column_stack((x_values[i:i+history_length], y_values[i:i+history_length]))
                target = np.column_stack((x_values[i+history_length:i+total_sequence_length], y_values[i+history_length:i+total_sequence_length]))
                validation.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))
    
        with open(train_data_file, 'wb') as handle:
            pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(validation_data_file, 'wb') as handle:
            pickle.dump(validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, train_datafile, output_LSTM_file):
        with open(train_datafile, 'rb') as handle:
            train_dataset = pickle.load(handle)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        history_length = self.history_length
        prediction_length = self.prediction_length
        input_size = self.input_size  
        hidden_size = self.hidden_size
        output_size = self.output_size
        # Create an instance of the model
        model = LSTMModel(input_size, hidden_size, output_size * prediction_length)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 30
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
        torch.save(model_metadata, output_LSTM_file)
        print("model saved", output_LSTM_file)

    def validation(self, validation_data_file, model_file):
        with open(validation_data_file, 'rb') as handle:
            validation_dataset = pickle.load(handle)
        validation_dataset = DataLoader(validation_dataset, batch_size=64, shuffle=False)
        self.load_model(model_file)
        self.model.eval()
        total_mse = 0.0
        num_batches = len(validation_dataset)
        criterion = torch.nn.MSELoss()

        for batch_input, batch_target in validation_dataset:
            # batch_input: batch_size * history_length * feature_size
            # batch_target: batch_size * prediction_length * feature_size
            print(batch_input, batch_target)
            with torch.no_grad():  # Disable gradient computation during evaluation
                batch_input, batch_target = batch_input.to(self.device), batch_target.to(self.device)
                output = self.model(batch_input) # batch_size * (prediction_length * feature_size)
                output = output.view(-1, self.prediction_length, self.output_size)# batch_size * prediction_length * feature_size
                loss = criterion(output, batch_target)
                total_mse += loss.item()
        average_mse = total_mse / num_batches
        print(f"Average Mean Squared Error (MSE) on the validation set: {average_mse}")


    def load_model(self, model_file):
        # Create an instance of the model
        # history_length = 0, prediction_legnth = 0, input_size = 0)
        loaded_model = torch.load(model_file)
        self.input_size = loaded_model['input_size']
        self.hidden_size = loaded_model['hidden_size']
        self.output_size  = loaded_model['output_size']
        self.prediction_length = loaded_model['output_size']
        self.history_length = loaded_model['history_length']
        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size * self.prediction_length)
        self.model.load_state_dict(loaded_model['state_dict'])
        print("history length", self.history_length, "prediction length", self.prediction_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    def conformal_prediction(self, horizon):
        ACP = [()] * horizon
        return ACP

    def compute_prediction_error(self, A, B):
        N = len(A) // 2
        distance = 0
        for i in range(0, N):
            distance += ((A[i * 2] - B[i* 2]) ** 2 + (A[2 * i + 1] - B[2 * i + 1]) ** 2) ** (0.5)
            # distance += abs(A[i] - B[i]) + abs(A[i+1] - B[i+1])
        distance /= N
        # distance = np.linalg.norm(A - B)
        # print("distance",distance)
        return distance
    
    # def compute_prediction_error(self, A, B):
    #     N = len(A)
    #     distance = 0
    #     for (x, est_x) in zip(A, B):
    #         distance += (x - est_x)

if __name__ == "__main__":
    path = './OpenTraj/datasets/ETH/seq_eth/'
    raw_file =  path + 'obsmat.txt'
    lstm_model = path + 'model_weights.pth'
    train_data_file = path + 'train_dataset.pickle'
    validation_data_file = path + 'test_dataset.pickle'
    test_data_file = path + "test_dynammic_agents"

    prediction_model = Predictor()
    prediction_model.preprocess(raw_file, train_data_file, validation_data_file, test_data_file)
    
    # pred.train(train_data_file, lstm_model)
    # pred.validation(validation_data_file, lstm_model)

    # prediction_model.load_model(lstm_model)
    # H = prediction_model.prediction_length
    # dynamic_agents =  pd.read_csv(test_data_file + "_1.csv", index_col=0)
    # num_agents_tracked = len(dynamic_agents.columns) // 2 
    # starting_col_index = 0
    # max_steps = 30
    # estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 
    
    # for cur_time in range(10, 11):
    #     estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )
    #     for i, row in enumerate(estimation):
    #         estimation_moving_agents[cur_time][i+1] = row
    #         Y_est = row
    #         Y_cur = dynamic_agents.loc[cur_time + i + 1,:].values[starting_col_index:]
    #         estimation_error = prediction_model.compute_prediction_error(Y_cur, Y_est)#
    #         print(estimation_error, cur_time, i+1, )
    #         # print(row)
    #         # print(Y_cur)
    # print("done")
    # # et = pred.predict(dynamic_agents, 0)
    # # t1 = time.time()
    # # # for cur_time in range(40):
    # # #     p = pred.predit_in_batch(dynamic_agents, cur_time)
    # # t2 = time.time()
    # # for cur_time in range(4):
    # #     p = pred.predict(dynamic_agents, cur_time)
    # #     print(p, len(p))
    # # t3 = time.time()
    # # print(t2-t1, t3 -t2)
    # t = dynamic_agents.loc[-prediction_model.history_length + 1 : 0 , :]
    # print(t)
    # print(dynamic_agents.columns)