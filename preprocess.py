import pandas as pd
import pickle
import numpy as np
import os
import yaml
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

def filter_length(raw_dataset, min_length = 10):
    trajectory_lengths = raw_dataset.groupby('id')['x'].apply(len)
    filtered = trajectory_lengths[trajectory_lengths >= min_length].index.to_list()
    raw_dataset = raw_dataset[raw_dataset.id.isin(filtered)]
    return raw_dataset

def reposition(raw_dataset):
    for col in ["x", "y"]:
        raw_dataset[col] = (raw_dataset[col] - raw_dataset[col].min()) #/ (raw_dataset[col].max() - raw_dataset[col].min())
    return raw_dataset

def describle_SSD(SDD_path):
    scaler_file = os.path.join(SDD_path, 'estimated_scales.yml')
    with open(scaler_file, 'r') as hf:
        scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
    cols = ["id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    record  = []
    for scene_name in os.listdir(SDD_path):
        if not os.path.isdir(SDD_path+ scene_name): continue
        if scene_name in ["quad"]: continue
        for scene_video_id in os.listdir((SDD_path + scene_name)):
            file = os.path.join(SDD_path, scene_name, scene_video_id, "annotations.txt")
            data = pd.read_table(file, sep = " ", header= None, names = cols)
            data = data.loc[data.label == 'Pedestrian']
            sc = scales_yaml_content[scene_name][scene_video_id]['scale']
            for col in ["xmin",  "xmax", "ymin", "ymax"]:
                data[col] *= sc
            xmax = max(data["xmin"].max(), data["xmax"].max())
            ymax = max(data["ymin"].max(), data["ymax"].max())
            df = pd.DataFrame({
                'scene_name': [scene_name],
                'scene_video_id': [scene_video_id],
                'xmax': [xmax],
                'ymax': [ymax],
                'size_max':[max(xmax, ymax)],
                "labels": data.label.unique(),
                "unique pedestrian": data.id.nunique(),
            })
            record.append(df)
            print(scene_name , scene_video_id, xmax, ymax)
            # print(scene_name, data["xmin"].min(), data["ymin"].min(), data["xmin"].max(), data["ymin"].max() )
    r = pd.concat(record, ignore_index = True)
    r.to_csv(SDD_path + "r.csv")

def create_test_dataset(df, test_data_file, min_cooldown, max_cooldown):
    new_df = pd.DataFrame()
    total_length = 3000
    for pid, group in df.groupby("id"):
        trajectory_length = len(group)
        direction = 1
        cnt = 0
        cool_down = random.randint(min_cooldown, max_cooldown)
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
                cool_down = random.randint(min_cooldown, max_cooldown)
    new_df = new_df.set_index(pd.RangeIndex(start=-300, stop=-300 + len(new_df)))
    new_df.to_csv(test_data_file + "dynamic_agents.csv")

def split_train_validation_test(raw_dataset_path, testSize = 0.1, validationSize = 0.2):
    raw_dataset = pd.read_csv(os.path.join(raw_dataset_path, "rawdata.csv"))
    raw_dataset = filter_length(raw_dataset, min_length = 10)
    print("xmin","xmax",raw_dataset.x.min(), raw_dataset.x.max(), raw_dataset.y.min(), raw_dataset.y.max())

    train_validation_id, test_id = train_test_split(raw_dataset.id.unique(), test_size= testSize, random_state = 42)
    test_raw_dataset = raw_dataset[raw_dataset.id.isin(test_id)]
    create_test_dataset(test_raw_dataset, raw_dataset_path, min_cooldown=5, max_cooldown=20)

    train_id, validation_id = train_test_split(train_validation_id, test_size = validationSize, random_state = 42)
    train_raw_dataset = raw_dataset[raw_dataset.id.isin(train_id)]
    validation_raw_dataset = raw_dataset[raw_dataset.id.isin(validation_id)]
    
    print("size of training, validation, test", len(train_id), len(validation_id), len(test_id))

    train_raw_dataset.to_csv(os.path.join(raw_dataset_path, "rawdata_train.csv"))
    validation_raw_dataset.to_csv(os.path.join(raw_dataset_path, "rawdata_validation.csv"))

def preprocess_ETH(ETH_path, file): 
    #, raw_data_file, train_data_file, validation_data_file, test_data_file):
    raw_data_file =  ETH_path + file
    csv_columns = ["frame_id", "id", "x", "z", "y", "vel_x", "vel_z", "vel_y"]
    raw_dataset = pd.read_csv(raw_data_file, sep=r"\s+", header=None, names=csv_columns)
    raw_dataset["timestamp"]= raw_dataset.frame_id
    start_frame = raw_dataset.frame_id.min()
    d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
    fps = d_frame[0] * 2.5  # 2.5 is the common annotation
    for i in raw_dataset.index:
        raw_dataset.loc[i, "timestamp"] = (raw_dataset.loc[i, "frame_id"] - start_frame) / fps
    ####### raw_dataset['pos'] = raw_dataset.apply(lambda row: [row['x'], row['y']], axis=1)
    # raw_dataset = raw_dataset.loc[:, ["timestamp", "id", "x", "y"]]
    raw_dataset = raw_dataset.loc[:, ["id", "x", "y"]]
    raw_dataset["id"] = raw_dataset["id"].astype(int)
    raw_dataset = reposition(raw_dataset)
    raw_data_file_path = "./test_data/ETH/"
    os.makedirs(raw_data_file_path, exist_ok=True)
    raw_dataset.to_csv(raw_data_file_path + "rawdata.csv")
    split_train_validation_test(raw_data_file_path)

def preprocess_SSD(SDD_path, scene_name, scene_video_id, scales_yaml_content):
    scale = scales_yaml_content[scene_name][scene_video_id]['scale']
    print(scale)
    file = os.path.join(SDD_path, scene_name, scene_video_id, "annotations.txt")
    cols = ["id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
    raw_dataset = pd.read_table(file, sep = " ", header= None, names = cols)
    raw_dataset = raw_dataset.loc[raw_dataset["lost"] != 1]
    raw_dataset = raw_dataset.loc[raw_dataset["label"] == "Pedestrian"]
    raw_dataset["x"] = scale * (raw_dataset["xmin"] + raw_dataset["xmax"]) / 2
    raw_dataset["y"] = scale * (raw_dataset["ymin"] + raw_dataset["ymax"]) / 2
    raw_dataset["id"] = raw_dataset["id"].astype(int)
    results = []
    for agent_id in raw_dataset.id.unique():
        agent = raw_dataset.loc[raw_dataset.id == agent_id, :]
        n = int(30 / 2.5)  # for 30 fps to 2.5 fps
        agent = agent[agent['frame'] % n == 0]
        results.append(agent)
    
    raw_dataset = pd.concat(results, ignore_index=True)
    raw_dataset = raw_dataset[["id", "x", "y"]]
    raw_dataset = reposition(raw_dataset)
    raw_data_file_path = "./test_data/SDD-{}-{}/".format(scene_name, scene_video_id)
    os.makedirs(raw_data_file_path, exist_ok=True)
    raw_dataset.to_csv(raw_data_file_path + "rawdata.csv")
    split_train_validation_test(raw_data_file_path)

def create_training_validation_dataset(raw_dataset_path, history_length, prediction_length):
    train_raw_dataset = pd.read_csv(raw_dataset_path + "rawdata_train.csv")
    validation_raw_dataset = pd.read_csv(raw_dataset_path + "rawdata_validation.csv")

    record = {"train_dataset": train_raw_dataset, "validation_dataset": validation_raw_dataset}
    for data_name, data in record.items():
        total_sequence_length = history_length + prediction_length
        train_dataset = []
        for pid, group in train_raw_dataset.groupby("id"):
            # timestamps = group["timestamp"].values
            x_values = group["x"].values
            y_values = group["y"].values
            for i in range(len(group) - total_sequence_length + 1):
                # Extract the sequence and target
                sequence = np.column_stack((x_values[i:i+history_length], y_values[i:i+history_length]))
                target = np.column_stack((x_values[i+history_length:i+total_sequence_length], y_values[i+history_length:i+total_sequence_length]))
                train_dataset.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))
        train_data_file = "history-{}-prediction-{}-{}.pkl".format(history_length, prediction_length, data_name)
        train_data_file = os.path.join(raw_dataset_path, train_data_file)
        with open(train_data_file , 'wb') as handle:
            pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(data_path, data_name, history_length, prediction_length):
    filename = "history-{}-prediction-{}-{}.pkl".format(history_length, prediction_length, data_name)
    path_file = os.path.join(data_path, filename)
    if not os.path.exists(path_file):
        print(filename, "not existed. Creating...")
        create_training_validation_dataset(data_path, history_length, prediction_length)
        print("created")
    with open(path_file, 'rb') as file:
        training = pickle.load(file)
    print(filename, "loaded")
    return training

def preprocess_dataset():
    preprocess_ETH( ETH_path = './OpenTraj/datasets/ETH/seq_eth/', file = 'obsmat.txt')
    
    SDD_path = "./OpenTraj/datasets/SDD/"
    scene_name = "deathCircle"
    scene_video_id = "video0"
    scaler_file = os.path.join(SDD_path, 'estimated_scales.yml')
    with open(scaler_file, 'r') as hf:
        scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
    preprocess_SSD(SDD_path, scene_name, scene_video_id, scales_yaml_content)

if __name__ == "__main__":
    preprocess_dataset()
    # # raw_dataset_path = './test_data/SDD-deathCircle-video0/'
    # raw_dataset_path = './test_data/ETH/'
    # history_length = 4
    # prediction_length = 4
    # create_training_validation_dataset(raw_dataset_path, history_length, prediction_length)
    # training_length_settings = [(4, 3)] #[(history_length, prediction_length)]
    # prepare_train_validation_test(raw_dataset_path, training_length_settings)
    pass