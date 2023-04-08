import os
import json
import logging
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

def preprocess_data(df, forecast_lead=None, train_test_split=0.8):
    """Normalizes feature values and saves column means and stds to json 
       to use for converting back at inference
    """
    features = list(df.columns)
    

    test_start = int(len(df) * train_test_split)

    df_train = df.iloc[:test_start].copy()
    df_test = df.iloc[test_start:].copy()
    import json

    col_stats = {}
    for c in df_train.columns:
        #save these values to json to refer back to
        mean = df_train[c].mean()
        stdev = df_train[c].std()
        col_stats[f"{c}_mean"] = mean
        col_stats[f"{c}_std"] = stdev
        if mean == 0 and stdev == 0:
            df_train[c] = 0
            df_test[c] = 0
        else:
            df_train[c] = (df_train[c] - mean) / stdev
            df_test[c] = (df_test[c] - mean) / stdev
    with open('location_means_stds.json', 'w') as fp:
        json.dump(col_stats, fp)
    return df_train, df_test, features

class SequenceDataset(Dataset):
    def __init__(self, dataframe, features, sequence_length=30, forecast_lead=1):
        self.features = features
        self.forecast_lead = forecast_lead
        self.sequence_length = sequence_length
        self.horizon_length = horizon_length
        self.X = torch.tensor(dataframe[features].iloc[:-self.forecast_lead,:].values).float()
        self.y = torch.tensor(dataframe[features].iloc[self.forecast_lead:,:].values).float()
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i > self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding, x), 0)
        #y = self.X[i + self.forecast_lead]
        return x, self.y[i]
        
class HorizonSequenceDataset(Dataset):
    def __init__(self, dataframe, features, sequence_length=336, horizon_length=168, forecast_lead=1):
        self.features = features
        self.forecast_lead = forecast_lead
        self.sequence_length = sequence_length
        self.horizon_length = horizon_length
        self.X = torch.tensor(dataframe[features].iloc[:-self.forecast_lead,:].values).float()
        self.y = torch.tensor(dataframe[features].iloc[self.forecast_lead:,:].values).float()
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i > self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding, x), 0)
        #y = self.X[i + self.forecast_lead]
        print(f"dataloader x shape: {x.shape}")
        print(f"dataloader y shape: {self.y.shape}")
        return x, self.y[i:(i+self.horizon_length)]


class Seq2SeqDataset(Dataset):
    def __init__(self, dataframe, features, sequence_length=30, horizon_length=10, forecast_lead=1):
        self.features = features
        self.forecast_lead = forecast_lead
        self.sequence_length = sequence_length
        self.horizon_length = horizon_length
        self.X = torch.tensor(dataframe[features].iloc[:-self.forecast_lead,:].values).float()
        self.Y = torch.tensor(dataframe[features].iloc[self.forecast_lead:,:].values).float()
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i > self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
            #y = self.Y[i:(i+self.horizon_length)]
        else:
            padding_x = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding_x, x), 0)
        
        if i < self.__len__() - self.horizon_length:
            y = self.Y[i:(i+self.horizon_length)]
        else:
            dist_from_end = self.__len__() - i
            padding_y_length = self.horizon_length - dist_from_end
            padding_y = self.Y[-1].repeat(padding_y_length,1)
            y = self.Y[i:(i+self.horizon_length)]
            y = torch.cat((y, padding_y), 0)
        return x, y


def train_model(data_loader, model, loss_function, optimizer, device=torch.device("mps")):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        #X = X.to(device=device)
        #y = y.to(device=device)
        x_ = X
        for i in range(y.shape[1]):
            y_ = model(x_)
            if i == 0:
                y_pred = y_
            else:
                y_pred = torch.cat((y_pred, y_), 0)
            x_ = torch.cat((x_, y_pred.unsqueeze(0)), 1)[:, 1:, :]
        loss = loss_function(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss/num_batches
    return avg_loss

def score_model(data_loader, model, loss_function, device=torch.device("mps")):
    num_batches = len(data_loader)
    total_loss = 0
    #model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            #X = X.to(device)
            #y = y.to(device)
            if y.shape[1] == 1:
                x_ = X
                for i in range(y.shape[1]):
                    y_ = model(x_)
                    if i == 0:
                        y_pred = y_
                    else:
                        y_pred = torch.cat((y_pred, y_), 1)
                    print(f"x shape: {x_.shape}")
                    print(f"y shape: {y_pred.shape}")
                    print(f"y us shape: {y_pred.unsqueeze(0).shape}")
                    x_ = torch.cat((x_, y_pred), 1)[:, 1:, :]
            else:
                y_pred = model(X)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()
    avg_loss = total_loss/num_batches
    return avg_loss

def score_model2(data_loader, model, loss_function, device=torch.device("mps")):
    num_batches = len(data_loader)
    total_loss = 0
    #model = model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            #X = X.to(device)
            #y = y.to(device)
            x_ = X
            for i in range(y.shape[1]):
                y_ = model(x_)
                if i == 0:
                    y_pred = y_.unsqueeze(1)
                else:
                    y_pred = torch.cat((y_pred, y_.unsqueeze(1)), 1)
                print(f"x shape: {x_.shape}")
                print(f"y shape: {y_pred.shape}")
                print(f"y us shape: {y_pred.unsqueeze(0).shape}")
                x_ = torch.cat((x_, y_pred), 1)[:, 1:, :]
            loss = loss_function(y_pred, y)
            total_loss += loss.item()
    avg_loss = total_loss/num_batches
    return avg_loss

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output

def get_predictions(data_loader,model, df_test, target=None):
    #ystar_col = "Model Forecast"
    #df_test[ystar_col] = predict(data_loader, model).numpy()

    #df_out = df_test[[target, ystar_col]]

    #for c in df_out.columns:
    #    df_out[c] = df_out[c] * target_stdev + target_mean
    
    df_out = predict(data_loader, model).numpy()
    # Convert to dataframe with location column names
    # Then transform predictions back to unnormalized ((value*std)+mean)
    return df_out

def plot_predictions(df_preds, df_test):
    fig_dims = (40, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.lineplot(data=df_preds[:,3], ax=ax)
    sns.lineplot(data=df_test.loc[:, 3], ax=ax)
    plt.show()