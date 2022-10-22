import os
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

def preprocess_data(df, forecast_lead=15, train_test_split=0.8):
    features = list(df.columns)
    #target = f"{target_feature}_lead_{forecast_lead}"

    #df[target] = df[target_feature].shift(-forecast_lead)
    #df = df.iloc[:-forecast_lead]

    test_start = int(len(df) * train_test_split)

    df_train = df.iloc[:test_start].copy()
    df_test = df.iloc[test_start:].copy()

    #target_mean = df_train[target].mean()
    #target_stdev = df_train[target].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    return df_train, df_test, features


class SequenceDataset(Dataset):
    def __init__(self, dataframe, features, sequence_length=5, forecast_lead=15):
        self.features = features
        self.forecast_lead = forecast_lead
        self.sequence_length = sequence_length
        self.X = torch.tensor(dataframe[features].iloc[:-self.forecast_lead,:].values).float()
        self.y = torch.tensor(dataframe[features].iloc[self.forecast_lead:,:].values).float()
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        if i > self.sequence_length -1:
            i_start = i - self.sequence_length +1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i+1), :]
            x = torch.cat((padding, x), 0)
        #y = self.X[i + self.forecast_lead]
        return x, self.y[i]


def train_model(data_loader, model, loss_function, optimizer, device=torch.device("mps")):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        #X = X.to(device=device)
        #y = y.to(device=device)
        output = model(X)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/num_batches
    #print(f"Train loss: {avg_loss}")
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
            output = model(X)
            total_loss += loss_function(output, y).item()
    avg_loss = total_loss/num_batches
    #print(f"Test loss: {avg_loss}")
    return avg_loss

def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output

def get_predictions(data_loader,model, df_test, target):
    ystar_col = "Model Forecast"
    #df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(data_loader, model).numpy()

    df_out = df_test[[target, ystar_col]]

    #for c in df_out.columns:
    #    df_out[c] = df_out[c] * target_stdev + target_mean
    
    return df_out

def plot_predictions(df_preds):
    fig_dims = (40, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.lineplot(data=df_preds,ax=ax)
    plt.show()