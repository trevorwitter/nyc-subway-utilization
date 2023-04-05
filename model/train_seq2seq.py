import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_data, SequenceDataset, train_model, score_model, log, get_predictions, plot_predictions
from model import Seq2Seq

from utils import (
    train_model,
    preprocess_data, 
    SequenceDataset, 
    score_model, 
    predict,
    get_predictions,
    plot_predictions)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_lead", default=0, type=int, help="Number of sequential steps ahead to predict")
    parser.add_argument("--batch_size", default=4, type=int, help="Training Batch size")
    parser.add_argument("--sequence_length", default=336, type=int, help="Sequence length")
    parser.add_argument("--horizon_length", default=168, type=int, help="Sequence length")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Model learning rate")
    parser.add_argument("--hidden_units", default=32, type=int, help="Number of hidden LSTM units")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of LSTM layers in model")
    parser.add_argument("--dropout", default=0.5, type=float, help="probability (0-1) of LSTM units randomly dropped out during each training epoch")
    parser.add_argument("--num_epochs", default=2, type=int, help="Number of training epochs")
    return parser.parse_args()

def train(
    df, 
    forecast_lead=1,
    batch_size=32,
    sequence_length=336,
    horizon_length=168,
    learning_rate = 5e-5,
    num_hidden_units=16,
    num_layers=1,
    dropout=0,
    num_epochs=2
):
    logger = log(path="logs/",file="timeseries_training.logs")

    df_train, df_test, features = preprocess_data(
        df,
        #forecast_lead=forecast_lead,
        train_test_split=0.8
        )
    train_dataset = SequenceDataset(
        df_train,
        features=features,
        sequence_length=sequence_length,
        horizon_length=168,
        forecast_lead=forecast_lead
        )
    test_dataset = SequenceDataset(
        df_test,
        features=features,
        sequence_length=sequence_length,
        horizon_length=horizon_length,
        forecast_lead=forecast_lead
        )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Seq2Seq(
        num_features=len(features), 
        window=sequence_length, 
        horizon=horizon_length, 
        hidden_units=num_hidden_units,
        )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    for ix_epoch in range(num_epochs):
        logger.info(f"Epoch: {ix_epoch}")
        #train_score = train_model(train_loader, model, loss_function, optimizer=optimizer)
        num_batches = len(train_loader)
        total_loss = 0
        model.train()
        
        for X, y in train_loader:
            output = model(X)
            loss = loss_function(output, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_score = total_loss/num_batches
    test_score = score_model(test_loader, model, loss_function)
    logger.info(f"Epoch {ix_epoch} -- Train Loss: {train_score}; Test Loss: {test_score}")
    model_name = f"seq2seq_{num_hidden_units}unit_{num_layers}layer_{sequence_length}seq.pt"
    torch.save(model, f"models/{model_name}")
    logger.info(f"model saved to: models/{model_name}")
    return model


if __name__ == "__main__":
    torch.manual_seed(101)
    args = arg_parse()
    filename = "../data/mta_subway_221231_100wk_dbscan.parquet"
    df = pd.read_parquet(filename)
    df = df.fillna(0)

    model = train(
        df,
        forecast_lead=args.forecast_lead,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        horizon_length=args.horizon_length,
        learning_rate=args.learning_rate,
        num_hidden_units=args.hidden_units,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
    )

    df, 
    forecast_lead=1,
    batch_size=32,
    sequence_length=336,
    horizon_length=168,
    learning_rate = 5e-5,
    num_hidden_units=16,
    num_layers=1,
    dropout=0,
    num_epochs=2
