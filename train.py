import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_data, SequenceDataset, train_model, score_model, log, get_predictions, plot_predictions
from model import LSTMRegression
from torch.utils.tensorboard import SummaryWriter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default='14 ST-UNION SQ_entries', help="Select feature to predict future values of")
    parser.add_argument("--forecast_lead", default=15, type=int, help="Number of sequential steps ahead to predict")
    parser.add_argument("--batch_size", default=4, type=int, help="Training Batch size")
    parser.add_argument("--sequence_length", default=30, type=int, help="Sequence length")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Model learning rate")
    parser.add_argument("--hidden_units", default=16, type=int, help="Number of hidden LSTM units")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of LSTM layers in model")
    parser.add_argument("--dropout", default=0, type=float, help="probability (0-1) of LSTM units randomly dropped out during each training epoch")
    parser.add_argument("--num_epochs", default=2, type=int, help="Number of training epochs")
    return parser.parse_args()


def train(
    df, 
    target_feature, 
    forecast_lead=15,
    batch_size=32,
    sequence_length=30,
    learning_rate = 5e-5,
    num_hidden_units=16,
    num_layers=1,
    dropout=0,
    num_epochs=2
    ):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #logger = log(path="logs/",file="timeseries_training.logs")
    
    df_train, df_test, features, target = preprocess_data(
        df,
        target_feature, 
        forecast_lead=forecast_lead,
        train_test_split=0.8
        )

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
        )

    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    tb = SummaryWriter()
    model = LSTMRegression(
        num_features=len(features), 
        hidden_units=num_hidden_units,
        num_layers=num_layers,
        dropout=dropout
        )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    seqs, labels = next(iter(train_loader))
    tb.add_graph(model,seqs)
    
    test_score = score_model(test_loader, model, loss_function)

    
    for ix_epoch in range(num_epochs):
        #logger.info(f"Epoch: {ix_epoch}")
        train_score = train_model(train_loader, model, loss_function, optimizer=optimizer)
        tb.add_scalar("Train Loss", train_score, ix_epoch)
        #logger.info(f"Train score: {train_score}")
        test_score = score_model(test_loader, model, loss_function)
        tb.add_scalar("Test Loss", test_score, ix_epoch)
        #logger.info(f"Test score: {test_score}")
        print(f"Epoch {ix_epoch} -- Train Loss: {train_score}; Test Loss: {test_score}")
    tb.close()
       
    
    model_name = f"LSTM_{num_hidden_units}unit_{num_layers}layer_{sequence_length}seq.pt"
    torch.save(model, f"models/{model_name}")
    #logger.info(f"model saved to: models/{model_name}")

    df_preds = get_predictions(test_loader,model, df_test, target)
    plot_predictions(df_preds)

    return model


if __name__=="__main__":
    torch.manual_seed(101)
    args = arg_parse()

    filename = [x for x in os.listdir(path='./data/') if x[-4:]=='.csv'][0]
    data = f'data/{filename}'
    print(data)
    df = pd.read_csv(
        data,
        index_col='time'
        )

    model = train(
        df,
        target_feature=args.target,
        forecast_lead=args.forecast_lead,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        num_hidden_units=args.hidden_units,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        )
    