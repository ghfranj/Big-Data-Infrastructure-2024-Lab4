import configparser

from src import process_data
from src.Classifier import Classifier, train_model
import torch.nn as nn
import torch.optim as optim
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Working on {device}')
    config = configparser.ConfigParser()
    config.read("config.ini")
    process_data.preprocess_data()
    train_loader, test_loader, max_features = process_data.build_loaders()
    model = Classifier(max_features)
    criterion = nn.BCELoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs=3)
    torch.save(model.state_dict(), config['MODEL']['classifier'])
    model.load_state_dict(torch.load(config['MODEL']['classifier']))
    print('loaders_printed_successfully')


