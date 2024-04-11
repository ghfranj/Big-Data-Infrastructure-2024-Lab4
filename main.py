import configparser

from src import process_data
from src.Classifier import train_model, get_classifier
import torch.nn as nn
import torch
import coverage
import dvc
import numpy
import pandas
import sklearn
import yaml
import torch
import tqdm
import nltk

if __name__ == '__main__':
    modules = {
        "coverage": coverage,
        "dvc": dvc,
        "numpy": numpy,
        "pandas": pandas,
        "scikit-learn": sklearn,
        "pyyaml": yaml,
        "torch": torch,
        "tqdm": tqdm,
        "nltk": nltk,
    }

    for module_name, module in modules.items():
        try:
            version = module.__version__
            print(f"{module_name}: {version}")
        except AttributeError:
            print(f"Cannot determine version for {module_name}")
# if __name__ == '__main__':
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f'Working on {device}')
#
#     config = configparser.ConfigParser()
#     config.read("config.ini")
#
#     process_data.preprocess_data()
#     train_loader, test_loader, max_features = process_data.build_loaders(int(config['CLASSIFIER_HYPERPARAMETERS']['batch_size']))
#     print('loaders built successfully')
#
#     config['CLASSIFIER_HYPERPARAMETERS']['input_size'] = str(max_features)
#     model = get_classifier(config)
#     learning_rate = float(config['CLASSIFIER_HYPERPARAMETERS']['learning_rate'])
#     optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
#     criterion = nn.BCELoss()
#     epochs = int(config['CLASSIFIER_HYPERPARAMETERS']['epochs'])
#     train_model(model, train_loader, test_loader, optimizer, criterion, epochs=epochs)
#
#     torch.save(model.state_dict(), config['MODEL']['classifier'])
#     model.load_state_dict(torch.load(config['MODEL']['classifier']))
#     print('Model stored successfully')


