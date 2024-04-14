import configparser
import json
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.classes.Classifier import Classifier
from src.utils.Classifier_utils import get_classifier, train_model
from src.classes.TwitterDataset import TwitterDataset
import scipy.sparse as sp


class TestClassifierTools(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        with open(config['TEST_DATA']['classifier_hyperparameters'], 'r') as f:
            parameters = json.load(f)
        self.config = {'CLASSIFIER_HYPERPARAMETERS': parameters}
        self.input_size = self.config['CLASSIFIER_HYPERPARAMETERS']['input_size']
        self.hidden_size = self.config['CLASSIFIER_HYPERPARAMETERS']['hidden_size']
        self.dropout_rate1 = self.config['CLASSIFIER_HYPERPARAMETERS']['dropout_rate1']
        self.output_size = self.config['CLASSIFIER_HYPERPARAMETERS']['output_size']
        self.model = Classifier(self.input_size, self.hidden_size, self.dropout_rate1, self.output_size)

    def test_forward(self):
        x = torch.randn(32, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, torch.Size([32, self.output_size]))

    def test_get_classifier(self):
        model = get_classifier(self.config)
        self.assertTrue(isinstance(model, Classifier))

    def test_train_model(self):
        x_train = sp.csr_matrix(torch.randn(100, self.input_size))
        y_train = torch.randint(0, 2, (100,))
        x_test = sp.csr_matrix(torch.randn(20, self.input_size))
        y_test = torch.randint(0, 2, (20,))
        train_dataset = TwitterDataset(x_train, y_train)
        test_dataset = TwitterDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()

        # Train the model
        train_model(self.model, train_loader, test_loader, optimizer, criterion, epochs=3)


if __name__ == '__main__':
    unittest.main()
