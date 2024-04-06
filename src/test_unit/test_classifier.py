import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from src.Classifier import Classifier, get_classifier, train_model
from src.process_data import TwitterDataset
import scipy.sparse as sp


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.input_size = 100
        self.hidden_size = 64
        self.dropout_rate1 = 0.5
        self.output_size = 1
        self.model = Classifier(self.input_size, self.hidden_size, self.dropout_rate1, self.output_size)

    def test_forward(self):
        # Test the forward pass
        x = torch.randn(32, self.input_size)
        output = self.model(x)
        self.assertEqual(output.shape, torch.Size([32, self.output_size]))

    def test_get_classifier(self):
        # Test the get_classifier function
        config = {'CLASSIFIER_HYPERPARAMETERS': {'input_size': '100', 'hidden_size': '64', 'dropout_rate1': '0.5', 'output_size': '1'}}
        model = get_classifier(config)
        self.assertTrue(isinstance(model, Classifier))

    def test_train_model(self):
        # Dummy data
        # X_train_sparse = sp.csr_matrix(X_train)
        X_train = sp.csr_matrix(torch.randn(100, self.input_size))
        y_train = torch.randint(0, 2, (100,))
        X_test = sp.csr_matrix(torch.randn(20, self.input_size))
        y_test = torch.randint(0, 2, (20,))
        train_dataset = TwitterDataset(X_train, y_train)
        test_dataset = TwitterDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCELoss()

        # Train the model
        train_model(self.model, train_loader, test_loader, optimizer, criterion, epochs=3)


if __name__ == '__main__':
    unittest.main()
