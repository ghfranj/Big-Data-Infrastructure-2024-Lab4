import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate1, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop5 = nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.drop2 = nn.Dropout(dropout_rate1)
        self.fc3 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.drop5(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(self.drop2(x))
        x = self.sigmoid(x)
        return x
