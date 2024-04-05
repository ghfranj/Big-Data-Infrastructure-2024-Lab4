import torch.nn as nn
import torch
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.drop5 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(4096, 32)
        self.drop2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.drop5(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(self.drop2(x))
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nTraining the model...\n")
    model.train()
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for batch in tqdm(train_loader, desc = f'Training {epoch+1}/{epochs}'):
            inputs = batch['input'].to(device)
            labels = batch['output'].to(device)
            optimizer.zero_grad()
            output = model(inputs)
            output = output.squeeze(-1).squeeze(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"\nEpoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}\n")
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in tqdm(test_loader, desc = 'Evaluating'):
                inputs = batch['input'].to(device)
                labels = batch['output'].to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(-1).squeeze(-1)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("\nAccuracy:", accuracy)
        print(' ')
