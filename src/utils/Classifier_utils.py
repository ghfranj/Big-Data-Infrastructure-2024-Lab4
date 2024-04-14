import torch
from tqdm import tqdm
from src.classes.Classifier import Classifier


def get_classifier(config):
    input_size = int(config['CLASSIFIER_HYPERPARAMETERS']['input_size'])
    hidden_size = int(config['CLASSIFIER_HYPERPARAMETERS']['hidden_size'])
    dropout_rate1 = float(config['CLASSIFIER_HYPERPARAMETERS']['dropout_rate1'])
    output_size = int(config['CLASSIFIER_HYPERPARAMETERS']['output_size'])
    model = Classifier(input_size, hidden_size, dropout_rate1, output_size)
    return model


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
