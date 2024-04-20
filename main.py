import configparser
from src.utils import process_data_tools
from src.utils.Classifier_utils import train_model, get_classifier
from src.classes.Database import Database  # Import the Database class
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Working on {device}')

    config = configparser.ConfigParser()
    config.read("config.ini")

    db = Database()  # Create an instance of the Database class

    if db.get_table_counts() < 25000:
        process_data_tools.preprocess_data(db)
    else:
        print("data found in database...")

    train_loader, test_loader, max_features = process_data_tools.build_loaders(db, int(config['CLASSIFIER_HYPERPARAMETERS']['batch_size']))
    print('Loaders built successfully')
    #
    config['CLASSIFIER_HYPERPARAMETERS']['input_size'] = str(max_features)
    model = get_classifier(config)
    learning_rate = float(config['CLASSIFIER_HYPERPARAMETERS']['learning_rate'])
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    epochs = int(config['CLASSIFIER_HYPERPARAMETERS']['epochs'])
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs=epochs)
    #
    # # torch.save(model.state_dict(), config['MODEL']['classifier'])
    # # model.load_state_dict(torch.load(config['MODEL']['classifier']))
    # # print('Model stored successfully')

    db.close_connection()  # Close the database connection
