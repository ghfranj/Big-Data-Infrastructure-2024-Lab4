import configparser
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    return data

def clean_text(text):
    cleaned_text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
    tokens = nltk.word_tokenize(cleaned_text)
    return ' '.join(tokens)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def vectorize(data):
    vectorizer = TfidfVectorizer(max_features=10000)
    x = vectorizer.fit_transform(list(data['cleaned_text']))
    return x


def preprocess_data():
    config = configparser.ConfigParser()
    config.read("../config.ini")
    data = load_data(config["DATA"]["main_data_file"])
    print("cleaning texts ...")
    data['cleaned_text'] = data['SentimentText'].apply(clean_text)
    print("removing stopwords ...")
    data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)
    print("vectorizing data ...")
    x = data['cleaned_text']
    y = data['Sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Convert X_train and X_test to DataFrames with appropriate column names
    x_train_df = pd.DataFrame(data=list(x_train), columns=['cleaned_text'])
    x_test_df = pd.DataFrame(data=list(x_test), columns=['cleaned_text'])

    # Save X_train and X_test to CSV files
    x_train_df.to_csv(config['SPLIT_DATA']['x_train'], index=False)
    x_test_df.to_csv(config['SPLIT_DATA']['x_test'], index=False)

    # Convert y_train and y_test to DataFrames with appropriate column names
    y_train_df = pd.DataFrame(y_train, columns=['Sentiment'])
    y_test_df = pd.DataFrame(y_test, columns=['Sentiment'])

    # Save y_train and y_test to CSV files
    y_train_df.to_csv(config['SPLIT_DATA']['y_train'], index=False)
    y_test_df.to_csv(config['SPLIT_DATA']['y_test'], index=False)
    print("Built split files successfully")


class TwitterDataset(Dataset):
    def __init__(self, x_file_path, y_file_path):
        inputs = load_data(x_file_path)
        outputs = load_data(y_file_path)
        vectorized_data = vectorize(inputs)
        self.inputs = list(vectorized_data)
        self.outputs = list(outputs['Sentiment'])

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx].toarray()  # Convert sparse matrix to dense array
        outputs = self.outputs[idx]
        return {
            'input': torch.tensor(inputs).float(),  # Convert to PyTorch tensor
            'output': torch.tensor(outputs).float()  # Convert to PyTorch tensor
        }


def build_loaders(batch_size=16):
    config = configparser.ConfigParser()
    config.read("config.ini")
    train_dataset = TwitterDataset(config['SPLIT_DATA']['x_train'], config['SPLIT_DATA']['y_train'])
    test_dataset = TwitterDataset(config['SPLIT_DATA']['x_test'], config['SPLIT_DATA']['y_test'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

