import configparser
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')[:30000]
    data = data.dropna()
    return data


def clean_text(text):
    cleaned_text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
    import nltk
    nltk.download('punkt')
    tokens = nltk.word_tokenize(cleaned_text)
    return ' '.join(tokens)


def remove_stopwords(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])


def vectorize(data):
    vectorizer = TfidfVectorizer(max_features=10000)
    x = vectorizer.fit_transform(data)
    return x


def preprocess_data():
    config = configparser.ConfigParser()
    config.read("config.ini")
    data = load_data(config["DATA"]["main_data_file"])
    print("cleaning texts ...")
    data['cleaned_text'] = data['SentimentText'].apply(clean_text)
    print("removing stopwords ...")
    data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

    x = data['cleaned_text']
    y = data['Sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Convert X_train and X_test to DataFrames with appropriate column names
    x_train_df = pd.DataFrame(data=x_train, columns=['cleaned_text'])
    x_test_df = pd.DataFrame(data=x_test, columns=['cleaned_text'])

    # Save X_train and X_test to CSV files
    x_train_df.to_csv(config['SPLIT_DATA']['x_train'])
    x_test_df.to_csv(config['SPLIT_DATA']['x_test'])

    # Convert y_train and y_test to DataFrames with appropriate column names
    y_train_df = pd.DataFrame(data=list(y_train), columns=['Sentiment'])
    y_test_df = pd.DataFrame(data=list(y_test), columns=['Sentiment'])

    # Save y_train and y_test to CSV files
    y_train_df.to_csv(config['SPLIT_DATA']['y_train'], index=False)
    y_test_df.to_csv(config['SPLIT_DATA']['y_test'], index=False)
    print("Built split files successfully")


class TwitterDataset(Dataset):
    def __init__(self, x_data, y_data):
        inputs = x_data
        outputs = y_data
        self.inputs = list(inputs)
        self.outputs = list(outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx].toarray()
        outputs = self.outputs[idx]
        return {
            'input': torch.tensor(inputs).float(),
            'output': torch.tensor(outputs).float()
        }


def get_train_test_data(x_train_path, y_train_path, x_test_path, y_test_path):
    inputs = list(pd.read_csv(x_train_path)['cleaned_text']) + list(pd.read_csv(x_test_path)['cleaned_text'])
    outputs = list(pd.read_csv(y_train_path)['Sentiment']) + list(pd.read_csv(y_test_path)['Sentiment'])
    inputs = pd.DataFrame(data=inputs, columns=['cleaned_text'])
    outputs = pd.DataFrame(data=outputs, columns=['Sentiment'])
    data = pd.concat([inputs, outputs], axis=1)
    data.dropna(inplace=True)
    inputs = data['cleaned_text']
    outputs = data['Sentiment']
    vectorized_data = list(vectorize(inputs))
    x_train = vectorized_data[:int(0.2*len(outputs))]
    x_test = vectorized_data[int(0.2*len(outputs)):]
    y_train = outputs[:int(0.2*len(outputs))]
    y_test = outputs[int(0.2*len(outputs)):]
    return x_train, y_train, x_test, y_test


def build_loaders(batch_size=64):
    config = configparser.ConfigParser()
    config.read("config.ini")
    x_train, y_train, x_test, y_test = get_train_test_data(config['SPLIT_DATA']['x_train'], config['SPLIT_DATA']['y_train'],\
                                                      config['SPLIT_DATA']['x_test'], config['SPLIT_DATA']['y_test'])
    train_dataset = TwitterDataset(x_train, y_train)
    test_dataset = TwitterDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, x_train[0].shape[1]

