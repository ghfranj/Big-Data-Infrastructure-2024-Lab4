import configparser
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.classes.TwitterDataset import TwitterDataset
from src.classes.Database import Database

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin1')[:30000]
    data = data.dropna()
    return data


def clean_text(text):
    cleaned_text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
    if not nltk.download('punkt', quiet=True):
        print("Punkt tokenizer already downloaded")
    tokens = nltk.word_tokenize(cleaned_text)
    return ' '.join(tokens)


def remove_stopwords(text):
    if not nltk.download('stopwords', quiet=True):
        print("Stopwords already downloaded")
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])


def vectorize(data):
    vectorizer = TfidfVectorizer(max_features=10000)
    x = vectorizer.fit_transform(data)
    return x


def preprocess_data(db):
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Load and preprocess the data
    data = load_data(config["DATA"]["main_data_file"])
    data['cleaned_text'] = data['SentimentText'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Insert training data into the database
    # for _, row in train_data.iterrows():
    #     text = row['cleaned_text']
    #     sentiment = row['Sentiment']
    #     db.insert_training_data(text, sentiment)
    data = [(row['cleaned_text'], row['Sentiment']) for _, row in train_data.iterrows()]
    data_ids = db.insert_training_data_bulk(data)
    print("train data storing in database completed successfully")
    # Insert test data into the database
    # for _, row in test_data.iterrows():
    #     text = row['cleaned_text']
    #     sentiment = row['Sentiment']
    #     db.insert_testing_data(text, sentiment)

    data = [(row['cleaned_text'], row['Sentiment']) for _, row in test_data.iterrows()]
    data_ids = db.insert_testing_data_bulk(data)
    print("test data storing in database completed successfully")

    print("Data preprocessing and storage completed successfully")



def get_train_test_data(db):
    # Get training data from the database
    train_data = db.get_training_data()
    # Get testing data from the database
    test_data = db.get_testing_data()
    # print('train_data is:', train_data)
    # Concatenate training and testing data
    data = train_data + test_data
    df = pd.DataFrame(data, columns=['id', 'cleaned_text', 'Sentiment'])

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Split data into inputs and outputs
    inputs = df['cleaned_text']
    outputs = df['Sentiment']

    # Vectorize inputs
    vectorized_data = list(vectorize(inputs))

    # Split data into training and testing sets
    split_index = int(0.2 * len(outputs))
    x_train = vectorized_data[:split_index]
    x_test = vectorized_data[split_index:]
    y_train = outputs[:split_index]
    y_test = outputs[split_index:]

    return x_train, y_train, x_test, y_test


def build_loaders(db, batch_size=64):
    config = configparser.ConfigParser()
    config.read("config.ini")
    x_train, y_train, x_test, y_test = get_train_test_data(db)
    train_dataset = TwitterDataset(x_train, y_train)
    test_dataset = TwitterDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, x_train[0].shape[1]

