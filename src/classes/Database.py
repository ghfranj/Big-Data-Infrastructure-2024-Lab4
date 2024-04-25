import numpy as np
import psycopg2
from hvac import Client
import os


class Database:
    def __init__(self):
        vault_url = os.getenv('VAULT_URL')
        vault_token = os.getenv('VAULT_TOKEN')
        vault_path = os.getenv('VAULT_PATH')
        print("connection information", vault_url, vault_token, vault_path)
        vault_client = Client(url=vault_url, token=vault_token)
        if vault_client.is_authenticated():
            print("Connected to Vault successfully.")
        else:
            print("Failed to connect to Vault.")
        database_secrets = vault_client.read(vault_path)['data']['data']
        print('got ', database_secrets)
        self.conn = psycopg2.connect(
            host= database_secrets["DB_HOST"],
            port= database_secrets["DB_PORT"],
            user= database_secrets["DB_USER"],
            password= database_secrets["DB_PASSWORD"],
            dbname= database_secrets["DB_NAME"]
        )
        self.cur = self.conn.cursor()
        # print('connected to ', database_secrets["DB_NAME"])
        self.create_training_data_table()
        self.create_testing_data_table()

    def get_table_counts(self):
        query = """
        SELECT COUNT(*) as count FROM train_data
        UNION ALL
        SELECT COUNT(*) as count FROM test_data;
        """
        self.cur.execute(query)
        return np.asarray([row[0] for row in self.cur.fetchall()]).sum()

    def create_training_data_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS train_data (
            id SERIAL PRIMARY KEY,
            text TEXT,
            sentiment VARCHAR(10)
        );
        """
        self.cur.execute(query)
        self.conn.commit()


    def create_testing_data_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS test_data (
            id SERIAL PRIMARY KEY,
            text TEXT,
            sentiment VARCHAR(10)
        );
        """
        self.cur.execute(query)
        self.conn.commit()

    def insert_training_data_bulk(self, data):
        query = """
        INSERT INTO train_data (text, sentiment)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cur.executemany(query, data)
        # data_ids = [row[0] for row in self.cur.fetchall()]
        self.conn.commit()
        # return data_ids
    def insert_training_data(self, text, sentiment):
        query = """
        INSERT INTO train_data (text, sentiment)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cur.execute(query, (text, sentiment))
        data_id = self.cur.fetchone()[0]
        self.conn.commit()
        return data_id
    def insert_testing_data_bulk(self, data):
        query = """
        INSERT INTO test_data (text, sentiment)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cur.executemany(query, data)
        # data_ids = [row[0] for row in self.cur.fetchall()]
        self.conn.commit()
        # return data_ids
    def insert_testing_data(self, text, sentiment):
        query = """
        INSERT INTO test_data (text, sentiment)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cur.execute(query, (text, sentiment))
        data_id = self.cur.fetchone()[0]
        self.conn.commit()
        return data_id

    def get_training_data(self):
        query = """
        SELECT id, text, sentiment FROM train_data;
        """
        self.cur.execute(query)
        return self.cur.fetchall()

    def get_testing_data(self):
        query = """
        SELECT id, text, sentiment FROM test_data;
        """
        self.cur.execute(query)
        return self.cur.fetchall()

    def close_connection(self):
        self.cur.close()
        self.conn.close()

# Usage example:
if __name__ == "__main__":
    db = Database()
    db.create_training_data_table()
    db.create_testing_data_table()
    data_id = db.insert_training_data("I love this product", "positive")
    print("Training data inserted with ID:", data_id)
    data_id = db.insert_testing_data("I love this product", "positive")
    print("Testing data inserted with ID:", data_id)
    db.close_connection()
