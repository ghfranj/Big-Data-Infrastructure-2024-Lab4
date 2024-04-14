import configparser
import unittest
import json
from src.utils import process_data_tools
import numpy as np

class TestProcessDataTools(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read("config.ini")
        with open(config['TEST_DATA']['clean_text'], 'r') as f:
            self.clean_text_data = json.load(f)

        with open(config['TEST_DATA']['remove_stopwords'], 'r') as f:
            self.remove_stopwords_data = json.load(f)

        with open(config['TEST_DATA']['vectorize'], 'r') as f:
            self.vectorize_data = json.load(f)

    def test_clean_text(self):
        for sentence in self.clean_text_data['sentences']:
            result = process_data_tools.clean_text(sentence)
            expected_output = self.clean_text_data['expected_output']
            self.assertIn(result, expected_output)

    def test_remove_stopwords(self):
        for sentence, expected_output in zip(self.remove_stopwords_data['sentences'], self.remove_stopwords_data['expected_output']):
            result = process_data_tools.remove_stopwords(sentence)
            self.assertEqual(result, expected_output)

    def test_vectorize(self):
        for sentence, expected_output in zip(self.vectorize_data['sentences'], self.vectorize_data['expected_output']):
            result = process_data_tools.vectorize(sentence)
            self.assertEqual(int(result.toarray().sum()*100), int(np.asarray(expected_output).sum()*100))


if __name__ == '__main__':
    unittest.main()
