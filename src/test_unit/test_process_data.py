import unittest
from src import process_data
from sklearn.feature_extraction.text import TfidfVectorizer

class TestProcessData(unittest.TestCase):
    def test_clean_text(self):
        result = process_data.clean_text('Hello, how are you?')
        self.assertEqual(result, "hello how are you")

        result = process_data.clean_text("I'm fine and you?")
        self.assertEqual(result, "im fine and you")


    def test_remove_stopwords(self):
        result = process_data.remove_stopwords('Hello, how are you?')
        self.assertEqual(result, "Hello, you?")

        result = process_data.clean_text("I'm fine and you?")
        self.assertEqual(result, "im fine and you")


    def test_vectorize(self):
        result = process_data.vectorize(["hello"])
        self.assertEqual(result.toarray().sum(), \
                         TfidfVectorizer(max_features=10000).fit_transform(["hello"]).toarray().sum())

        result = process_data.vectorize(["How are you?"])
        self.assertEqual(result.toarray().sum(), \
                         TfidfVectorizer(max_features=10000).fit_transform(["How are you?"]).toarray().sum())


if __name__ == '__main__':
    unittest.main()
