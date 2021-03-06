import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sentimentanalysis import render


# params-building helper
def P(column_name=''):
    return {'column_name': column_name}


class SentimentAnalysisTest(unittest.TestCase):
    def test_empty_table(self):
        result = render(pd.DataFrame({'A': []}, dtype=str), P('A'))
        expected = pd.DataFrame({'Sentiment': [1.0], 'A': ['a']}).drop(0)
        assert_frame_equal(result, expected)

    def test_no_column_no_op(self):
        result = render(pd.DataFrame({'A': ['yes']}), P(''))
        expected = pd.DataFrame({'A': ['yes']})
        assert_frame_equal(result, expected)

    def test_sentiment(self):
        result = render(pd.DataFrame({'A': ['yes', 'no']}), P('A'))
        expected = pd.DataFrame({
            'Sentiment': [0.4019, -0.296],
            'A': ['yes', 'no']
        })
        assert_frame_equal(result, expected)

    def test_ignore_na(self):
        result = render(pd.DataFrame({'A': [np.nan, 'no']}), P('A'))
        expected = pd.DataFrame({
            'Sentiment': [np.nan, -0.296],
            'A': [np.nan, 'no']
        })
        assert_frame_equal(result, expected)
