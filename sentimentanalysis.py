from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


def render(table, params):
    column = params['column_name']
    if not column:
        return table

    texts = table[column]
    sentiment = []

    # TODO quick-fix instead of converting to str
    na = texts.isna()
    texts = texts.astype(str)
    texts[na] = np.nan

    sid = SentimentIntensityAnalyzer()

    def _get_sentiment(text):
        return sid.polarity_scores(text)['compound']

    sentiment = texts.map(_get_sentiment, na_action='ignore')
    sentiment = sentiment.astype(np.float64)  # in case the table is empty

    # add column to existing table
    table.insert(0, "Sentiment", sentiment)
    return table
