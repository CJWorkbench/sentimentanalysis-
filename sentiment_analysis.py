from nltk.sentiment.vader import SentimentIntensityAnalyzer

def render(table, params):

    column = params['column_name']
    if column == '' or column == None:
        return table

    all_texts = table[column]
    sentiment = []

    sid = SentimentIntensityAnalyzer()

    for text in all_texts:
        if type(text) != str:
            text = str(text)
        score = sid.polarity_scores(text)
        sentiment.append(score['compound'])

    # add column to existing table
    table.insert(0, "Sentiment", pd.Series(np.asarray(sentiment, dtype=np.float32)))
    return table
