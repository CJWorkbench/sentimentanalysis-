VADER Sentiment Analysis, using NLTK

## Setting up a test environment

1. `pipenv sync`
2. `pipenv run python -m nltk.downloader vader_lexicon`

## Running tests

`pipenv run python -m unittest discover`

## Developing

1. Write a test case in `test_sentimentanalysis.py`
2. Make it pass in `sentimentanalysis.py`
3. Submit a pull request

## Developing within Workbench

1. Install Workbench and run `bin/dev start`
2. In the Workbench directory, `bin/dev develop-module sentimentanalysis`
3. Edit this directory; `develop-module` will push the changes to Workbench
4. Edit parameters to re-render
