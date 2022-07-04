# COMP90042 Assignment3 
# Rumour Detection
For each tweet and applies, predict rumour/non-rumour
## Requirements
Please run the following command line to install all requirements
```angular2html
pip install -r requirements.txt
```
Python version: 3.9

## Crawl Tweets
Please run the following to crawl tweets by IDs
```angular2html
cd model
python tweets_crawler.py
```
In the file tweets_crawler.py, replace the Twitter API key with yours.
 
## Project Structure
For Part1

"main.py" is the main file in this assignment,

Please run the following to start the project,
```angular2html
python main.py
```
This will eventually output a file called "submissions.csv" which contains the predictions of test set
and it is used for Kaggle competition 

config.yaml contains all the hyperparameter used in the model, you may modify them

For part2

There are three files corresponding to statistics analysis, sentiment analysis, topic analysis 
```angular2html
Covid_Tweet_Analysis.ipynb
Covid_tweet_sentiment.ipynb
Covid_tweet_Topic.ipynb
```
## Wandb
(optional)
We uses wandb to log the metrics,
if you want to use it, please run the following
```angular2html
wandb login
```
And then copy the key from the website https://wandb.ai/quickstart/pytorch-lightning