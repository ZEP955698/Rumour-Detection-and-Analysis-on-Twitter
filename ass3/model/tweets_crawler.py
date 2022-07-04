# COMP90042 Assignment3
# Authors Yanbei Jiang Student ID 1087029
# This file used to crawl the twitter object by tweet IDs
import tweepy
import json
from tqdm import tqdm
# Use your twitter Authentication Key, to get them go to https://developer.twitter.com
consumer_token = "1bRM90rGTsTk6L3J5XpYfXVcl"
consumer_secret = "3383RAW0Dzru7seYUoyvU8qg6mvMsqq0tzutq8Zu3Stv4E1xpQ"
key = "1516322935832674304-7ejiiaXfOH8dN2zVQVplR5hhEu6FJ5"
secret = "DNlJGVJQKZfHD3oGjC7w3Mra3dIaPG5q4AiWv5BpL9L9o"

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(key, secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# file to read tweet IDs
def retrieve_tweet_ids(file_in):
    tweet_ids = [tweet_id for line in open(file_in) for tweet_id in line.strip().split(",")]
    print(len(tweet_ids))
    return list(map(int, tweet_ids))


def crawl_tweets(file_in, file_out):
    tweet_ids = retrieve_tweet_ids(file_in)
    for tweet_id in tqdm(tweet_ids):
        try:
            tweet = api.get_status(tweet_id)._json
            json.dump(tweet, open(file_out + str(tweet_id) + ".json", "w"))
        except tweepy.errors.TweepyException:
            pass


if __name__ == "__main__":
    print("Crawler starts!!")
    # crawl_tweets('../data/train.data.txt', '../data/train_tweets/')
    # crawl_tweets('../data/dev.data.txt', '../data/dev_tweets/')
    crawl_tweets('../data/covid.data.txt', '../data/covid_tweets/')
    print("Crawler ends!!")