# COMP90042 Assignment3
# Authors Zepang Student ID 955698
# This file defined the dataset to use and include preprocessing, tokenization
from torch.utils.data import Dataset
import os
import json
from datetime import datetime
import torch


class TweetDataset(Dataset):
    def __init__(self, mode, max_length, tokenizer):
        filename_data = '../data/{}.data.txt'.format(mode)
        data_list = self.read_txt_data(filename_data)

        # train data
        available_tweets_ind = []

        self.data = []
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length

        if mode != 'test':
            folder_name = '../data/{}_tweets/'.format(mode)
            filename_tweets = folder_name + '{}.json'
        else:
            filename_tweets = '../data/tweet-objects/{}.json'
            # filename_tweets = '../data/covid_tweets/{}.json'
        for i in range(len(data_list)):
            if not os.path.exists(filename_tweets.format(data_list[i][0])):
                continue
            available_tweets_ind.append(i)
            temp = []
            for ind in data_list[i]:
                try:
                    temp.append(self.preprocess_data(json.load(open(filename_tweets.format(ind)))))
                except:
                    continue

            self.data.append(sorted(temp, key=lambda dic: dic['datetime_obj']))

        # train/dev label
        if mode != 'test':
            filename_label = '../data/{}.label.txt'.format(mode)
            label_list = [int(i == 'rumour') for i in self.read_txt_label(filename_label)]
            self.label = [label_list[i] for i in available_tweets_ind]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.mode != "test":
            return self.data[item], self.label[item]
        else:
            return self.data[item]

    def collate_fn(self, batch):
        if self.mode != "test":
            tweet_text = []
            labels = []
            for row, label in batch:
                text_for_one_row = []
                for tweet in row:
                    text_for_one_row.append(tweet["text"])
                tweet_text.append(self.tokenizer.sep_token.join(text_for_one_row))
                labels.append(label)
            encoding = self.tokenizer(
                tweet_text,
                max_length=self.max_length,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )

            res_dict = {'input_ids': encoding.input_ids,
                        'attention_mask': encoding['attention_mask'],
                        'labels': torch.LongTensor(labels)}
            # res_dict['retweet_count'] = torch.tensor(curr_data[0]['retweet_count'])
            # res_dict['verified'] = torch.tensor(curr_data[0]['verified'])
            # res_dict['followers_count'] = torch.tensor(curr_data[0]['followers_count'])

            return res_dict
        else:
            tweet_text = []
            for row in batch:
                text_for_one_row = []
                for tweet in row:
                    text_for_one_row.append(tweet["text"])
                tweet_text.append(self.tokenizer.sep_token.join(text_for_one_row))
            encoding = self.tokenizer(
                tweet_text,
                max_length=self.max_length,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )

            res_dict = {'input_ids': encoding['input_ids'],
                        'attention_mask': encoding['attention_mask']}
            # res_dict['retweet_count'] = torch.tensor(curr_data[0]['retweet_count'])
            # res_dict['verified'] = torch.tensor(curr_data[0]['verified'])
            # res_dict['followers_count'] = torch.tensor(curr_data[0]['followers_count'])

            return res_dict

    @staticmethod
    def read_txt_data(filename):
        with open(filename) as f:
            return [line.rstrip().split(",") for line in f]

    @staticmethod
    def read_txt_label(filename):
        with open(filename) as f:
            return [line.rstrip() for line in f]

    @staticmethod
    def preprocess_data(data):

        data['datetime_obj'] = datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S %z %Y')

        splitted = []
        for word in data['text'].split(" "):
            word = '@user' if word.startswith('@') and len(word) > 1 else word
            word = 'http' if word.startswith('http') else word
            splitted.append(word)

        return {
            # 'id': data['id'],
            'datetime_obj': data['datetime_obj'],
            'text': " ".join(splitted),
            # 'lang': data['lang'],
            # 'retweet_count': data['retweet_count'],
            # 'verified': int(data['user']['verified']),
            # 'followers_count': data['user']['followers_count'],
            # 'friends_count': data['user']['friends_count'],
            # 'favourites_count': data['user']['favourites_count'],
        }


