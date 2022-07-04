# COMP90042 Assignment3
# Authors Zepang Student ID 955698
# This file contains all the models that used in the project, you may uncomment one of them and comment all the others
# First Model BERT Fine-tuning
# Second Model BERT + LSTM
# Third Model BERT + Attention
# Fourth Model BERT + Attention + Statistics
import torch
from torch import nn
from transformers import AutoModel

# Model1 baseline bert+linear layer
class MyModel(nn.Module):
    def __init__(self, model_name):
        super(MyModel, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(model_name)
        self.cls_layer = nn.Linear(self.bert_layer.config.hidden_size, 2)

    def forward(self, seq, attn_masks):
        # From COMP90042 workshop "10-bert.ipynb"
        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, return_dict=True)
        cont_reps = outputs.last_hidden_state
        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)
        return logits


# model2 bert+LSTM
# class MyModel(nn.Module):
#     def __init__(self, model_name):
#         super(MyModel, self).__init__()
#         self.bert_layer = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert_layer.config.hidden_size
#         self.cls_layer = nn.Linear(hidden_size, 2)
#         self.lstm = nn.LSTMCell(hidden_size, hidden_size)
#
#     def forward(self, input_ids, attention_mask):
#         bert_emb = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         hidden_layers = []
#
#         for batch in range(input_ids.shape[0]):
#             h0 = bert_emb[batch, 0, :].view(1, -1)
#             c0 = bert_emb[batch, 0, :].view(1, -1)
#
#             for ind, token in enumerate(input_ids[batch]):
#                 if token == 2:
#                     h0, c0 = self.lstm(bert_emb[batch, ind, :].view(1, -1), (h0, c0))
#             hidden_layers.append(h0)
#
#         logits = self.cls_layer(torch.cat(hidden_layers, 0))
#         return logits

# model3 bert+Attention
# class LSTMAttenModel(nn.Module):
#     def __init__(self, model_name):
#         super(LSTMAttenModel, self).__init__()
# 
#         # bert embedding
#         self.bert_layer = AutoModel.from_pretrained(model_name)
#         self.hidden_size = self.bert_layer.config.hidden_size
#         self.lstm_forward = nn.LSTMCell(self.hidden_size, self.hidden_size)
#         self.cls_layer = nn.Linear(self.hidden_size, 2)
#         # self.dropout = nn.Dropout(0.2)
# 
#         # attention params
#         self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
#         self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))
# 
#         nn.init.uniform_(self.w_omega, -0.1, 0.1)
#         nn.init.uniform_(self.u_omega, -0.1, 0.1)
# 
#     def forward(self, input_ids, attention_mask):
#         bert_emb = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         hidden_layers = []
# 
#         for batch in range(input_ids.shape[0]):
#             output = torch.cat(bert_emb[batch, :], 0)
#             attn_output = self.attention_net(output)
#             hidden_layers.append(attn_output)
#         logits = self.cls_layer(torch.cat(hidden_layers, 0))
#         return logits
# 
#     def attention_net(self, x):
#         u = torch.tanh(torch.matmul(x, self.w_omega))
#         att = torch.matmul(u, self.u_omega)
#         att_score = F.softmax(att, dim=0)
#         scored_x = x * att_score
#         context = torch.sum(scored_x, dim=0)
#         return context.unsqueeze(0)


# model4 bert+Attention+Statistics
# class MyModel(nn.Module):
#     def __init__(self, model_name):
#         super(MyModel, self).__init__()
# 
#         # bert embedding
#         self.bert_layer = AutoModel.from_pretrained(model_name)
#         self.hidden_size = self.bert_layer.config.hidden_size
#         self.cls_layer = nn.Linear(self.hidden_size+2, 2)
#         self.bm = nn.BatchNorm1d(self.hidden_size + 2)
#         # self.dropout = nn.Dropout(0.2)
# 
#         # attention params
#         self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
#         self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))
# 
#         nn.init.uniform_(self.w_omega, -0.1, 0.1)
#         nn.init.uniform_(self.u_omega, -0.1, 0.1)
# 
#     def forward(self, input_ids, attention_mask, retweet_count, followers_count):
#         bert_emb = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
#         hidden_layers = []
# 
#         for batch in range(input_ids.shape[0]):
#             output = torch.cat(bert_emb[batch, :], 0)
#             attn_output = self.attention_net(output)
#             hidden_layers.append(attn_output)
# 
#         concated = torch.cat(hidden_layers, 0)
#         concated = torch.cat(
#             (concated, retweet_count.view(input_ids.shape[0], -1), followers_count.view(input_ids.shape[0], -1)), 1)
#         concated = self.bm(concated)
#         logits = self.cls_layer(concated)
#         return logits
# 
#     def attention_net(self, x):
#         u = torch.tanh(torch.matmul(x, self.w_omega))
#         att = torch.matmul(u, self.u_omega)
#         att_score = F.softmax(att, dim=0)
#         scored_x = x * att_score
#         context = torch.sum(scored_x, dim=0)
#         return context.unsqueeze(0)


