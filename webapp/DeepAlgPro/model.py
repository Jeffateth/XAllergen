import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(
            hidden_size / num_attention_heads
        )  # Fixed the syntax error here
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, return_attention=False):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)

        if return_attention:
            return context, attention_probs  # (output, weights)
        else:
            return context


class convATTnet(nn.Module):
    def __init__(self):
        super(convATTnet, self).__init__()
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 12, 24)
        self.fc1 = nn.Linear(in_features=985 * 24, out_features=1)

    def forward(self, x, return_attention=False):
        x = F.one_hot(x, num_classes=21).float()
        x = x.permute(0, 2, 1)

        # convolutional layer
        x = self.conv1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)

        # max pooling
        x = self.maxpool(x)
        x = self.dropout(x)

        # attention
        if return_attention:
            x, attn_weights = self.selfattention(x, return_attention=True)
        else:
            x = self.selfattention(x)

        x = x.view(-1, 985 * 24)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        if return_attention:
            return x, attn_weights
        else:
            return x
