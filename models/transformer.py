import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, head_num, p):
        super().__init__()

        self.head_num = head_num
        self.hidden_dim = input_dim

        self.dropout = nn.Dropout(p)

        self.query_layer = nn.Linear(input_dim, self.hidden_dim * head_num)
        self.key_layer = nn.Linear(input_dim, self.hidden_dim * head_num)
        self.value_layer = nn.Linear(input_dim, self.hidden_dim * head_num)

        self.output_layer = nn.Linear(self.hidden_dim * head_num, input_dim)

    def _scale_dot_product_attention(self, query, key, value, mask=None):
        scores = torch.div(torch.matmul(query, key.transpose(-2, -1)), math.sqrt(query.size(-1)))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = self.dropout(func.softmax(scores, dim=-1))

        return torch.matmul(attention, value)

    def forward(self, query, key, value, pos, mask=None):

        query = self.query_layer(query + pos)
        key = self.key_layer(key + pos)
        value = self.value_layer(value + pos)

        new_shape = query.size()[:-1] + (self.head_num, self.hidden_dim)
        query = query.view(*new_shape).permute(0, 2, 1, 3)
        key = key.view(*new_shape).permute(0, 2, 1, 3)
        value = value.view(*new_shape).permute(0, 2, 1, 3)

        output = self._scale_dot_product_attention(query, key, value, mask).permute(0, 2, 1, 3).contiguous()
        new_shape = output.size()[:2] + (self.head_num * self.hidden_dim, )
        output = output.view(*new_shape)

        return self.output_layer(output)


class TimeAwareMultiHeadAttention(nn.Module):

    def __init__(self, input_dim, head_num, p):
        super().__init__()

        self.head_num = head_num
        self.hidden_dim = input_dim

        self.dropout = nn.Dropout(p)

        self.query_layer = nn.Linear(input_dim, input_dim * head_num)
        self.key_layer = nn.Linear(input_dim, input_dim * head_num)
        self.value_layer = nn.Linear(input_dim, input_dim * head_num)
        self.pos_layer = nn.Linear(input_dim, head_num)

        self.output_layer = nn.Linear(input_dim * head_num, input_dim)

    def _scale_dot_product_attention(self, query, key, value, pos, mask=None):
        f_scores = torch.div(torch.matmul(query, key.transpose(-2, -1)), math.sqrt(query.size(-1)))

        shape = pos.size()[:2] + pos.size()[1:]
        pos_feature = torch.abs(pos.unsqueeze(-2).expand(shape) - pos.unsqueeze(-3).expand(shape))
        p_scores = self.pos_layer(pos_feature).permute(0, 3, 1, 2)
        scores = self.dropout(f_scores + p_scores)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = func.softmax(scores, dim=-1)
        return torch.matmul(attention, value)

    def forward(self, query, key, value, pos, mask=None):

        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        new_shape = query.size()[:-1] + (self.head_num, self.hidden_dim)
        query = query.view(*new_shape).permute(0, 2, 1, 3)
        key = key.view(*new_shape).permute(0, 2, 1, 3)
        value = value.view(*new_shape).permute(0, 2, 1, 3)

        output = self._scale_dot_product_attention(query, key, value, pos, mask).permute(0, 2, 1, 3).contiguous()
        new_shape = output.size()[:2] + (self.head_num * self.hidden_dim, )
        output = output.view(*new_shape)

        return self.output_layer(output)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=128):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divide_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * math.log(10000.0) / dim)
        divide_term_ = torch.exp(torch.arange(0, dim - 1, 2, dtype=torch.float) * math.log(10000.0) / dim)

        self.position_encoding = torch.zeros(max_len, dim)
        self.position_encoding[:, 0::2] = torch.sin(position * divide_term)
        self.position_encoding[:, 1::2] = torch.cos(position * divide_term_)

        self.position_encoding.require_grad = False
        self.position_encoding = nn.Parameter(self.position_encoding)

    def forward(self, position):
        return self.position_encoding[position, :]


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, p):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        hidden_layer = self.dropout(self.activation(self.w_1(x)))
        return self.w_2(hidden_layer)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, input_dim, head_num, hidden_dim, p=0, attention='MultiHeadAttention'):
        super().__init__()

        self.att_dict = dict(
            MultiHeadAttention=MultiHeadAttention,
            TimeAwareMultiHeadAttention=TimeAwareMultiHeadAttention
        )

        self.attention = self.att_dict[attention](input_dim=input_dim, head_num=head_num, p=p)
        self.attention_norm = nn.LayerNorm(input_dim)

        self.feed_forward = PositionWiseFeedForward(input_dim=input_dim, hidden_dim=hidden_dim, p=p)
        self.feed_forward_norm = nn.LayerNorm(input_dim)

        self.dropout_1 = nn.Dropout(p)
        self.dropout_2 = nn.Dropout(p)

    def forward(self, x, p, mask):

        attention_x = self.dropout_1(self.attention(x, x, x, p, mask))
        x = self.attention_norm(x + attention_x)

        feed_forward_x = self.dropout_2(self.feed_forward(x))
        x = self.feed_forward_norm(x + feed_forward_x)

        return x
