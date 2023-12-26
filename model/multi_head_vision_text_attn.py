import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class TextConfiguration:
    def __init__(self):
        self.HIDDEN_SIZE = 768  # Hidden size for both image and text features
        self.MULTI_HEAD = 8  # Number of attention heads
        self.HIDDEN_SIZE_HEAD = self.HIDDEN_SIZE // self.MULTI_HEAD  # Hidden size per head
        self.FF_SIZE = 768  # Feed-forward network size
        self.DROPOUT_R = 0.1  # Dropout rate
        self.LAYER = 5  # Number of layers in the stacked architecture


class ImageConfiguration:
    def __init__(self):
        self.HIDDEN_SIZE = 768  # Hidden size for both image and text features
        self.MULTI_HEAD = 8  # Number of attention heads
        self.HIDDEN_SIZE_HEAD = self.HIDDEN_SIZE // self.MULTI_HEAD  # Hidden size per head
        self.FF_SIZE = 768  # Feed-forward network size
        self.DROPOUT_R = 0.1  # Dropout rate
        self.LAYER = 5  # Number of layers in the stacked architecture

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# Define MLP class for the Feed Forward Nets (FFN)
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r, use_relu=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, out_size)
        self.dropout = nn.Dropout(dropout_r)
        self.use_relu = use_relu

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
    

class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(x)

class SGA(nn.Module):
    def __init__(self, __IMG_C, _TEXT_C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__IMG_C)
        self.mhatt2 = MHAtt(_TEXT_C)
        self.ffn = FFN(_TEXT_C)

        self.dropout1 = nn.Dropout(_TEXT_C.DROPOUT_R)
        self.norm1 = LayerNorm(__IMG_C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__IMG_C.DROPOUT_R)
        self.norm2 = LayerNorm(_TEXT_C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(_TEXT_C.DROPOUT_R)
        self.norm3 = LayerNorm(_TEXT_C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
    
if __name__ == "__main__":

    __IMAGE_C = ImageConfiguration()
    _TEXT_C = TextConfiguration()

    sga_modules = nn.ModuleList([SGA(__IMAGE_C, _TEXT_C) for _ in range(1)])
    print(sga_modules)