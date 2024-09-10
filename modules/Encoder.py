import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules.utils import clones, LayerNorm
import math


# UniSurv ============================================================================================================
class UniSurv(nn.Module):

    def __init__(self, T_max, T_period, in_channel, layer, N, d_model, dropout, num_features, num_cnn_features):
        super(UniSurv, self).__init__()
        self.max_time = T_max
        self.T_peroid = T_period
        self.cnn_number = self.max_time // self.T_peroid
        self.fc = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU())
        # self.parallelcnn = ParallelCNN(self.cnn_number, in_channel)
        self.singlecnn = SingleCNN(self.cnn_number, in_channel)
        self.trans_encoder = Encoder(layer, N, d_model, dropout, num_features+num_cnn_features)  # 20

    def forward(self, feature, images):
        feature_img = self.singlecnn(images.to(torch.float32))
        # feature_img = self.parallelcnn(images.to(torch.float32))  # a list (len = 16) of tensor (B*L = 32*20)
        feature_img_copy = [i for i in feature_img for _ in range(self.T_peroid)]

        feature_invar = self.fc(feature.to(torch.float32))
        feature_invar_copy = self.max_time * [feature_invar]

        merged_feature = torch.stack(
            [torch.cat((feature_img_copy[i], feature_invar_copy[i]), axis=1) for i in range(self.max_time)], dim=1)

        output = self.trans_encoder(merged_feature)
        return output


# CNN ==================================================================================================================
class ParallelCNN(nn.Module):
    def __init__(self, cnn_number, in_channel):
        super(ParallelCNN, self).__init__()
        self.in_channel = in_channel
        self.cnn_num = cnn_number
        self.cnn_models = nn.ModuleList()
        for _ in range(self.cnn_num):
            cnn_model = nn.Sequential(nn.Conv2d(self.in_channel, 1, kernel_size=(3, 10), stride=(1, 5), padding=0),
                                      nn.ReLU(), nn.Flatten())
            self.cnn_models.append(cnn_model)

    def forward(self, img):
        outputs = []
        for i, cnn_model in enumerate(self.cnn_models):
            output = cnn_model(img[:, i, :, :, :])  # img: B*T*C*H*W
            outputs.append(output)
        return outputs


# ----------------------------------------------------------------------------------------------------------------------
class SingleCNN(nn.Module):
    def __init__(self, cnn_number, in_channel):
        super(SingleCNN, self).__init__()
        self.in_channel = in_channel
        self.cnn_num = cnn_number
        self.cnn = nn.Sequential(nn.Conv2d(self.in_channel, 1, kernel_size=(3, 10), stride=(1, 5), padding=0),
                                 nn.ReLU(), nn.Flatten())

    def forward(self, img):
        outputs = []
        for i in range(self.cnn_num):
            output = self.cnn(img[:, i, :, :, :])
            outputs.append(output)
        return outputs


# Transformer ==========================================================================================================
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)

    def forward(self, x, mask=None):
        """Pass the input (and mask) through each layer in turn."""
        x = x.to(torch.float32)
        x = self.position_encode(self.src_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)


# initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TranFinalLayer(nn.Module):  # final layer for the transformer
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model // 2)
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.norm(x)
        x = self.w_2(x)
        # return torch.sigmoid(x.squeeze(-1))
        # return F.softmax(x.squeeze(-1), dim=-1)
        return x
# ======================================================================================================================
