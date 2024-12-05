#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import random
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

import einops


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SCA(nn.Module):
    def __init__(self, in_planes, kerenel_size, ratio=1):
        super(SCA, self).__init__()
        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, kerenel_size, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, kerenel_size, padding='same', bias=False),
        )

    def forward(self, x):
        # 输入形状：(batch_size, channels, sample_points)
        x = self.sharedMLP(x)
        return x


class EfficientAdditiveAttention(nn.Module):

    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out


class Spatio_Block(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatio_Block, self).__init__()
        self.relu = nn.ReLU()
        # # 定义多个网络组件
        ##########################################
        self.net_a = weight_norm(nn.Conv1d(n_inputs, 128, 1))
        self.net_b = weight_norm(nn.Conv1d(128, 128, 1))  #
        self.net_c = weight_norm(nn.Linear(n_channels, 128))
        self.net_d = weight_norm(nn.Linear(128, 128))
        self.net_e = weight_norm(nn.Conv1d(128, n_inputs, 1))
        self.net_f = weight_norm(nn.Linear(128, n_channels))
        # 定义顺序网络
        ###########################################
        self.net = nn.Sequential(self.net_a,
                                 # self.net_b,
                                 self.net_c,
                                 # self.net_d,
                                 self.net_e,
                                 self.net_f,
                                 self.relu, )
        self.init_weights()

    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)
        self.net_e.weight.data.normal_(0, 0.01)
        self.net_f.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.net(x)
        return y


class Spatio_ConvNet(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatio_ConvNet, self).__init__()
        self.relu = nn.ReLU()

        self.cpca_block = CPCABlock(in_channels=n_inputs, out_channels=n_inputs, channelAttention_reduce=4)

        # 引入 SCA 模块
        self.sca = SCA(in_planes=n_channels, kerenel_size=3, ratio=2)

        ##########################################
        self.net_a = weight_norm(nn.Conv1d(n_inputs, 128, 1))
        self.net_b = weight_norm(nn.Conv1d(128, 128, 1))  #
        self.net_c = weight_norm(nn.Linear(n_channels, 128))
        self.net_d = weight_norm(nn.Linear(128, 128))  #
        self.pool1 = nn.MaxPool2d(2, 2)
        ###########################################
        self.net_e = weight_norm(nn.Conv1d(64, 64, 1))  #
        self.net_f = weight_norm(nn.Conv1d(64, 32, 1))
        self.net_g = weight_norm(nn.Linear(64, 64))  #
        self.net_h = weight_norm(nn.Linear(64, 32))
        self.pool2 = nn.MaxPool2d(2, 2)
        ###########################################
        self.net_i = weight_norm(nn.Conv1d(16, 16, 1))  #
        self.net_j = weight_norm(nn.Conv1d(16, 4, 1))
        self.net_k = weight_norm(nn.Linear(16, 16))  #
        self.net_l = weight_norm(nn.Linear(16, 4))

        self.net = nn.Sequential(self.net_a, self.net_b, self.net_c, self.net_d, self.relu, self.pool1,
                                 self.net_e, self.net_f, self.net_g, self.net_h, self.relu, self.pool2,
                                 self.net_i, self.net_j, self.net_k, self.net_l, )

        self.fc = weight_norm(nn.Linear(16, 1))
        self.init_weights()

    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)

        self.net_e.weight.data.normal_(0, 0.01)
        self.net_f.weight.data.normal_(0, 0.01)
        self.net_g.weight.data.normal_(0, 0.01)
        self.net_h.weight.data.normal_(0, 0.01)

        self.net_i.weight.data.normal_(0, 0.01)
        self.net_j.weight.data.normal_(0, 0.01)
        self.net_k.weight.data.normal_(0, 0.01)
        self.net_l.weight.data.normal_(0, 0.01)
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 输入形状：(batch_size, channels, sample_points)
        # print("输入形状:", x.shape)

        # 引入 SCA 模块
        x = self.sca(x)
        # print("经过 SCA 后的形状:", x.shape)

        # 经过网络的其余部分
        x = self.net(x).squeeze(2)
        # print("经过网络后的形状:", x.shape)

        # 全连接层
        y = self.fc(x.view(-1, 16))
        # print("全连接层后的形状:", y.shape)

        y = self.relu(y)
        return y


class Temporal_and_Spatio_Block(nn.Module):
    def __init__(self, n_inputs, input_length, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(Temporal_and_Spatio_Block, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 自定义模块或函数，通常用于去除卷积产生的额外填充。
        self.relu1 = nn.ReLU()  # 非线性变换
        self.dropout1 = nn.Dropout(dropout)  # 和过拟合防控。
        # -------------------------------------------------------#
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.temporal_net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, )
        # self.conv2, self.chomp2,self.relu2, self.dropout2,)
        self.spatio_net = Spatio_Block(n_inputs, input_length)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.temporal_net(x)

        y2 = self.spatio_net(x)
        return self.relu(x + y1 + y2)


class Temporal_and_Spatio_ConvNet(nn.Module):
    def __init__(self, num_inputs, input_length, num_channels, kernel_size, dropout):
        super(Temporal_and_Spatio_ConvNet, self).__init__()
        # self.transformer = EfficientAdditiveAttention(in_dims=64, token_dim=32)
        layers = []  # 用于存储网络的层。
        num_levels = len(num_channels)  # 计算共有多少个卷积层。
        for i in range(num_levels):
            dilation_size = 2 ** i  # 使得每一层使用的卷积的膨胀率是先前层的两倍。
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 当前层的输入通道数
            out_channels = num_channels[i]  # 当前层的输出通道数
            layers += [Temporal_and_Spatio_Block(in_channels, input_length, out_channels, kernel_size, stride=1,
                                                 dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                                 dropout=dropout)]
        self.transformer = EfficientAdditiveAttention(in_dims=64, token_dim=64)
        self.temporal_and_spatio_network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.transformer(x)
        return self.temporal_and_spatio_network(x)


class LSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM_Block, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入形状：(batch_size, channels, sequence_length)
        # LSTM 要求输入形状：(batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # 转换形状以适应 LSTM 输入
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的隐藏状态
        out = self.fc(lstm_out[:, -1, :])
        return out


class TnS_net(nn.Module):
    # input_channel：模型接受的输入数据的通道数。
    # input_length：输入数据的时间序列长度。
    # output_channel：模型输出的通道数。
    # num_channels：定义各层卷积通道大小的列表。
    # kernel_size：卷积核的大小。
    # dropout：dropout 概率，用于防止过拟合。
    def __init__(self, input_channel, input_length, output_channel, num_channels, kernel_size, lstm_hidden_size,
                 lstm_num_layers, dropout):
        super(TnS_net, self).__init__()
        self.TnS = Temporal_and_Spatio_ConvNet(input_channel, input_length, num_channels, kernel_size, dropout)
        self.Spatio_net = Spatio_ConvNet(input_channel, input_length)
        self.lstm_block = LSTM_Block(input_channel, lstm_hidden_size, lstm_num_layers, output_channel, dropout)
        self.linear = nn.Linear(num_channels[-1], output_channel)

    def forward(self, x):
        y = self.TnS(x)  # input should have dimension (N, C, L)
        y1 = self.linear(y[:, :, -1])
        y2 = self.Spatio_net(x)
        # LSTM 模块
        y_lstm = self.lstm_block(x)
        return torch.sigmoid(y1 + y2 + y_lstm)
