import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.Embedding import XSEembedding, XTEembedding
from model.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from model.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
# import ipdb


class SpatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    num_heads:        number of attention heads
    dim_heads:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, num_heads, dim_heads):
        super(SpatialAttention, self).__init__()
        D = num_heads * dim_heads
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.linear_Q = nn.Linear(D, D)
        self.linear_K = nn.Linear(D, D)
        self.linear_V = nn.Linear(D, D)
        self.linear_output1 = nn.Linear(D, D)
        self.linear_output2 = nn.Linear(D, D)
        self.activation = nn.ReLU()

    def forward(self, xse):
        batch_size = xse.shape[0]

        # 对Q,K,V进行线性变换
        Q = self.linear_Q(xse)
        K = self.linear_K(xse)
        V = self.linear_V(xse)
        # D = num_heads * dim_heads
        # 对Q,K,V进行多头注意力机制的切分 [batch_size, num_step, num_vertex, num_heads * dim_heads]
        # -> [num_heads * batch_size, num_step, num_vertex, dim_heads]
        Q = torch.cat(torch.split(Q, self.num_heads, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.num_heads, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.num_heads, dim=-1), dim=0)
        
        # 计算attention score
        # [num_heads * batch_size, num_step, num_vertex, num_vertex]
        attention_score = torch.matmul(Q, K.permute(0,1,3,2)) / (self.dim_heads ** 0.5)
        attention_score = F.softmax(attention_score, dim=-1)
        
        # 计算attention output
        attention_output = torch.matmul(attention_score, V)

        # 合并多头注意力机制的输出 -> [batch_size, num_step, num_vertex, D]
        res = torch.cat(torch.split(attention_output, batch_size, dim=0), dim=-1)

        # 输出层
        res = self.linear_output1(res)
        res = self.activation(res)
        res = self.linear_output2(res)
        return res, attention_score


class GatedFusion(nn.Module):
    '''
    gated fusion
    H_spatial:      [batch_size, num_step, num_vertex, D]
    H_temporal:     [batch_size, num_step, num_vertex, D]
    return:         [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.mlp_Xs = nn.Linear(D, D, bias=False)
        self.mlp_Xt = nn.Linear(D, D)
        self.mlp_output1 = nn.Linear(D, 2*D)
        self.activations = nn.ReLU()
        self.mlp_output2 = nn.Linear(2*D, D)

    def forward(self, H_spatial, H_temporal):
        Xs = self.mlp_Xs(H_spatial)
        Xt = self.mlp_Xt(H_temporal)
        gate = torch.sigmoid(torch.add(Xs, Xt))
        output = torch.add(torch.mul(gate, H_spatial), torch.mul(1-gate, H_temporal))
        output = self.mlp_output1(output)
        output = self.activations(output)
        output = self.mlp_output2(output)
        
        return output
    

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化用于压缩空间维度，仅保留通道信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 两层全连接网络，用于学习每个通道的权重
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        
        # 使用 sigmoid 激活函数，将权重限制在 0 到 1 的范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origin_x = x
        
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        
        # 将通道通过 sigmoid 函数
        out = self.sigmoid(avg_out)
        
        # 与原始输入相乘以获得增强后的特征图
        return x * out + origin_x
    

class MultiScaleAttentiveFusion(nn.Module):
    def __init__(self, D):
        super(MultiScaleAttentiveFusion, self).__init__()
        self.conv3 = nn.Conv2d(D, D, kernel_size=(1, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(D, D, kernel_size=(1, 5), stride=(1, 1))
        self.conv7 = nn.Conv2d(D, D, kernel_size=(1, 7), stride=(1, 1))

        self.CA3 = ChannelAttention(D)
        self.CA5 = ChannelAttention(D)
        self.CA7 = ChannelAttention(D)

        self.conv3_1 = nn.Conv2d(10, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv5_1 = nn.Conv2d(8, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv7_1 = nn.Conv2d(6, 12, kernel_size=(1, 1), stride=(1, 1))

        # mlp fusion
        self.fc = nn.Sequential(
            nn.Conv2d(6 * D, D, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(D, D, kernel_size=1, bias=False)
        )

    def forward(self, x):
        origin_x = x
        x = x.permute(0, 3, 2, 1)

        x3 = self.conv3(x)
        x3 = self.CA3(x3)
        x3 = self.conv3_1(x3.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_3 = torch.cat((x, x3), dim=1)

        x5 = self.conv5(x + x3)
        x5 = self.CA5(x5)
        x5 = self.conv5_1(x5.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_5 = torch.cat((x, x5), dim=1)

        x7 = self.conv7(x + x5)
        x7 = self.CA7(x7)
        x7 = self.conv7_1(x7.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_7 = torch.cat((x, x7), dim=1)

        concat = torch.cat((x_3, x_5, x_7), dim=1)
        concat = self.fc(concat)
        concat = concat.permute(0, 3, 2, 1)
        
        output = concat + origin_x
        
        return output


class STMACN(nn.Module):
    '''
        INPUT:
            X:      [batch_size, num_his, num_vertex]
            TE:     [batch_size, num_his + num_pred, 2](time-of-day, day-of-week)
            SE:     [num_vertex, num_heads * dim_heads]
            configs:
                - num_heads
                - dim_heads
                - num_his
                - factor=3
                - dropout=0.1
                - d_ff=128
                - moving_avg=16
                - encoder_layers=1
                - decoder_layers=1
        OUTPUT:
            Y_hat:  [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, SE, configs):
        super(STMACN, self).__init__()
        # Embedding
        D = configs['num_heads'] * configs['dim_heads']
        self.num_his = configs['num_his']
        self.SE = SE
        self.xte_embedding = XTEembedding(D, configs['T'])
        self.xse_embedding = XSEembedding(D)
        
        # linear layer
        self.input_linear1 = nn.Linear(1, D)
        self.input_linear2 = nn.Linear(D, D)
        self.output_linear1 = nn.Linear(D, D)
        self.output_linear2 = nn.Linear(D,1)
        self.activation = nn.ReLU()

        # Decomp
        kernel_size = configs['moving_avg']
        self.decomp = series_decomp(kernel_size)

        # Spatial Attention
        self.spatial_attention = SpatialAttention(configs['num_heads'], configs['dim_heads'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        d_model=D, 
                        n_heads=configs['num_heads']
                    ),
                    d_model=D,
                    d_ff=configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                ) for l in range(configs['encoder_layers'])
            ],
            norm_layer=my_Layernorm(D)
        )

        # Gated Fusion
        self.gated1 = GatedFusion(D)
        self.gated2 = GatedFusion(D)
        self.msf = MultiScaleAttentiveFusion(D)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        D,
                        configs['num_heads']
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            configs['factor'],
                            attention_dropout=configs['dropout'],
                        ),
                        D,
                        configs['num_heads']
                    ),
                    d_model=D,
                    d_ff=configs['d_ff'],
                    moving_avg=configs['moving_avg'],
                    dropout=configs['dropout'],
                )
                for l in range(configs['decoder_layers'])
            ],
            norm_layer=my_Layernorm(D),
        )

    
    def forward(self, X, TE, return_attention=False):
        # X -> [batch_size, num_his, num_vertex, D]
        X = X.unsqueeze(-1) # (16, 12, 307)
        X = self.input_linear1(X)
        X = self.activation(X)
        X = self.input_linear2(X)

        # Embedding
        xse = self.xse_embedding(X, self.SE)
        xte = self.xte_embedding(X, TE)
        xte_his  = xte[:, :self.num_his, ...]
        xte_pred = xte[:, self.num_his:, ...]

        # Encoder
        enc_out1, att_score = self.spatial_attention(xse)
        enc_out2 = self.encoder(xte_his)

        # ipdb.set_trace()
        # gated fusion
        enc_out = self.gated1(enc_out1, enc_out2) # torch.Size([16, 12, 307, 64])
        enc_out = self.msf(enc_out)
        
        # Decoder
        dec_out1, dec_out2 = self.decoder(xte_his, enc_out, xte_pred)

        dec_out = self.gated2(dec_out1, dec_out2)
        out = self.output_linear1(dec_out)
        out = self.activation(out)
        out = self.output_linear2(out)

        # [B,L,V,1] -> [B,L,V]
        out = out.squeeze(-1)

        if return_attention:
            return out, att_score
        else:
            return out