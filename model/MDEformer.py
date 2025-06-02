import torch.nn as nn
import torch
import math
import torch.nn.init as init
from torchinfo import summary
import numpy as np
import torch.nn.functional as F

class NodePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        """
        :param embed_dim: 位置编码的维度,也就是每个时间步的特征数
        :param max_len: 最大序列长度
        """
        super(NodePositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() *
                    -(math.log(10000.0) / embed_dim)).exp()
        # 对于偶数维度使用正弦，对于奇数维度使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加一个批次维度，作为 buffer 来存储该位置编码
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入的张量，形状为 (16, 12, 170, 120)
        :return: 添加了位置编码的张量
        """
        batch_size, seq_len, num_nodes, embed_dim = x.size()
        pos_encoding = self.pe[:, :seq_len, :]  # [1, seq_len, embed_dim]
        pos_encoding = pos_encoding.unsqueeze(2).expand(batch_size, seq_len, num_nodes, embed_dim)
        return x + pos_encoding

class GraphLearn(nn.Module):

    def __init__(self, num_nodes, init_feature_num):
        super(GraphLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = nn.Parameter(torch.rand(num_nodes, dtype=torch.float32), requires_grad=True)
        self.w1 = nn.Parameter(torch.zeros((num_nodes, init_feature_num), dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros((num_nodes, init_feature_num), dtype=torch.float32), requires_grad=True)
        self.attn = nn.Conv2d(2, 1, kernel_size=1)
        # Initialize weights
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, adj_mx):
        device = self.w1.device  # 获取模型参数的设备
        adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(device)
        # adj_mx = adj_mx.to(device)
        # -------------------------- 从这里开始修改-----------------------------
        # new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj_mx = torch.matmul(self.w1, self.w2.T) - torch.matmul(self.w2, self.w1.T)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))

        attn_input = torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)
        attn = torch.sigmoid(self.attn(attn_input).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1).pow(-0.5)
        new_adj_mx = (d.view(-1, 1) * new_adj_mx * d) - self.epsilon
        new_adj_mx = torch.relu(new_adj_mx.clone())
        # Final normalization step
        d = new_adj_mx.sum(dim=1).pow(-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx

class ChebConv(nn.Module):  # 定义图卷积层的类

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize  # 正则化参数,True or False
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c] ,第二个1是维度扩张，计算方便,有没有都不影响参数的大小,nn.Parameter就是把参数转换成模型可改动的参数.
        # 之所以要k+1,是因为k是从0开始的
        init.xavier_normal_(self.weight)  # 用正态分布填充
        if bias:  # 偏置,就是一次函数中的b
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))  # 前面的两个1是为了计算简单，因为输出的维度是3维
            init.zeros_(self.bias)  # 用0填充
        else:
            self.register_parameter("bias", None)
        # self.act = nn.ReLU()
        self.K = K + 1

    def forward(self, inputs, graph):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # graph = torch.tensor(graph, dtype=torch.float32, device='cuda:0')
        graph = graph.clone().detach().to(device='cuda:0', dtype=torch.float32)
        L = ChebConv.get_laplacian(graph, self.normalize)  # 得到拉普拉斯矩阵，形状为[N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]，这个就是多阶的切比雪夫多项式，K就是阶数，N是节点数量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mul_L = mul_L.to(device)  # (16,12,170,152)
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        num_nodes = inputs.shape[2]  # 170
        model_dim = inputs.shape[3]  # 152
        inputs = inputs.reshape(batch_size * seq_length, num_nodes, model_dim).to(device)
        # inputs = inputs.to(device)
        # 计算切比雪夫多项式 (mul_L) 和输入特征 (inputs) 的乘积
        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]，这个就是计算完后乘x
        # 这一步相当于对每个节点特征做了线性变换
        result = torch.matmul(result, self.weight)  # [K, B, N, D]，计算上一步之后乘W
        # 可尝试添加ReLU()
        # result = self.act(result)
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]，求和
        return result

    def cheb_polynomial(self, laplacian):  # 计算切比雪夫多项式,也就是前面公式中的 T_k(L)

        N = laplacian.size(0)  # [N, N] ,节点个数
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N],初始化一个全0的多项式拉普拉斯矩阵
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)  # 将第0阶的切比雪夫多项式设置为单位矩阵
        if self.K == 1:  # 当 (K=1) 时，只需要返回第0阶的切比雪夫多项式，即单位矩阵
            return multi_order_laplacian
        else:  # 大于等于1阶
            multi_order_laplacian[1] = laplacian
            if self.K == 2:  # 1阶切比雪夫多项式就是拉普拉斯矩阵L本身
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - \
                                               multi_order_laplacian[k - 2]  # 切比雪夫多项式的递推式:T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):  # 计算拉普拉斯矩阵

        if normalize:  # 如果 normalize 为真，则计算归一化拉普拉斯矩阵
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))  # 这里的graph就是邻接矩阵,这个D为度矩阵
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph),
                                                                                            D)  # L = I - D * A * D,这个也就是正则化
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L  # 返回计算得到的拉普拉斯矩阵 (L)

class RH(nn.Module):
    def __init__(self, input_size, hidden_size, k1=2, k2=2, embed_dim=24):
        super(RH, self).__init__()
        self.GL = GraphLearn(num_nodes=170, init_feature_num=input_size)
        self.sta = ChebConv(in_c=input_size, out_c=hidden_size, K=k1)
        self.act_re = nn.ReLU()
        # self.node_embeddings = nn.Parameter(torch.randn(170, 24), requires_grad=True)
        # self.act_sigmo = nn.Sigmoid()

    def forward(self, x):
        # x(16,12,170,120)
        adj_matrix_path = 'pems08_01_adj.npy'
        A = np.load(adj_matrix_path)  # (num_nodes, num_nodes)
        # A = torch.tensor(A, dtype=torch.float32, device='cuda')
        graph = self.GL(A)
        # graph = torch.from_numpy(A).float()
        output1 = self.sta(x, graph)
        output3 = self.act_re(output1)
        return output3

class Spa_AttentionLayer1(nn.Module):  # s
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.DSGFG_Q = RH(input_size=model_dim, hidden_size=model_dim)
        self.DSGFG_K = RH(input_size=model_dim, hidden_size=model_dim)
        self.DSGFG_V = RH(input_size=model_dim, hidden_size=model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q、K、V（16，12，170，152）
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        num_nodes = query.shape[2]  # 170
        model_dim = query.shape[3]  # 120
        query = self.DSGFG_Q(query)
        key = self.DSGFG_K(key)
        value = self.DSGFG_V(value)
        query = query.reshape(batch_size , seq_length, num_nodes, model_dim)
        key = key.reshape(batch_size , seq_length, num_nodes, model_dim)
        value = value.reshape(batch_size , seq_length, num_nodes, model_dim)
        query = query.transpose(1, 2)  # (16,12,170,120)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)
        attn_score = (query @ key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class Spa_SelfAttentionLayer1(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.attn = Spa_AttentionLayer1(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)  # (16, 170, 12, 152)
        out = self.dropout1(out)
        out = out.permute(0, 2, 1, 3)  # 变为 (16, 170, 12, 152)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out

class LearnableRelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len):
        super().__init__()
        # 可学习的相对位置编码（基于头数和最大序列长度）
        self.relative_positions = nn.Parameter(torch.randn(num_heads, max_len, max_len))

    def forward(self, seq_len, device):
        # 获取和当前序列长度相匹配的相对位置编码
        return self.relative_positions[:, :seq_len, :seq_len].to(device)
class Tem_AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.GRU_Q = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.GRU_K = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.GRU_V = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.out_proj = nn.Linear(model_dim, model_dim)
        # self.relative_position_bias = nn.Parameter(torch.randn(num_heads, 12, 12))
        self.relative_position_bias = LearnableRelativePositionBias(num_heads, 12)

    def forward(self, query, key, value):
        # Q、K、V(16,170,12,152)
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        num_nodes = query.shape[2]  # 170
        model_dim = query.shape[3]  # 120
        query = query.reshape(batch_size * num_nodes, seq_length, model_dim)
        key = key.reshape(batch_size * num_nodes, seq_length, model_dim)
        value = value.reshape(batch_size * num_nodes, seq_length, model_dim)
        query, _ = self.GRU_Q(query)
        key, _ = self.GRU_K(key)
        value, _ = self.GRU_V(value)
        query = query.reshape(batch_size, num_nodes, seq_length, model_dim)
        key = key.reshape(batch_size, num_nodes, seq_length, model_dim)
        value = value.reshape(batch_size, num_nodes, seq_length, model_dim)
        query = query.transpose(1, 2)  # (16,12,170,152)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)
        num_nodes = key.shape[1]
        attn_score = (query @ key) / self.head_dim ** 0.5
        # relative_bias = self.relative_position_bias.expand(batch_size, num_nodes, -1, tgt_length, src_length)
        # relative_bias = relative_bias.contiguous().view(batch_size * self.num_heads, num_nodes, tgt_length, src_length)
        # attn_score = attn_score + relative_bias
        # 获取可学习的相对位置编码
        relative_bias = self.relative_position_bias(tgt_length, device=query.device)

        # Ensure relative_bias has the shape [num_heads, tgt_length, src_length]
        relative_bias = relative_bias.unsqueeze(0)  # Shape: [1, num_heads, tgt_length, src_length]

        # Expand relative_bias to match batch_size * num_heads, num_nodes, tgt_length, src_length
        relative_bias = relative_bias.expand(batch_size, self.num_heads, tgt_length,
                                             src_length)  # Shape: [batch_size, num_heads, tgt_length, src_length]

        # Now, expand relative_bias to match attn_score shape: [batch_size * num_heads, num_nodes, tgt_length, src_length]
        relative_bias = relative_bias.unsqueeze(1)  # Shape: [batch_size, 1, num_heads, tgt_length, src_length]
        relative_bias = relative_bias.expand(batch_size, num_nodes, self.num_heads, tgt_length,
                                             src_length)  # Shape: [batch_size, num_nodes, num_heads, tgt_length, src_length]

        # Finally, reshape relative_bias to match attn_score's shape
        relative_bias = relative_bias.contiguous().view(batch_size * self.num_heads, num_nodes, tgt_length,
                                                        src_length)  # Shape: [batch_size * num_heads, num_nodes, tgt_length, src_length]

        # Add relative_bias to attn_score
        attn_score = attn_score + relative_bias

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = self.out_proj(out)
        return out


class Tem_SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attn = Tem_AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)  # 开始的（16，12，170，152）  ----》  x: (16,170,12,152)
        residual = x  # (16,170,12,152)
        out = self.attn(x, x, x)
        out = self.dropout1(out)  # (16,12,170,152)
        # out = out.permute(0, 2, 1, 3)  # 变为 (16, 170, 12, 152)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)  # (16,170,12,152)
        out = out.transpose(dim, -2)  # (16,12,170,152)
        return out

class MDEformer(nn.Module):
    def __init__(
            self, num_nodes, in_steps=12, out_steps=12, steps_per_day=288, input_dim=5, output_dim=1,
            input_embedding_dim=24, hol_embedding_dim=24,
            tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0, temporal_embedding_dim = 24,
            feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.hol_embedding_dim = hol_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.temporal_embedding_dim = temporal_embedding_dim
        self.model_dim = (input_embedding_dim + tod_embedding_dim + dow_embedding_dim + hol_embedding_dim + spatial_embedding_dim + temporal_embedding_dim)
        self.num_heads = num_heads
        self.num_layers = num_layers  # 3
        self.use_mixed_proj = use_mixed_proj
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if hol_embedding_dim > 0:
            self.hol_embedding = nn.Embedding(4, hol_embedding_dim)
        if temporal_embedding_dim > 0:
            self.node_emb1 = nn.Parameter(torch.empty(self.in_steps, self.temporal_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb1)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if use_mixed_proj:  # True
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        # ---------------------------- 时间transformer --------------------------------
        self.attn_layers_t = nn.ModuleList(
            [
                Tem_SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)  # num_layers = 3
            ]
        )
        # --------------------------- 空间transformer -------------------------------
        self.attn_layers_s = nn.ModuleList(
            [
                Spa_SelfAttentionLayer1(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding = NodePositionalEncoding(self.model_dim, 12)
        self.conv = nn.Conv2d(
            in_channels=288,  # 输入通道数
            out_channels=144,  # 输出通道数
            kernel_size=1,  # 1x1 卷积核
            stride=1,  # 步长为 1
            padding=0  # 不使用 padding，确保没有额外的填充
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0] # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.tod_embedding_dim > 0:  # 24
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.hol_embedding_dim > 0:
            hol = x[..., 3]

        x = x[..., : self.input_dim] # 这里的input_dim = 4
        x = self.input_proj(x)  # self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.hol_embedding_dim > 0:
            hol_emb = self.hol_embedding(hol.long())
            features.append(hol_emb)
        if self.temporal_embedding_dim > 0:
            temporal_emb = self.node_emb1.expand(batch_size, self.num_nodes, *self.node_emb1.shape)
            temporal_emb = temporal_emb.permute(0, 2, 1, 3)
            features.append(temporal_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            features.append(spatial_emb)
        x = torch.cat(features, dim=-1)  # (16, 12, 170, model_dim)
        x = self.positional_encoding(x)

        x_time = x
        x_space = x
        for attn in self.attn_layers_t:
            x_time = attn(x_time, dim=1)
        for attn in self.attn_layers_s:
            x_space = attn(x_space, dim=2)

        x_combined = torch.cat([x_time, x_space], dim=-1)  # 拼接之后的形状为 (16, 12, 170, 240)
        x_combined = x_combined.permute(0, 3, 1, 2)  # (16, 240, 12, 170)
        x = self.conv(x_combined)  # 输出形状为 (16, 170, 12, 120)
        x = x.permute(0, 2, 3, 1)
        # x = self.drop(x)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (batch_size, out_steps, num_nodes, output_dim)
        return out


if __name__ == "__main__":
    model = MDEformer(170, 12, 12)
    summary(model, [16, 12, 170, 120])

