from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size=10):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
        self.activation = nn.Sigmoid()

    def forward(self, u, v):
        U = self.user_emb(u).cuda()
        V = self.item_emb(v).cuda()
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        out = (U * V).sum(1) + b_u + b_v
        return self.activation(out)
