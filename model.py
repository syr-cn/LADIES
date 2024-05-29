from utils import *

class GCNConv(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GCNConv, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x, adj):
        out = self.linear(x)
        return self.act(torch.spmm(adj, out))

class GINConv(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        raise NotImplementedError
        super(GINConv, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, adj):
        out = torch.spmm(adj, x)
        # out = (1 + self.eps) * x + out
        out = self.linear(out)
        return F.relu(out)

class GNN(nn.Module):
    def __init__(self, gnn_type, nfeat, nhid, layers, dropout):
        super(GNN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        GNNConv = {
            'gcn': GCNConv,
            'gin': GINConv
        }[gnn_type]
        self.gcs.append(GNNConv(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GNNConv(nhid,  nhid))
    def forward(self, x, adjs):
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x

class GNNCls(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(GNNCls, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x
