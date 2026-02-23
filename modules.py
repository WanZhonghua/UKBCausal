import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import preprocess_adj_new
from utils import preprocess_adj_new1


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs, rel_rec, rel_send):
        adj_A1 = torch.sinh(3.*self.adj_A)
        adj_Aforz = preprocess_adj_new(adj_A1)
        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa
        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDecoder(nn.Module):
    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,do_prob=0.):
        super(MLPDecoder, self).__init__()
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.batch_size = batch_size
        self.data_variable_size = data_variable_size
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)
        return mat_z, out, adj_A_tilt


