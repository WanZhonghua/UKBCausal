
import os
import time
import math
import json
import torch
import pickle
import argparse
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

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

def load_data(args, batch_size=1000):
    X = np.load(args.data_dir)
    print(X.shape)
    feat_train = torch.FloatTensor(X)
    train_data = TensorDataset(feat_train, feat_train)
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot


def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)


def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5

def A_connect_loss(A, tol, z):
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss +=  2 * tol - torch.sum(torch.abs(A[:,i])) - torch.sum(torch.abs(A[i,:])) + z * z
    return loss

def A_positive_loss(A, z_positive):
    result = - A + z_positive * z_positive
    loss =  torch.sum(result)
    return loss


def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized


def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default= '/home/wzh/UKB-SJTU/Unified-Dimension/X_node_tensor.npy',help='data file name containing the discrete files.')
parser.add_argument('--data_variable_size', type=int, default=45,help='the number of variables in synthetic generated data')
parser.add_argument('--x_dims', type=int, default=1,help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,help='The number of latent variable dimensions: default the same as variable size.')
parser.add_argument('--optimizer', type = str, default = 'Adam',help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=0.0,help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 0.,help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 1,help='coefficient for absolute value h(A).')
parser.add_argument('--use_A_connect_loss',  type = int, default= 0,help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default = 0,help = 'flag to enforce A must have positive values')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default= 300,help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default = 100, help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=3e-3,help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,help='Temperature for Gumbel softmax.')
parser.add_argument('--k_max_iter', type = int, default = 1e2,help ='the max iteration number for searching lambda and c')
parser.add_argument('--no-factor', action='store_true', default=False,help='Disables factor graph model.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',help='Where to load the trained model if finetunning. ' +'Leave empty to train from scratch')
parser.add_argument('--h_tol', type=float, default = 1e-8,help='the tolerance of error of h(A) to zero')
parser.add_argument('--lr-decay', type=int, default=200,help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,help='LR decay factor.')
parser.add_argument('--var', type=float, default=5e-5,help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,help='Uses discrete samples in training forward pass.')


args = parser.parse_args()
args.factor = not args.no_factor
print(args)


torch.manual_seed(args.seed)

exp_counter = 0
now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
encoder_file = os.path.join(save_folder, 'encoder.pt')
decoder_file = os.path.join(save_folder, 'decoder.pt')
log_file = os.path.join(save_folder, 'log.txt')
log = open(log_file, 'w')
pickle.dump({'args': args}, open(meta_file, "wb"))
args_txt_file = os.path.join(save_folder, "args.txt")
with open(args_txt_file, "w") as f:
    f.write("Training arguments:\n")
    f.write("=" * 50 + "\n")
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")


train_loader = load_data(args, args.batch_size)


off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)


rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

num_nodes = args.data_variable_size
adj_A = np.zeros((num_nodes, num_nodes))

encoder = MLPEncoder(args.data_variable_size * args.x_dims, 
                     args.x_dims, args.encoder_hidden,
                     int(args.z_dims), 
                     adj_A,
                     batch_size = args.batch_size,
                     do_prob = args.encoder_dropout, 
                     factor = args.factor).double()

decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                     args.z_dims, args.x_dims, encoder,
                     data_variable_size = args.data_variable_size,
                     batch_size = args.batch_size,
                     n_hid=args.decoder_hidden,
                     do_prob=args.decoder_dropout).double()


if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,gamma=args.gamma)

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)

    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    encoder.train()
    decoder.train()
    scheduler.step()
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)
    for batch_idx, (data, relations) in enumerate(train_loader):
        data, relations = Variable(data).double(), Variable(relations).double()
        relations = relations.unsqueeze(2)
        optimizer.zero_grad()
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data,rel_rec, rel_send)
        edges = logits
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)
        target = data
        preds = output
        variance = 0.
        loss_nll = nll_gaussian(preds, target, variance)
        loss_kl = kl_gaussian_sem(logits)
        loss = loss_kl + loss_nll
        one_adj_A = origin_A
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))


        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap
        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 1.0 * torch.trace(origin_A*origin_A) + sparse_loss
        loss.backward()
        loss = optimizer.step()
        myA.data = stau(myA.data, args.tau_A*lr)
        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    print(f"h(a):{h_A.item()}")
    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'time: {:.4f}s'.format(time.time() - t))
    
    ELBO_loss = np.mean(kl_train) + np.mean(nll_train)

    if args.save_folder and ELBO_loss < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()


    return ELBO_loss, np.mean(nll_train), np.mean(mse_train), origin_A

t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []
c_A = args.c_A
lambda_A = args.lambda_A
h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

for step_k in range(k_max_iter):
    while c_A < 1e+20:
        for epoch in range(args.epochs):
            ELBO_loss, NLL_loss, MSE_loss, origin_A = train(epoch, best_ELBO_loss, lambda_A, c_A, optimizer)
            if ELBO_loss < best_ELBO_loss:
                best_ELBO_loss = ELBO_loss
                best_epoch = epoch
            if NLL_loss < best_NLL_loss:
                best_NLL_loss = NLL_loss
                best_epoch = epoch
            if MSE_loss < best_MSE_loss:
                best_MSE_loss = MSE_loss
                best_epoch = epoch

        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if ELBO_loss > 2 * best_ELBO_loss:
            break

        A_new = origin_A.data.clone()
        h_A_new = _h_A(A_new, args.data_variable_size)
        if h_A_new.item() > 0.25 * h_A_old:
            c_A*=2
        else:
            break
    h_A_old = h_A_new.item()
    lambda_A += c_A * h_A_new.item()

    if h_A_new.item() <= h_tol:
        break

if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

A = origin_A.detach().cpu().numpy()
predG_npy = os.path.join(save_folder, "predG.npy")
np.save(predG_npy, A)

print("Matrix shape:", A.shape) 


with open("label.json", "r") as f:
    label_dict = json.load(f)

labels = [None] * len(label_dict)
for name, idx in label_dict.items():
    labels[idx] = name

plt.figure(figsize=(16, 14))

ax = sns.heatmap(A,cmap="RdBu_r",center=0,square=True,cbar=True,xticklabels=labels,yticklabels=labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

ax.axhline(27, color="black", linewidth=0.5)
ax.axvline(27, color="black", linewidth=0.5)
ax.axhline(34, color="red", linewidth=0.5)
ax.axvline(34, color="red", linewidth=0.5)

plt.title("Learned Adjacency Matrix (DAG-GNN)", fontsize=16)
plt.xlabel("Source Node", fontsize=12)
plt.ylabel("Target Node", fontsize=12)
plt.tight_layout()
plt.savefig(f"{save_folder}/adjacency_heatmap.png", dpi=300)

if log is not None:
    print(save_folder)
    log.close()
