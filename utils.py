
import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


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