
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from multilayer_Q import multi_Q

import scipy.io as io


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_input, n_z):
        super(GAE, self).__init__()
        self.gcn_1 = GNNLayer(n_input, n_enc_1)
        self.gcn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gcn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gcn_4 = GNNLayer(n_enc_3, n_z)

    def forward(self, x, adj):
        enc_h1 = F.relu(self.gcn_1(x, adj))
        enc_h2 = F.relu(self.gcn_2(enc_h1, adj))
        enc_h3 = F.relu(self.gcn_3(enc_h2, adj))
        enc_h4 = F.relu(self.gcn_4(enc_h3, adj))
        h = F.normalize(enc_h4, p=2, dim=1)
        A_hat = torch.sigmoid(torch.mm(h, h.t()))
        return enc_h1, enc_h2, enc_h3, enc_h4, h, A_hat


class FC(nn.Module):

    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.enc = Linear(in_features, out_features)

    def forward(self, x, active=True):
        output = F.relu(self.enc(x))
        return output


class GCFM(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_input, n_z, n_clusters, v=1):
          super(GCFM, self).__init__()

          self.gae_1 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input,
            n_z=n_z)

          self.gae_2 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input,
            n_z=n_z)

          self.gae_3 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input,
            n_z=n_z)

          self.gae_4 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input,
            n_z=n_z)

          self.fc_1 = FC(n_input, n_enc_1)
          self.fc_2 = FC(n_enc_1, n_enc_2)
          self.fc_3 = FC(n_enc_2, n_enc_3)
          self.fc_4 = FC(n_enc_3, n_z)

          # cluster layer
          self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
          torch.nn.init.xavier_normal_(self.cluster_layer.data)

          # degree
          self.v = v

    def forward(self, X1, X2, X3, X4, adj1, adj2, adj3, adj4):
        H1_1, H1_2, H1_3, H1_4, H_1, A_hat1 = self.gae_1(X1, adj1)
        H2_1, H2_2, H2_3, H2_4, H_2, A_hat2 = self.gae_2(X2, adj2)
        H3_1, H3_2, H3_3, H3_4, H_3, A_hat3 = self.gae_3(X3, adj3)
        H4_1, H4_2, H4_3, H4_4, H_4, A_hat4 = self.gae_4(X4, adj4)
        
        # Z_1 = self.fc_1(    X1  + X2  +X3  +X4)
        Z_2 = self.fc_2( H1_1 + H2_1 + H3_1 + H4_1)
        Z_3 = self.fc_3(Z_2 + H1_2 + H2_2 + H3_2 + H4_2)
        Z  = self.fc_4(Z_3 + H1_3 + H2_3 + H3_3 + H4_3)
        Z  = Z + H1_4+H2_4+H3_4+H4_4

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1)-self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q, Z, A_hat1, A_hat2, A_hat3, A_hat4


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_GCFM(dataset_1, dataset_2, dataset_3, dataset_4):
    model = GCFM(512, 1024, 2048,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # load graph
    adj_1,adj_m1,adj_label_1 = load_graph(args.name, args.k, 1)
    adj_2,adj_m2,adj_label_2 = load_graph(args.name, args.k, 2)
    adj_3,adj_m3,adj_label_3 = load_graph(args.name, args.k, 3)
    adj_4,adj_m4,adj_label_4 = load_graph(args.name, args.k, 4)

    adj_1 = adj_1.to(device)
    adj_2 = adj_2.to(device)
    adj_3 = adj_3.to(device)
    adj_4 = adj_4.to(device)
    adj_label_1 = adj_label_1.to(device)
    adj_label_2 = adj_label_2.to(device)
    adj_label_3 = adj_label_3.to(device)
    adj_label_4 = adj_label_4.to(device)

    adj_m1 = torch.from_numpy(adj_m1)
    adj_m2 = torch.from_numpy(adj_m2)
    adj_m3 = torch.from_numpy(adj_m3)
    adj_m4 = torch.from_numpy(adj_m4)
    adj_m1 = adj_m1.float()
    adj_m2 = adj_m2.float()
    adj_m3 = adj_m3.float()
    adj_m4 = adj_m4.float()
    
    # load initial embedding
    data_1 = torch.Tensor(dataset_1.x).to(device)
    data_2 = torch.Tensor(dataset_2.x).to(device)
    data_3 = torch.Tensor(dataset_3.x).to(device)
    data_4 = torch.Tensor(dataset_4.x).to(device)

    y = dataset_1.y

    with torch.no_grad():
      _, z, _, _, _, _ = model(data_1, data_2, data_3, data_4, adj_1, adj_2, adj_3, adj_4)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    kmeans.fit_predict(z.data.to(device).numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    # strat training
    model.train()
    for epoch in range(50): 
        tmp_q, Z, A_hat_1, A_hat_2, A_hat_3, A_hat_4= model(data_1, data_2, data_3, data_4, adj_1, adj_2, adj_3, adj_4)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)
        
        # reconstruction loss
        re_loss1 = F.binary_cross_entropy(A_hat_1.view(-1), adj_label_1.to_dense().view(-1))
        re_loss2 = F.binary_cross_entropy(A_hat_2.view(-1), adj_label_2.to_dense().view(-1))
        re_loss3 = F.binary_cross_entropy(A_hat_3.view(-1), adj_label_3.to_dense().view(-1))
        re_loss4 = F.binary_cross_entropy(A_hat_4.view(-1), adj_label_4.to_dense().view(-1))
        # kl loss
        kl_loss = F.kl_div(tmp_q.log(), p, reduction='batchmean')  
        
        y_pred=tmp_q.cpu().numpy().argmax(1)
        
        loss = re_loss1 + re_loss2 + re_loss3 + re_loss4 +  0.01 * kl_loss

        print('Epoch{}: loss:{}'.format(epoch, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
      q, _, _, _, _, _ = model(data_1, data_2, data_3, data_4, adj_1, adj_2, adj_3, adj_4)
    y_pred = q.cpu().numpy().argmax(1)
    
    # NMI, ARI, Purity
    eva(y, y_pred, None, 'last')
    
    # Modularity
    array_dict={
      "layer1":adj_m1,
      "layer2":adj_m2,
      "layer3":adj_m3,
      "layer4":adj_m4
    }
    cluster_dict={
      "layer1":list(y_pred),
      "layer2":list(y_pred),
      "layer3":list(y_pred),
      "layer4":list(y_pred)
    }
    print('multi_Q',multi_Q(array_dict, cluster_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='mLFR200_mu0.35')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    
    dataset_1 = load_data(args.name, 1)   
    dataset_2 = load_data(args.name, 2)
    dataset_3 = load_data(args.name, 3)
    dataset_4 = load_data(args.name, 4)
      
    if args.name == 'mLFR200_mu0.35':
      args.k=None
      args.n_clusters = 2
      args.n_input =128
    setup_seed(321)
    train_GCFM(dataset_1, dataset_2, dataset_3, dataset_4)