import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv 
from torch_geometric.utils import degree
import args 
from efficient_kan import KAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class TGAE(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, dropout_rate=0.5):
        super(TGAE, self).__init__()

        self.encoder_gcn1 = GCNConv(num_features, hidden_dim)
        self.encoder_gcn2 = GCNConv(hidden_dim, hidden_dim) 
        self.encoder_kan = KAN([hidden_dim, latent_dim]) 
        self.dropout = nn.Dropout(dropout_rate)



        self.feature_decoder_gcn = GCNConv(latent_dim, hidden_dim)
        self.feature_decoder_kan = KAN([hidden_dim, num_features]) 

      
        self.edge_decoder_kan = KAN([latent_dim, hidden_dim, 1])

       
        self.degree_decoder_kan1 = KAN([latent_dim, hidden_dim])
        self.degree_decoder_kan2 = KAN([hidden_dim, 1]) 


        self.log_sigma_feature = nn.Parameter(torch.zeros(1, device=device))
        self.log_sigma_edge = nn.Parameter(torch.zeros(1, device=device))
        self.log_sigma_degree = nn.Parameter(torch.zeros(1, device=device))


    def encode(self, x, edge_index):
       
        h1 = self.encoder_gcn1(x, edge_index)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.encoder_gcn2(h1, edge_index)
        h2 = F.relu(h2) 
        h2 = self.dropout(h2)
        
        z = self.encoder_kan(h2)
        return z

    def decode_features(self, z, edge_index):
        h = self.feature_decoder_gcn(z, edge_index)
        h = F.relu(h) 
        h = self.dropout(h)
        
        reconstructed_x = self.feature_decoder_kan(h)
        return reconstructed_x

    def decode_edges(self, z, edge_index_to_predict_for):
        source_nodes_z = z[edge_index_to_predict_for[0]] 
        target_nodes_z = z[edge_index_to_predict_for[1]] 
        
        edge_features = source_nodes_z * target_nodes_z
        
        edge_logits = self.edge_decoder_kan(edge_features).squeeze(-1) 
        return torch.sigmoid(edge_logits) 

    def decode_degree(self, z):
        h = self.degree_decoder_kan1(z)
        h = F.relu(h)
        
        degree_pred = self.degree_decoder_kan2(h)
        degree_pred = F.relu(degree_pred) 
        return degree_pred.squeeze(-1) 

    def forward(self, x, edge_index, edge_index_for_edge_prediction):
        z = self.encode(x, edge_index)
        
        reconstructed_x = self.decode_features(z, edge_index)
        edge_probabilities = self.decode_edges(z, edge_index_for_edge_prediction)
        predicted_degrees = self.decode_degree(z)
        
        return reconstructed_x, edge_probabilities, predicted_degrees, z

        
