import datetime
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
from preprocessing import sparse_to_tuple 
import args

from model import TEAGraphAutoencoder 

from networkx_adj_features import construct_contextual_tabular_graph

from torch_geometric.utils import degree, from_scipy_sparse_matrix, to_scipy_sparse_matrix
import networkx as nx
import pandas as pd

Time_Log = "Runtime_log.txt"
t_start = datetime.datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load and Preprocess Data using Contextual Tabular Graph ---
adj_sparse, features_roberta_sparse, nodes_ordered_list, G_networkx = construct_contextual_tabular_graph(
    tables_data=args.dataset, 
    pfk_relations=args.pfk_relations,
    semantic_sim_threshold=0.75, 
    device=device
)

if not nodes_ordered_list:
    raise ValueError("Graph construction resulted in no nodes. Exiting.")

features_dense_tensor = torch.FloatTensor(features_roberta_sparse.toarray()).to(device)
num_features = features_dense_tensor.shape[1]
num_nodes = features_dense_tensor.shape[0]

if G_networkx.number_of_nodes() != len(nodes_ordered_list):
    print("Warning: Node count mismatch between G_networkx and nodes_ordered_list. Reordering G_networkx.")
    edge_index, _ = from_scipy_sparse_matrix(adj_sparse)
    edge_index = edge_index.to(device)
else:
    adj_for_edge_index = nx.to_scipy_sparse_array(G_networkx, nodelist=nodes_ordered_list)
    edge_index, _ = from_scipy_sparse_matrix(adj_for_edge_index)
    edge_index = edge_index.to(device)

print(f"Number of nodes: {num_nodes}")
print(f"Number of features: {num_features}")
print(f"Edge index shape: {edge_index.shape}")

edge_index_for_prediction = edge_index 
positive_edge_labels = torch.ones(edge_index_for_prediction.shape[1]).to(device)

true_degrees = degree(edge_index[0], num_nodes=num_nodes).float().to(device)

model = TEAGraphAutoencoder(
    num_features=num_features,
    hidden_dim=args.hidden1_dim, 
    latent_dim=args.hidden2_dim, 
    dropout_rate=args.dropout 
).to(device)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
print("Model Initialized: TEAGraphAutoencoder")
print(model)

print("Starting training...")
for epoch in range(args.num_epoch):
    t_epoch_start = time.time()
    model.train()
    optimizer.zero_grad()

    reconstructed_x, edge_probabilities, predicted_degrees, node_embeddings_latent = model(
        x=features_dense_tensor,
        edge_index=edge_index,
        edge_index_for_edge_prediction=edge_index_for_prediction
    )
    
    loss_feat = F.mse_loss(reconstructed_x, features_dense_tensor)
    loss_edge = F.binary_cross_entropy(edge_probabilities, positive_edge_labels)
    loss_deg = F.mse_loss(predicted_degrees, true_degrees)

    sigma_feature_sq = torch.exp(model.log_sigma_feature)**2
    sigma_edge_sq = torch.exp(model.log_sigma_edge)**2
    sigma_degree_sq = torch.exp(model.log_sigma_degree)**2
    
    epsilon = 1e-8 

    term_feat = (1 / (2 * sigma_feature_sq + epsilon)) * loss_feat + model.log_sigma_feature
    term_edge = (1 / (2 * sigma_edge_sq + epsilon)) * loss_edge + model.log_sigma_edge
    term_deg = (1 / (2 * sigma_degree_sq + epsilon)) * loss_deg + model.log_sigma_degree
    
    total_loss = term_feat + term_edge + term_deg
    
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0 :
        print(f"Epoch: {epoch+1:04d}/{args.num_epoch:04d} | "
              f"Loss: {total_loss.item():.5f} | "
              f"L_feat: {loss_feat.item():.5f} | "
              f"L_edge: {loss_edge.item():.5f} | "
              f"L_deg: {loss_deg.item():.5f}")
        print(f"  sig_feat: {torch.exp(model.log_sigma_feature).item():.4f}, "
              f"sig_edge: {torch.exp(model.log_sigma_edge).item():.4f}, "
              f"sig_deg: {torch.exp(model.log_sigma_degree).item():.4f} | "
              f"Time: {time.time() - t_epoch_start:.2f}s")

print("Training finished.")

model.eval()
with torch.no_grad():
    _, _, _, final_node_embeddings = model(
        x=features_dense_tensor,
        edge_index=edge_index,
        edge_index_for_edge_prediction=edge_index_for_prediction 
    )
final_node_embeddings_np = final_node_embeddings.cpu().numpy()

run_tag_default = "tea_run"
if isinstance(args.dataset, str):
    run_tag = args.dataset.split(os.sep)[-1].split('-')[0] if os.sep in args.dataset else args.dataset.split('-')[0]
else:
    run_tag = run_tag_default

embeddings_dir = 'embeddings'
os.makedirs(embeddings_dir, exist_ok=True)
embeddings_path = os.path.join(embeddings_dir, f"{run_tag}.emb")

with open(embeddings_path, 'w') as file:
    if len(nodes_ordered_list) == final_node_embeddings_np.shape[0]:
        print(f"Saving {final_node_embeddings_np.shape[0]} embeddings with dimension {final_node_embeddings_np.shape[1]}")
        file.write(f"{final_node_embeddings_np.shape[0]} {final_node_embeddings_np.shape[1]}\n")
        for i in range(final_node_embeddings_np.shape[0]):
            node_id_str = str(nodes_ordered_list[i])
            embedding_values = ' '.join(map(str, final_node_embeddings_np[i]))
            file.write(f"{node_id_str} {embedding_values}\n")
    else:
        print("Error: Mismatch between number of nodes in list and embeddings generated.")
        print(f"Nodes in list: {len(nodes_ordered_list)}, Embeddings shape: {final_node_embeddings_np.shape}")

print(f"Embeddings saved to: {embeddings_path}")

t_end = datetime.datetime.now()
dt = t_end - t_start
os.makedirs(os.path.dirname(Time_Log), exist_ok=True)
with open(Time_Log, 'a') as t_file:
    t_file.write(f"{run_tag} Training Time required: {dt.total_seconds():.2f} s\n")

from testEQ import test
from entity_resolution import entity_resolution
from schema_matching import schema_matching

if args.test_type == 'EQ':
    print("Running Embedding Quality Test...")
    test(embeddings_path)
if args.test_type == 'ER':
    print("Running Entity Resolution Test...")
    print("ER test placeholder - ensure entity_resolution function and info_file are set up.")
if args.test_type == 'SM':
    print("Running Schema Matching Test...")
    print("SM test placeholder - ensure schema_matching function and info_file are set up.")