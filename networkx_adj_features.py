import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import scipy.sparse as sp

def generate_roberta_features(node_contents_list, batch_size=32, device='cpu'):
    """
    Generates RoBERTa embeddings for a list of textual contents.

    Args:
        node_contents_list (list): A list of strings, where each string is the content of a node.
        batch_size (int): Batch size for processing.
        device (str): 'cuda' or 'cpu'.

    Returns:
        np.ndarray: A 2D numpy array of embeddings.
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()

    all_embeddings = []
    print(f"Generating RoBERTa features for {len(node_contents_list)} unique node contents...")
    for i in tqdm(range(0, len(node_contents_list), batch_size), desc="Generating RoBERTa features"):
        batch_contents = node_contents_list[i:i + batch_size]
        inputs = tokenizer(batch_contents, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(batch_embeddings)

    if not all_embeddings:
        return np.array([])
    return np.vstack(all_embeddings)


def construct_contextual_tabular_graph(tables_data, pfk_relations, semantic_sim_threshold=0.85, device='cpu'):
    G = nx.Graph()
    node_contents_map = {} 
    
    def add_node_for_embedding(node_id, content_str):
        if node_id not in G:
            G.add_node(node_id, content=content_str)
            node_contents_map[node_id] = content_str

    column_map = {}
    global_col_idx = 1
    
    cell_nodes_set = set() 
    row_nodes_set = set()
    column_nodes_set = set()
    table_nodes_set = set()

    print("Processing tables to define nodes...")
    for table_name, df in tables_data.items():
        table_node_id = f"TABLE__{table_name}"
        table_caption = getattr(df, 'caption', table_name)
        add_node_for_embedding(table_node_id, table_caption)
        table_nodes_set.add(table_node_id)

        for col_name in df.columns:
            if (table_name, col_name) not in column_map:
                col_node_id = f"COL__C{global_col_idx}"
                column_map[(table_name, col_name)] = col_node_id
                add_node_for_embedding(col_node_id, str(col_name))
                column_nodes_set.add(col_node_id)
                global_col_idx += 1

        for r_idx, row in df.iterrows():
            row_node_id = f"ROW__{table_name}_R{r_idx}"
            add_node_for_embedding(row_node_id, f"Row {r_idx} of {table_name}")
            row_nodes_set.add(row_node_id)
            
            for col_name, cell_value in row.items():
                cell_content = str(cell_value)
                cell_node_id = f"CELL__{cell_content}"
                add_node_for_embedding(cell_node_id, cell_content)
                cell_nodes_set.add(cell_node_id)

    nodes_ordered_list = list(node_contents_map.keys()) 
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes_ordered_list)}

    node_features_dict = {}
    if nodes_ordered_list: 
        unique_contents_ordered = [node_contents_map[nid] for nid in nodes_ordered_list]
        roberta_features_array = generate_roberta_features(unique_contents_ordered, device=device)
        
        if roberta_features_array.size > 0:
            for i, node_id in enumerate(nodes_ordered_list):
                node_features_dict[node_id] = roberta_features_array[i]
                G.nodes[node_id]['feature_vector'] = roberta_features_array[i] 
        else:
            print("Warning: RoBERTa feature generation resulted in an empty array.")

            default_feature_dim = 768 
            print(f"Warning: RoBERTa feature generation failed or produced no features. Using zero vectors of dim {default_feature_dim}.")
            roberta_features_array = np.zeros((len(nodes_ordered_list), default_feature_dim))
            for i, node_id in enumerate(nodes_ordered_list):
                node_features_dict[node_id] = roberta_features_array[i]
                G.nodes[node_id]['feature_vector'] = roberta_features_array[i]


    else:
        print("Warning: No nodes found in the graph. Skipping feature generation.")
        roberta_features_array = np.array([]) 

    print("Defining edge sets...")
    for table_name, df in tables_data.items():
        for r_idx, row in df.iterrows():
            row_node_id = f"ROW__{table_name}_R{r_idx}"
            for col_name, cell_value in row.items():
                cell_node_id = f"CELL__{str(cell_value)}"
                col_node_id = column_map[(table_name, col_name)]
                if cell_node_id in node_to_idx and row_node_id in node_to_idx:
                     G.add_edge(cell_node_id, row_node_id, type='ECR')
                if cell_node_id in node_to_idx and col_node_id in node_to_idx:
                     G.add_edge(cell_node_id, col_node_id, type='ECA')

    for table_name, df in tables_data.items():
        table_node_id = f"TABLE__{table_name}"
        for col_name in df.columns:
            col_node_id = column_map[(table_name, col_name)]
            if col_node_id in node_to_idx and table_node_id in node_to_idx:
                G.add_edge(col_node_id, table_node_id, type='EAT')

    for src_table, src_col, trg_table, trg_col in pfk_relations:
        src_col_node_id = column_map.get((src_table, src_col))
        trg_col_node_id = column_map.get((trg_table, trg_col))
        if src_col_node_id and trg_col_node_id and \
           src_col_node_id in node_to_idx and trg_col_node_id in node_to_idx:
            G.add_edge(src_col_node_id, trg_col_node_id, type='PFK_SCHEMA')
        else:
            print(f"Warning: PFK column nodes missing or not in main node list for {(src_table, src_col)} or {(trg_table, trg_col)}")

    if cell_nodes_set and node_features_dict:

        semantic_cell_node_ids_with_features = [
            nid for nid in cell_nodes_set if nid in node_features_dict and nid in node_to_idx
        ]
        
        if len(semantic_cell_node_ids_with_features) > 1:
            cell_embeddings_list = [node_features_dict[nid] for nid in semantic_cell_node_ids_with_features]
            cell_embeddings_matrix = np.array(cell_embeddings_list)

            if cell_embeddings_matrix.ndim == 1:
                 cell_embeddings_matrix = cell_embeddings_matrix.reshape(-1, 1)
            
            if cell_embeddings_matrix.size > 0 : 
                similarity_matrix = cosine_similarity(cell_embeddings_matrix)
                
                print(f"Calculating semantic similarity for {len(semantic_cell_node_ids_with_features)} cell nodes...")
                for i, j in tqdm(itertools.combinations(range(len(semantic_cell_node_ids_with_features)), 2), 
                                 total=len(list(itertools.combinations(range(len(semantic_cell_node_ids_with_features)), 2)))):
                    sim = similarity_matrix[i, j]
                    if sim > semantic_sim_threshold:
                        node1_id = semantic_cell_node_ids_with_features[i]
                        node2_id = semantic_cell_node_ids_with_features[j]
                        G.add_edge(node1_id, node2_id, type='ESS_CELL', weight=sim)
            else:
                print("Warning: Cell embeddings matrix is empty, skipping semantic similarity.")
    else:
        print("Skipping semantic similarity for cells: no cell nodes or features available.")
        
    print(f"Graph construction complete. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    if nodes_ordered_list:
        adj_matrix = nx.adjacency_matrix(G, nodelist=nodes_ordered_list)
        
        if roberta_features_array.size > 0:
            if roberta_features_array.shape[0] == len(nodes_ordered_list):
                 features_sparse = sp.csr_matrix(roberta_features_array)
            else:
                print("Error: Mismatch in feature array length and node list length.")
                default_feature_dim = 768
                features_sparse = sp.csr_matrix(np.zeros((len(nodes_ordered_list), default_feature_dim)))
        else: 
            print("No features generated, creating a sparse zero feature matrix.")
            default_feature_dim = 768 
            features_sparse = sp.csr_matrix((len(nodes_ordered_list), default_feature_dim))
    else: 
        adj_matrix = sp.csr_matrix((0, 0))
        features_sparse = sp.csr_matrix((0, 0))

    return adj_matrix, features_sparse, nodes_ordered_list, G

