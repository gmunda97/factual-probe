'''
Module to train a GNN to generate embeddings from a graph
'''

import json
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split


def create_graph(file_path: str) -> nx.DiGraph:
    '''
    Create a NetworkX graph from a JSONL file
    '''
    graph = nx.Graph()

    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:               
            triple = json.loads(line)

            subject = triple['sub_label']
            object_ = triple['obj_label']
            predicate = triple['predicate_label']

            graph.add_node(subject)
            graph.add_node(object_)
            graph.add_edge(subject, object_, label=predicate)

    return graph

def graph_to_pyg_data(graph: nx.DiGraph) -> Data:
    '''
    Convert the NetworkX graph to PyTorch Geometric data
    '''
    node_to_index = {node: index for index, node in enumerate(graph.nodes)}
    edge_index = torch.tensor([[node_to_index[edge[0]], 
                                node_to_index[edge[1]]] for edge in graph.edges]).t().contiguous()

    x = torch.randn(graph.number_of_nodes(),100)  # Use random node features for illustration
    data = Data(x=x, edge_index=edge_index)

    return data


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

if __name__ == '__main__':
    JSONL_FILE_PATH = './data/dataset/TREx_all.jsonl'
    graph = create_graph(JSONL_FILE_PATH)
    data = graph_to_pyg_data(graph)
    print(data)
    print(f"Number of Nodes: {graph.number_of_nodes()}")
    print(f"Number of Edges: {graph.number_of_edges()}")

    model = GraphSAGEModel(in_channels=data.num_node_features, hidden_channels=600, out_channels=100) # 50 works very well

    loader = DataLoader([data], batch_size=1, shuffle=True)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_function(out, data.x)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        print(f"Shape of the embeddings: {embeddings.shape}")
        print(embeddings)
