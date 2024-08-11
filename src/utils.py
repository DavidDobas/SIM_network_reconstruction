import numpy as np
from src import renormalizable_model
import scipy
import igraph as ig

def weighted_adj_matrix(graph):
    return np.array(graph.get_adjacency(attribute='weight').data)

def adj_matrix(graph):
    return np.array(graph.get_adjacency().data)

def graph_from_adjacency(adj_matrix, weighted=True):
    if weighted: return ig.Graph.Weighted_Adjacency(adj_matrix)
    return ig.Graph.Adjacency(adj_matrix)

def _exp_num_of_links(z, strengths, num_of_links, self_loops=True):
    if self_loops:
        new_strengths = strengths
    else:
        new_strengths = renormalizable_model.convert_strengths_without_self_loops(strengths)
    out_strengths = new_strengths[:,0]
    in_strengths = new_strengths[:,1]
    num_nodes = len(out_strengths)
    prob_matrix = np.ones([num_nodes, num_nodes]) - np.exp(-z*out_strengths.reshape(num_nodes,1)@in_strengths.reshape(1,num_nodes))
    if not self_loops:
        prob_matrix -= np.diag(np.diag(prob_matrix))
    return np.sum(prob_matrix) - num_of_links

def estimate_z(strengths, num_of_links, self_loops=True, print_details=False):
    exp_z_details = scipy.optimize.root_scalar(lambda z: _exp_num_of_links(z, strengths, num_of_links, self_loops), method="bisect", bracket=[0,100])
    if print_details: print(exp_z_details)
    return exp_z_details.root

def compute_strengths(graph):
    out_strengths = np.sum(np.array(graph.get_adjacency(attribute='weight').data), axis=1)
    in_strengths = np.sum(np.array(graph.get_adjacency(attribute='weight').data), axis=0)
    return np.stack([out_strengths, in_strengths], axis=1)

def compute_num_edges(graph):
    return graph.ecount()

def edge_list(graph):
    return np.array(graph.get_edgelist())