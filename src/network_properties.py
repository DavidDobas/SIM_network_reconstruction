import numpy as np
from numba import jit

@jit(nopython=True)
def anns_matrices(adj_matrix, w_adj_matrix, mode):
    if mode=="out":
        # multiplied = np.matmul(adj_matrix, w_adj_matrix)
        multiplied = adj_matrix @ w_adj_matrix
        result = np.sum(multiplied, axis=1)/np.sum(adj_matrix, axis=1)
    if mode=="in":
        # multiplied = np.matmul(w_adj_matrix, adj_matrix)
        multiplied = w_adj_matrix @ adj_matrix
        result = np.sum(multiplied, axis=0)/np.sum(adj_matrix, axis=0)
    return result
    

def anns(graph, mode):
    adj_matrix = np.array(graph.get_adjacency().data).astype(np.float64)
    np.fill_diagonal(adj_matrix, 0)
    w_adj_matrix = np.array(graph.get_adjacency(attribute='weight').data).astype(np.float64)
    np.fill_diagonal(w_adj_matrix, 0)
    result = anns_matrices(adj_matrix, w_adj_matrix, mode)
    return result

def annd(graph, mode="all", len_deg_seq=None):
    if mode=="all":
        new_graph = graph.as_undirected()
    else:
        new_graph = graph
    annd_array = np.zeros(len(new_graph.vs))
    for i in range(len(new_graph.vs)):
        vertex = new_graph.vs[i]
        neighbors = vertex.neighbors(mode=mode)
        annd_array[i] = np.nanmean([neighbor.degree(mode=mode, loops=False) for neighbor in neighbors if neighbor != vertex])
    return annd_array

# ANNS in a similar manner as ANND
def anns_new(graph, mode="all"):
    if mode=="all":
        new_graph = graph.as_undirected()
    else:
        new_graph = graph
    anns_array = np.zeros(len(new_graph.vs))
    for i in range(len(new_graph.vs)):
        vertex = new_graph.vs[i]
        neighbors = vertex.neighbors(mode=mode)
        anns_array[i] = np.mean([neighbor.strength(mode=mode, loops=False) for neighbor in neighbors if neighbor != vertex])
    return anns_array


def clustering_coeff_old(graph):
    new_graph = graph.as_undirected()
    adj_matrix = np.array(new_graph.get_adjacency().data)
    np.fill_diagonal(adj_matrix, 0)
    multiplied = np.matmul(np.matmul(adj_matrix, adj_matrix), adj_matrix)
    degrees = np.sum(adj_matrix, axis=1)
    denominator = degrees*(degrees-1)
    result = np.diag(multiplied)/denominator
    return result

def clustering_coeff(graph):
    result = graph.transitivity_local_undirected()
    return result

def weighted_clustering_coeff(graph):
    w_cl_coeff_array = np.zeros(graph.vcount())
    for i in range(graph.vcount()):
        vertex = graph.vs[i]
        neighbors_out = vertex.neighbors(mode="out")
        neighbors_in = vertex.neighbors(mode="in")
        num_wedges = 0
        for neighbor_out in neighbors_out:
            if neighbor_out == vertex:
                continue
            for neighbor_in in neighbors_in:
                if neighbor_out != neighbor_in and neighbor_in != vertex:
                    num_wedges += 1
                    if graph.are_connected(neighbor_out, neighbor_in):
                        w_ij = graph.es[graph.get_eid(vertex, neighbor_out)]["weight"]
                        w_jk = graph.es[graph.get_eid(neighbor_out, neighbor_in)]["weight"]
                        w_ki = graph.es[graph.get_eid(neighbor_in, vertex)]["weight"]
                        w_cl_coeff_array[i] += (w_ij*w_jk*w_ki)**(1/3)
        w_cl_coeff_array[i] = w_cl_coeff_array[i]/num_wedges
    return w_cl_coeff_array