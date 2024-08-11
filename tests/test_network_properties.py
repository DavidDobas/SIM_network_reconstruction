from src.network_properties import annd, clustering_coeff, anns, weighted_clustering_coeff
import pytest
import igraph as ig
import numpy as np

test_graph = ig.Graph(n=5, edges=[[0,0], [0,1], [0,4], [1,2], [1,4], [2,1], [2,3], [2,4], [4,3]], directed=True)
exp_annd_out = [3/2,2,1,np.nan,0]
exp_annd_in = [np.nan, 1/2, 2, 2, 1]
exp_annd_all = [7/2, 3, 3, 7/2, 5/2]
exp_annd_k_out = [0, 7/4, 1]
exp_annd_k_in = [2, 5/4, 1]
exp_annd_k_all = [np.nan, 7/2, 3, 5/2]

# @pytest.mark.parametrize(['mode', 'exp_annd', 'exp_annd_k'],
#                 [('out', exp_annd_out, exp_annd_k_out),
#                  ('in', exp_annd_in, exp_annd_k_in),
#                  ('all', exp_annd_all, exp_annd_k_all),]
# )
# def test_annd(mode, exp_annd, exp_annd_k):
#     annd_measured, annd_k = annd(test_graph, mode)
#     assert np.array_equal(annd_measured, exp_annd, equal_nan=True)
#     assert np.array_equal(annd_k, exp_annd_k, equal_nan=True)

@pytest.mark.parametrize(['mode', 'exp_annd'],
                [('out', exp_annd_out),
                 ('in', exp_annd_in),
                 ('all', exp_annd_all),]
)
def test_annd(mode, exp_annd):
    annd_measured = annd(test_graph, mode)
    assert np.array_equal(annd_measured, exp_annd, equal_nan=True)

# test_graph_2 = ig.Graph(n=6, edges=[[0,0], [0,1], [1,2], [1,3], [2,3], [2,4], [3,4], [3,5]]) 
# exp_clustering_coeff_array = [np.nan, 1/3, 2/3, 2/6, 1, np.nan]
# exp_clustering_coeff_k = [np.nan, 1, 1/2, 1/3]
# def test_clustering_coeff():
#     clustering_coeff_array, clustering_coeff_k = clustering_coeff(test_graph_2)
#     assert np.array_equal( clustering_coeff_array, exp_clustering_coeff_array, equal_nan=True)
#     assert np.array_equal(clustering_coeff_k, exp_clustering_coeff_k, equal_nan=True)

test_graph_3 = ig.Graph(n=4, edges = [[1,1], [1,2], [1,3], [3,2]], directed=True)
test_graph_3.es["weight"] = [5, 4, 2, 3]
exp_anns_out = [np.nan, 3/2, np.nan, 0]
exp_anns_in = [np.nan, np.nan, 1, 0]
@pytest.mark.parametrize(['mode', 'exp_anns'],
                [('out', exp_anns_out),
                 ('in', exp_anns_in)]
)
def test_anns(mode, exp_anns):
    anns_measured = anns(test_graph_3, mode)
    assert np.array_equal(anns_measured, exp_anns, equal_nan=True)

test_graph_2 = ig.Graph(n=6, edges=[[0,0], [0,1], [1,2], [1,3], [2,3], [2,4], [3,4], [3,5]]) 
exp_clustering_coeff_array = [np.nan, 1/3, 2/3, 2/6, 1, np.nan]
def test_clustering_coeff():
    clustering_coeff_array = clustering_coeff(test_graph_2)
    assert np.array_equal(clustering_coeff_array, exp_clustering_coeff_array, equal_nan=True)

exp_weighted_cl_coeff_graph_3 = [np.nan, np.nan, np.nan, 0]
test_graph_4 = ig.Graph(n=6, edges=[[0,1], [1,2], [1,3], [1,4], [2,0], [2,3], [3,1], [5,1]], directed=True)
test_graph_4.es["weight"] = [8, 8, 2, 1, 8, 1, 1, 1]
exp_weighted_cl_coeff_graph_4 = [8, 10/8, 5, 2, np.nan, np.nan] 
def test_weighted_cl_coeff():
    weighted_cl_coeff_graph_3_measured = weighted_clustering_coeff(test_graph_3)
    assert np.array_equal(weighted_cl_coeff_graph_3_measured, exp_weighted_cl_coeff_graph_3, equal_nan=True)
    weighted_cl_coeff_graph_4_measured = weighted_clustering_coeff(test_graph_4)
    assert np.allclose(weighted_cl_coeff_graph_4_measured, exp_weighted_cl_coeff_graph_4, equal_nan=True)