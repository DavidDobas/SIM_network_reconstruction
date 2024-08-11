import igraph as ig
import numpy as np
from tqdm import tqdm
import pickle
import h5py
import os
from functools import cached_property 
from src import network_properties, utils
from joblib import Parallel, delayed, cpu_count
from numba import jit
import pickletools

def assign_weight(x_i, y_j, p_ij, W):
    return x_i*y_j/(p_ij*W)

def check_weighted_consistency(strengths):
    x_i_array, y_i_array = zip(*strengths)
    if np.sum(x_i_array) != np.sum(y_i_array):
        raise Exception("Inconsistent weights")

def initialize_graph(strengths, weighted=True, check_consistency=False):
    if weighted and check_consistency:
        check_weighted_consistency(strengths)
    if weighted:
        W = np.sum(strengths[:,0])
    n = len(strengths)
    graph = ig.Graph(n, edges=[], directed=True)
    return graph, W, n

def convert_strengths_without_self_loops(strengths):
    out_strengths = strengths[:,0]
    in_strengths = strengths[:,1]
    S = np.sum(out_strengths)
    new_out_strengths = np.zeros(len(out_strengths))
    new_in_strengths = np.zeros(len(in_strengths))
    for i in range(len(out_strengths)):
        w_ii = out_strengths[i]*in_strengths[i]/S
        new_out_strengths[i] = out_strengths[i] + w_ii
        new_in_strengths[i] = in_strengths[i] + w_ii
    return np.stack([new_out_strengths, new_in_strengths], axis=1)    

def create_RM_graph(strengths, z, weighted=True, check_consistency=False, self_loops=True):
    if self_loops:
        new_strengths = strengths
    else:
        new_strengths = convert_strengths_without_self_loops(strengths)
    out_strengths = new_strengths[:,0]
    in_strengths = new_strengths[:,1]
    W = np.sum(out_strengths)
    num_nodes = len(out_strengths)
    prob_matrix = np.ones([num_nodes, num_nodes]) - np.exp(-z*out_strengths.reshape(num_nodes,1)@in_strengths.reshape(1,num_nodes))
    random_numbers = np.random.random_sample([num_nodes,num_nodes])
    if weighted:
        weight_matrix = out_strengths.reshape(num_nodes,1)@in_strengths.reshape(1,num_nodes)/(W*prob_matrix)
        weighted_adj_matrix = weight_matrix*(random_numbers<prob_matrix)
        return utils.graph_from_adjacency(weighted_adj_matrix, weighted=True)
    return utils.graph_from_adjacency(random_numbers<prob_matrix)

def create_naive_RM_graph(strengths, z, weighted=True, check_consistency = False):
    graph, W, n = initialize_graph(strengths, weighted, check_consistency)
    edges_to_add = []
    weights_to_add = []
    random_numbers = np.random.random_sample([n,n])
    for i in range(n):
        x_i = strengths[i][0]
        p_k_i_nonzero = 1-np.exp(-z*x_i*W)
        for j in range(n):
            y_j = strengths[j][1]
            p_ij = (1 - np.exp(-z*x_i*y_j))/p_k_i_nonzero if z < np.infty else 1
            if random_numbers[i][j] < p_ij:
                edges_to_add.append((i,j))
                if weighted:
                    weights_to_add.append(assign_weight(x_i, y_j, p_ij, W))
    graph.add_edges(edges_to_add)
    if weighted:
        graph.es["weight"] = weights_to_add
    return graph

@jit(nopython=True)
def corrected_naive_RM_graph_adj_matrix(num_nodes, prob_matrix):
    adj_matrix = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        num_edges_i = 0
        while num_edges_i == 0:
            random_numbers = np.random.random_sample(num_nodes)
            edges_to_add = random_numbers < prob_matrix[i,:]
            num_edges_i = np.sum(edges_to_add)
        adj_matrix[i,:] = edges_to_add
    return adj_matrix

@jit(nopython=True)
def corrected_naive_RM_graph_weighted_adj_matrix(strengths, z, W, prob_matrix, adjacency_matrix):
    weight_matrix = np.outer(strengths[:,0], strengths[:,1])/(W*prob_matrix)
    p_k_i_nonzero = 1-np.exp(-z*strengths[:,0]*W)
    p_k_i_nonzero_matrix = np.diag(p_k_i_nonzero)
    weighted_adj_matrix = p_k_i_nonzero_matrix@(weight_matrix*adjacency_matrix)
    return weighted_adj_matrix

def create_corrected_naive_RM_graph(strengths, z, weighted=False, check_consistency=False):
    new_strengths = strengths
    out_strengths = new_strengths[:,0]
    in_strengths = new_strengths[:,1]
    W = np.sum(out_strengths)
    num_nodes = int(len(out_strengths))
    prob_matrix = np.ones((num_nodes, num_nodes)) - np.exp(-z*np.outer(out_strengths, in_strengths))
    adjacency_matrix = corrected_naive_RM_graph_adj_matrix(num_nodes, prob_matrix)
    if weighted:
        weighted_adj_matrix = corrected_naive_RM_graph_weighted_adj_matrix(strengths, z, W, prob_matrix, adjacency_matrix)
        return utils.graph_from_adjacency(weighted_adj_matrix, weighted=True)
    return utils.graph_from_adjacency(adjacency_matrix, weighted=False)

def generate_RM_ensemble(method, n, strengths, z, weighted=True, parallel=True, threads=None):
    if parallel:
        return Parallel(n_jobs=cpu_count() if not threads else threads)(delayed(method)(strengths, z, weighted=weighted) for _ in tqdm(range(n)))
    return [method(strengths, z, weighted=weighted) for _ in tqdm(range(n))]

# Assure that sum of outcoming strengths is the same as sum of incoming strengths
def make_strengths_consistent(strengths):
    new_strengths = np.copy(strengths)
    x_i_arr, y_i_arr = zip(*new_strengths)
    x_sum = np.sum(x_i_arr)
    y_sum = np.sum(y_i_arr)
    new_strengths[-1,1] = new_strengths[-1,1] + x_sum - y_sum
    return new_strengths

def generate_RM_ensemble_new(method, n, strengths, z, name, weighted=True):
    ensemble = Ensemble(name)
    for _ in tqdm(range(n)):
        ensemble.append(method(strengths, z, weighted=weighted))
    return ensemble

# class Ensemble:
#     def __init__(self, filename: str, directory: str = "data/saved_ensembles"):
#         self.directory = directory
#         self.filename = filename
#         self.filepath = os.path.join(self.directory, self.filename + ".pkl")
#         self.current_index = 0
#         self._length = self._calculate_length()

#         # Create directory if it doesn't exist
#         os.makedirs(self.directory, exist_ok=True)

#     def save_graph(self, graph):
#         with open(self.filepath, 'ab') as f:
#             pickle.dump(graph, f)
#         self._length += 1

#     def _calculate_length(self):
#         if not os.path.exists(self.filepath):
#             return 0

#         count = 0
#         try:
#             with open(self.filepath, 'rb') as f:
#                 while True:
#                     pickle.load(f)
#                     count += 1
#         except EOFError:
#             return count
#         except IOError as e:
#             print(f"IOError: {e}")
#             return 0

#     def __getitem__(self, index):
#         if isinstance(index, slice):
#             start, stop, step = index.indices(self._length)
#             return [self[i] for i in range(start, stop, step)]
#         else:
#             try:
#                 with open(self.filepath, 'rb') as f:
#                     for _ in range(index + 1):
#                         graph = pickle.load(f)
#                     return graph
#             except EOFError:
#                 raise IndexError("Index out of range")
#             except IOError as e:
#                 print(f"IOError: {e}")
#                 raise IndexError("Unable to access file")

#     def __iter__(self):
#         return self
    
#     def __len__(self):
#         return self._length

#     def __next__(self):
#         try:
#             with open(self.filepath, 'rb') as f:
#                 f.seek(self.current_index)
#                 graph = pickle.load(f)
#                 self.current_index = f.tell()
#                 return graph
#         except EOFError:
#             self.current_index = 0
#             raise StopIteration
#         except IOError as e:
#             print(f"IOError: {e}")
#             raise StopIteration
        
#     def append(self, graph):
#         self.save_graph(graph)
#         self._length += 1 
    
#     def clear(self):
#         open(self.filepath, 'w').close()
#         self._length = 0
#         self._current_index = 0
    
######## 2ND version
# class Ensemble:
#     def __init__(self, filename: str, directory: str = "data/saved_ensembles", batch_size: int = 100):
#         self.directory = directory
#         self.filename = filename
#         self.filepath = os.path.join(self.directory, self.filename + ".pkl")
#         self.current_index = 0
#         # self._length = self._calculate_length()
#         self._length = None
#         self._batch = []
#         self.batch_size = batch_size
#         self._iteration_buffer = []
#         self._iteration_buffer_index = 0
#         self._file_handle = None
#         # Create directory if it doesn't exist
#         os.makedirs(self.directory, exist_ok=True)

#     def save_graph(self, graph):
#         with open(self.filepath, 'ab') as f:
#             pickle.dump(graph, f)
#         self._length += 1

#     def save_batch(self):
#         with open(self.filepath, 'ab') as f:
#             for graph in self._batch:
#                 pickle.dump(graph, f)
#         self._length += len(self._batch)
#         self._batch = []

#     def _calculate_length(self):
#         if not os.path.exists(self.filepath):
#             return 0

#         count = 0
#         try:
#             with open(self.filepath, 'rb') as f:
#                 while True:
#                     pickle.load(f)
#                     count += 1
#         except EOFError:
#             return count
#         except IOError as e:
#             print(f"IOError: {e}")
#             return 0

#     def __getitem__(self, index):
#         if isinstance(index, slice):
#             start, stop, step = index.indices(self._length)
#             with open(self.filepath, 'rb') as f:
#                 for i in range(stop):
#                     try:
#                         graph = pickle.load(f)
#                         if start <= i < stop and (i - start) % step == 0:
#                             yield graph
#                     except EOFError:
#                         break
#         else:
#             try:
#                 with open(self.filepath, 'rb') as f:
#                     for _ in range(index):
#                         pickle.load(f)  # Skip over unwanted objects
#                     graph = pickle.load(f)
#                     return graph
#             except EOFError:
#                 raise IndexError("Index out of range")
#             except IOError as e:
#                 print(f"IOError: {e}")
#                 raise IndexError("Unable to access file")

#     def __iter__(self):
#         self._iteration_buffer = []
#         self._iteration_buffer_index = 0
#         self.current_index = 0
#         if self._file_handle:
#             self._file_handle.close()
#         self._file_handle = open(self.filepath, 'rb')
#         return self
    
#     def __len__(self):
#         if self._length is None:
#             self._length = self._calculate_length()
#         return self._length

#     def __next__(self):
#         if self._iteration_buffer_index >= len(self._iteration_buffer):
#             self._load_next_batch()
#             if not self._iteration_buffer:
#                 raise StopIteration
#         graph = self._iteration_buffer[self._iteration_buffer_index]
#         self._iteration_buffer_index += 1
#         return graph

#     def _load_next_batch(self):
#         self._iteration_buffer = []
#         self._iteration_buffer_index = 0
#         try:
#             for _ in range(self.batch_size):
#                 try:
#                     graph = pickle.load(self._file_handle)
#                     self._iteration_buffer.append(graph)
#                 except EOFError:
#                     break
#         except IOError as e:
#             print(f"IOError: {e}")

#     def append(self, graph):
#         self.save_graph(graph)

#     def append_to_batch(self, graph):
#         self._batch.append(graph)
#         if len(self._batch) >= self.batch_size:
#             self.save_batch()

#     def clear(self):
#         open(self.filepath, 'w').close()
#         self._length = 0
#         self.current_index = 0
#         self._batch = []
#         self._iteration_buffer = []
#         self._iteration_buffer_index = 0
#         if self._file_handle:
#             self._file_handle.close()
#             self._file_handle = None

#     def finalize(self):
#         """Save any remaining graphs in the batch."""
#         if self._batch:
#             self.save_batch()
#         if self._file_handle:
#             self._file_handle.close()
#             self._file_handle = None

# class EnsembleByMethod(Ensemble):
#     def __init__(self, method, ensemble_size: int, strengths: np.array, z: float, filename: str, directory: str = "data/saved_ensembles", batch_size: int = 100, weighted=True, parallel=True, threads=None):
#         super().__init__(filename, directory, batch_size)
#         self.ensemble_size = ensemble_size
#         self.strengths = strengths
#         self.z = z
#         self.method = method
#         def create_and_append(method, strengths, z, weighted, ensemble):
#             graph = method(strengths, z, weighted)
#             ensemble.append_to_batch(graph)
#         if parallel:
#             print("Running parallel")
#             for _ in tqdm(range(ensemble_size//batch_size)):
#                 Parallel(n_jobs=cpu_count() if not threads else threads)(delayed(create_and_append)(method, strengths, z, weighted=weighted, ensemble=self) for _ in range(self.batch_size))
#             for _ in tqdm(range(ensemble_size%batch_size)):
#                 create_and_append(self.method, self.strengths, self.z, weighted, self)
#         else:
#             for _ in tqdm(range(ensemble_size)):
#                 create_and_append(self.method, self.strengths, self.z, weighted, self)
#         self.finalize()

class Ensemble:
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        self.directory = directory
        self.filename = None
        self.filepath = None
        self._length = 0
        self.batch_size = batch_size
        self._batch = []
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)

    def save_graph(self, graph):
        self._batch.append(graph)
        if len(self._batch) >= self.batch_size:
            self.save_batch()

    def save_batch(self):
        if not self._batch:
            return
        with h5py.File(self.filepath, 'a') as f:
            start_idx = self._length
            end_idx = start_idx + len(self._batch)
            for idx, graph in enumerate(self._batch):
                grp = f.create_group(f'graph_{start_idx + idx}')
                grp.attrs['n'] = graph.vcount()
                grp.attrs['m'] = graph.ecount()
                grp.create_dataset('edges', data=graph.get_edgelist())
                grp.create_dataset('weights', data=graph.es['weight'] if 'weight' in graph.edge_attributes() else [1]*graph.ecount())
            self._length = end_idx
        self._batch = []

    def _calculate_length(self):
        if not os.path.exists(self.filepath):
            return 0
        with h5py.File(self.filepath, 'r') as f:
            return len(f.keys())

    def _load_graph(self, index):
        try:
            with h5py.File(self.filepath, 'r') as f:
                grp = f[f'graph_{index}']
                n = grp.attrs['n']
                edges = grp['edges'][:]
                weights = grp['weights'][:]
                graph = ig.Graph(n, directed=True)
                graph.add_edges(edges)
                graph.es['weight'] = weights
                return graph
        except KeyError:
            raise IndexError("Index out of range")

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            return [self._load_graph(i) for i in range(start, stop, step)]
        else:
            return self._load_graph(index)

    def __iter__(self):
        self._file_handle = h5py.File(self.filepath, 'r')
        self.current_index = 0
        return self

    def __len__(self):
        return self._length

    def __next__(self):
        if self.current_index >= self._length:
            self._file_handle.close()
            raise StopIteration
        graph = self._load_graph(self.current_index)
        self.current_index += 1
        return graph

    def append(self, graph):
        self.save_graph(graph)

    def append_to_batch(self, graph):
        self._batch.append(graph)
        if len(self._batch) >= self.batch_size:
            self.save_batch()
    
    def extend_batch(self, graphs):
        self._batch.extend(graphs)
        if len(self._batch) >= self.batch_size:
            self.save_batch()

    def clear(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        self._length = 0
        self._batch = []

    def finalize(self):
        if self._batch:
            self.save_batch()

    def _get_metadata(self):
        if not os.path.exists(self.filepath.replace(".h5", "_metadata.pkl")):
            raise FileNotFoundError("Metadata file not found")
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            return metadata
    
    def _set_metadata(self, metadata: dict):
        for key, value in metadata.items():
            setattr(self, key, value)

class EnsembleByMethod(Ensemble):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        super().__init__(directory, batch_size)

    def _build_ensemble_method(self, method, ensemble_size: int, strengths: np.array, z: float, filename: str, weighted=True, parallel=True, threads=None):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        # Check if file already exists
        if os.path.exists(self.filepath):
            print("File already exists, deleting")
            os.remove(self.filepath)
        self.ensemble_size = ensemble_size
        self.strengths = strengths
        self.z = z
        self.method = method

        if parallel:
            print("Running parallel")
            for _ in tqdm(range(ensemble_size//self.batch_size)):
                results = Parallel(n_jobs=cpu_count() if not threads else threads)(
                    delayed(self._create_graph)(self.method, self.strengths, self.z, weighted) 
                    for _ in tqdm(range(self.batch_size))
                )
                self.extend_batch(results)
            for _ in tqdm(range(ensemble_size%self.batch_size)):
                self._create_and_append(self.method, self.strengths, self.z, weighted, self)
        else:
            for _ in tqdm(range(ensemble_size)):
                self._create_and_append(self.method, self.strengths, self.z, weighted, self)
        self.finalize()
        self._save_metadata()

    def _save_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'wb') as f:
            pickle.dump({
                "ensemble_size": self.ensemble_size,
                "strengths": self.strengths,
                "z": self.z,
                "length": self._length
            }, f)
    def _load_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            self.ensemble_size = metadata["ensemble_size"]
            self.strengths = metadata["strengths"]
            self.z = metadata["z"]
            self._length = metadata["length"]

    def load_ensemble(self, filename):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        self._load_metadata()

    def _create_and_append(self, method, strengths, z, weighted, ensemble):
            graph = method(strengths, z, weighted)
            ensemble.append_to_batch(graph)

    def _create_graph(self, method, strengths, z, weighted):
        return method(strengths, z, weighted)

class RMEnsemble(EnsembleByMethod):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        directory = os.path.join(directory, "RM")
        super().__init__(directory, batch_size)
    def build_ensemble(self, ensemble_size: int, strengths: np.array, z: float, filename: str, weighted=True, parallel=True, threads=None):
        self._build_ensemble_method(create_RM_graph, ensemble_size, strengths, z, filename, weighted, parallel, threads)

class NaiveRMEnsemble(EnsembleByMethod):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        directory = os.path.join(directory, "NaiveRM")
        super().__init__(directory, batch_size)
    def build_ensemble(self, ensemble_size: int, strengths: np.array, z: float, filename: str, weighted=True, parallel=True, threads=None):
        self._build_ensemble_method(create_naive_RM_graph, ensemble_size, strengths, z, filename, weighted, parallel, threads)

class CorrectedNaiveRMEnsemble(EnsembleByMethod):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        directory = os.path.join(directory, "CorrectedNaiveRM")
        super().__init__(directory, batch_size)
    def build_ensemble(self, ensemble_size: int, strengths: np.array, z: float, filename: str, weighted=True, parallel=True, threads=None):
        self._build_ensemble_method(create_corrected_naive_RM_graph, ensemble_size, strengths, z, filename, weighted, parallel, threads)

## Metropolis-Hastings
@jit(nopython=True)
def check_degree_condition_fast(out_degrees_current, in_degrees_current, i, j, new_link_proposed):
        if new_link_proposed == 1:
            return True
        elif out_degrees_current[i] - 1 == 0 or in_degrees_current[j] - 1 == 0:
            return False
        else:
            return True

@jit(nopython=True)
def metropolis_hastings_fast(initial_adj_matrix, ratio_matrix_1, ratio_matrix_0, num_iter):
    num_nodes = len(initial_adj_matrix)
    random_positions = np.random.randint(0, num_nodes, size=(5*num_iter, 2))
    rand_numbers_criterion = np.random.rand(num_iter)
    out_degrees_current = np.sum(initial_adj_matrix, axis=1)
    in_degrees_current = np.sum(initial_adj_matrix, axis=0)
    adj_matrix_current = initial_adj_matrix.copy()
    l=0
    # num_accepted = 0
    for k in range(num_iter):
        condition = False
        while not condition:
            position = random_positions[l]
            i = position[0]
            j = position[1]
            new_link_proposed = 1 - adj_matrix_current[i, j]
            condition = check_degree_condition_fast(out_degrees_current, in_degrees_current, i, j, new_link_proposed)
            l += 1
            if l == 5*num_iter:
                random_positions = np.random.randint(0, num_nodes, size=(5*num_iter, 2))
                l = 0
        if new_link_proposed == 1:
            #ratio = p_ij / (1 - p_ij)
            ratio = ratio_matrix_1[i, j]
        else:
            # ratio = (1 - p_ij) / p_ij
            ratio = ratio_matrix_0[i, j]
        if rand_numbers_criterion[k] < ratio:
            out_degrees_current[i] += 2*new_link_proposed - 1
            in_degrees_current[j] += 2*new_link_proposed - 1
            adj_matrix_current[i, j] = new_link_proposed
            # num_accepted += 1
    return adj_matrix_current

class DegreeCorrectedRMEnsemble(Ensemble):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        directory = os.path.join(directory, "DegreeCorrectedRM")
        super().__init__(directory, batch_size)
        # self.build_ensemble(initial_adj_matrix)

    def check_file_exists(self):
        return os.path.exists(self.filepath)

    def build_ensemble(self, ensemble_size: int, strengths: np.array, z: float, initial_adj_matrix: np.array, filename: str, num_single_iter = 1000000, skip_first=200):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        if self.check_file_exists():
            print("File already exists, deleting")
            os.remove(self.filepath)
        self.ensemble_size = ensemble_size
        self.num_single_iter = num_single_iter
        self.skip_first = skip_first
        self.strengths = strengths
        self.z = z
        self.initial_adj_matrix = initial_adj_matrix

        self.out_strengths = self.strengths[:,0]
        self.in_strengths = strengths[:,1]
        self.num_nodes = len(self.out_strengths)
        self.prob_matrix = np.ones([self.num_nodes, self.num_nodes]) - np.exp(-z*self.out_strengths.reshape(self.num_nodes,1)@self.in_strengths.reshape(1,self.num_nodes))
        self.ratio_matrix_1 = self.prob_matrix / (1 - self.prob_matrix)
        self.ratio_matrix_0 = (1 - self.prob_matrix) / self.prob_matrix
        adj_matrix = self.initial_adj_matrix.copy()
        # num_accepted_total = 0
        print(f"Starting initialization - first skip {self.skip_first} iterations")
        adj_matrix = metropolis_hastings_fast(adj_matrix, self.ratio_matrix_1, self.ratio_matrix_0, self.num_single_iter*self.skip_first)
        print("Finished initialization")
        print("Generating ensemble")
        for _ in tqdm(range(self.ensemble_size)):
            adj_matrix = metropolis_hastings_fast(adj_matrix, self.ratio_matrix_1, self.ratio_matrix_0, self.num_single_iter, )
            self.append_to_batch(utils.graph_from_adjacency(adj_matrix))
            # num_accepted_total += num_accepted_current
        self.finalize()
        self._save_metadata()

    def _save_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'wb') as f:
            pickle.dump({
                "num_single_iter": self.num_single_iter,
                "strengths": self.strengths,
                "z": self.z,
                "length": self._length
            }, f)

    def _load_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            self.num_single_iter = metadata["num_single_iter"]
            self.strengths = metadata["strengths"]
            self.z = metadata["z"]
            self._length = metadata["length"]

    def load_ensemble(self, filename):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        self._load_metadata()


class DegreeCorrectedRMEnsembleWeighted(Ensemble):
    def __init__(self, directory: str = "data/saved_ensembles", batch_size: int = 100):
        directory = os.path.join(directory, "DegreeCorrectedRMWeighted")
        super().__init__(directory, batch_size)
    
    def check_file_exists(self):
        return os.path.exists(self.filepath)
    
    def adj_to_weighted_degree_corrected(self, graph, weight_matrix):
        return utils.graph_from_adjacency(weight_matrix*utils.adj_matrix(graph), weighted=True)

    def build_ensemble(self, ensemble, strengths, avg_prob_matrix, filename):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        if self.check_file_exists():
            print("File already exists, deleting")
            os.remove(self.filepath)
        self.avg_prob_matrix = avg_prob_matrix
        new_strengths = strengths
        out_strengths = new_strengths[:,0]
        in_strengths = new_strengths[:,1]
        avg_prob_matrix = avg_prob_matrix.copy()
        W = np.sum(out_strengths)
        num_nodes = len(out_strengths)
        avg_prob_matrix[avg_prob_matrix==0] = 1e-12
        weight_matrix = (out_strengths.reshape(num_nodes,1)@in_strengths.reshape(1,num_nodes))/(W*avg_prob_matrix)
        weight_matrix = np.nan_to_num(weight_matrix, posinf=0)
        
        for graph in tqdm(ensemble):
            self.append_to_batch(self.adj_to_weighted_degree_corrected(graph, weight_matrix))
        self.finalize()
        self._save_metadata()

    def _save_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'wb') as f:
            pickle.dump({
                "avg_prob_matrix": self.avg_prob_matrix,
                "length": self._length
            }, f)

    def _load_metadata(self):
        with open(self.filepath.replace(".h5", "_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            self.avg_prob_matrix = metadata["avg_prob_matrix"]
            self._length = metadata["length"]

    def load_ensemble(self, filename):
        self.filename = filename
        self.filepath = os.path.join(self.directory, self.filename + ".h5")
        self._load_metadata()

def make_ensemble_from_subset(ensemble_new: Ensemble, filename_new: str, ensemble_old: Ensemble, index_start: int, index_end: int):
    """
    Copy graphs from ensemble_old to ensemble_new from index_start to index_end

    Args:
        ensemble_new (Ensemble): Ensemble to copy graphs to
        ensemble_old (Ensemble): Ensemble to copy graphs from
        index_start (int): Start index
        index_end (int): End index
    """
    ensemble_new._set_metadata(dict(ensemble_old._get_metadata()))
    ensemble_new.filename = filename_new
    ensemble_new.filepath = os.path.join(ensemble_new.directory, ensemble_new.filename + ".h5")
    if ensemble_new.check_file_exists():
        print("File already exists, deleting")
        os.remove(ensemble_new.filepath)
    for i in tqdm(range(index_start, index_end)):
        ensemble_new.append_to_batch(ensemble_old[i])
    ensemble_new.finalize()
    ensemble_new._length = index_end - index_start