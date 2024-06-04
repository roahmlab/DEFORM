import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def edge_fn(distances, sigma=1.0):
    # distances is a matrix of distances between points
    return np.exp(-distances**2/(2*sigma**2))

def get_adjacency_matrix(data):
    data = data-data.mean(axis=0)
    sum_square = np.sum(data**2, axis=1)
    distance_matrix = np.sqrt(np.clip(sum_square[:,None] + sum_square[None,:] - 2*data.dot(data.T), a_min=1e-10, a_max=1e10))
    
    # find sigma using mean of second nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    sigma = np.mean(distances[:,1])
    graph = edge_fn(distance_matrix, sigma)
    np.fill_diagonal(graph, 0)
    return graph

def graph_diffuse(graph, steps=10, src=5, dst=46):
    # src is the index of the point we want to diffuse from
    # dst is the index of the point we want to diffuse to
    # steps is the number of steps we want to diffuse
    # print("start graph diffusion")
    heat_distribution = np.zeros(graph.shape[0])
    heat_distribution[src] = 1.0

    degree_matrix = np.diag(np.sum(graph, axis=1))
    adj_matrix = graph
    normalized_adj_matrix = np.linalg.inv(degree_matrix).dot(adj_matrix)

    for i in range(steps):
        heat_distribution = normalized_adj_matrix.dot(heat_distribution)
        heat_distribution[src]=1.0
        heat_distribution[dst]=0.0

    rank = np.argsort(heat_distribution)
    return rank
    idx_to_rank = np.zeros(rank.shape[0], dtype=np.int32)
    for i in range(rank.shape[0]):
        idx_to_rank[rank[i]]=i
    print(idx_to_rank)
    return idx_to_rank

def loss_function(start, end, graph, data):
    reordered_indices = graph_diffuse(graph, src=start, dst=end)
    reordered_data = data[reordered_indices, :]
    pairwise_distances = np.sqrt(np.sum(np.square(reordered_data[1:, :] - reordered_data[:-1, :]), axis=1))
    loss = np.sum(pairwise_distances**2)
    return loss

def optimize_start_end(graph, data):
    min_loss = float('inf')
    best_start = None
    best_end = None

    # Example: Brute-force search over all pairs of start and end vertices
    for start in range(graph.shape[0]):
        for end in range(graph.shape[0]):
            if start != end:
                current_loss = loss_function(start, end, graph, data)
                if current_loss < min_loss:
                    min_loss = current_loss
                    best_start = start
                    best_end = end

    return best_start, best_end

def self_index(data):
    graph = get_adjacency_matrix(data)
    best_start, best_end = optimize_start_end(graph, data)
    reordered_indices = graph_diffuse(graph, src=best_start, dst=best_end)
    reordered_data = data[reordered_indices, :]
    return reordered_data

def directed_index_fill(data, pred):
    num_dummies = len(pred) - len(data)
    if num_dummies != 0:
        dummy_point = [0, 0, 0]  # This can be any arbitrary point
        padded_data = np.vstack([data, [dummy_point] * num_dummies])
        cost_matrix = cdist(padded_data, pred, 'euclidean')
        # Adjust the cost for dummy points
        high_cost = 1e9  # Set a high cost for matching with dummy points
        for i in range(len(data), len(padded_data)):
            cost_matrix[i, :] = high_cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # print(col_ind)
        reorder_data = np.zeros_like(padded_data)
        for i in range(len(col_ind)):
            if i < len(data):
                # reorder_data[col_ind[i]] = (padded_data[i] * 0.5 + pred[col_ind[i]] *0.5
                reorder_data[col_ind[i]] = (padded_data[i] * 0.6 + pred[col_ind[i]] * 0.4)
                if col_ind[i] == 0:
                    reorder_data[col_ind[i]] = pred[0]
                if col_ind[i] == len(pred) - 1:
                    reorder_data[col_ind[i]] = pred[len(pred) -1]
            else:
                reorder_data[col_ind[i]] = pred[col_ind[i]]
        filtered_arr = reorder_data
        return filtered_arr

    else:
        cost_matrix = cdist(data, pred, 'euclidean')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        reorder_data = np.zeros_like(data)
        for i in range(len(col_ind)):
            reorder_data[col_ind[i]] = (data[i] + pred[col_ind[i]]) / 2.
            if i < len(data):
                if col_ind[i] == 0:
                    reorder_data[col_ind[i]] = pred[0]
                if col_ind[i] == len(pred) - 1:
                    reorder_data[col_ind[i]] = pred[len(pred) - 1]
        filtered_arr = reorder_data
        return filtered_arr




# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
#
# # Example 3D points
# points_set1 = np.array([[-1, -1, -1], [0.1, 0.1, 0.1], [1, 2, 0]])  # Smaller set
# points_set2 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])  # Larger set
#
# # Pad the smaller set with dummy points
# dummy_point = np.zeros(3)  # Arbitrary dummy point
# num_dummies = len(points_set2) - len(points_set1)
# padded_set1 = np.vstack([points_set1, [dummy_point] * num_dummies])
#
# # Create a cost matrix
# cost_matrix = cdist(padded_set1, points_set2, 'euclidean')
#
# # Adjust the cost for dummy points
# high_cost = 1e9  # High cost for matching with dummy points
# for i in range(len(points_set1), len(padded_set1)):
#     cost_matrix[i, :] = high_cost
#
# # Apply the Hungarian algorithm
# row_ind, col_ind = linear_sum_assignment(cost_matrix)
# new_test = np.zeros_like(padded_set1)
# for i in range(len(col_ind)):
#     new_test[col_ind[i]] = padded_set1[i]
# # Initialize an array to store the reordered set1
#
# mask = ~(np.all(new_test == [0, 0, 0], axis=1))
#
# # Use the mask to filter the array
# filtered_arr = new_test[mask]
# print("Reordered set1:", filtered_arr)

