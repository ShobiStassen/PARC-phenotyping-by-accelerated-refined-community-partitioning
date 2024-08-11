import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.logger import get_logger


logger = get_logger(__name__)


class PARC:
    def __init__(
        self,
        data,
        true_label=None,
        dist_std_local=3,
        jac_std_global="median",
        keep_all_local_dist="auto",
        too_big_factor=0.4,
        small_pop=10,
        jac_weighted_edges=True,
        knn=30,
        n_iter_leiden=5,
        random_seed=42,
        num_threads=-1,
        distance="l2",
        time_smallpop=15,
        partition_type="ModularityVP",
        resolution_parameter=1.0,
        knn_struct=None,
        neighbor_graph=None,
        hnsw_param_ef_construction=150
    ):
        # higher dist_std_local means more edges are kept
        # highter jac_std_global means more edges are kept
        if keep_all_local_dist == "auto":
            if data.shape[0] > 300000:
                keep_all_local_dist = True  # skips local pruning to increase speed
            else:
                keep_all_local_dist = False
        if resolution_parameter != 1:
            partition_type = "RBVP" # Reichardt and Bornholdt’s Potts model. Note that this is the same as ModularityVertexPartition when setting 𝛾 = 1 and normalising by 2m
        self.data = data
        self.true_label = true_label
        self.dist_std_local = dist_std_local   # similar to the jac_std_global parameter. avoid setting local and global pruning to both be below 0.5 as this is very aggresive pruning.
        self.jac_std_global = jac_std_global  #0.15 is also a recommended value performing empirically similar to "median". Generally values between 0-1.5 are reasonable.
        self.keep_all_local_dist = keep_all_local_dist #decides whether or not to do local pruning. default is "auto" which omits LOCAL pruning for samples >300,000 cells.
        self.too_big_factor = too_big_factor  #if a cluster exceeds this share of the entire cell population, then the PARC will be run on the large cluster. at 0.4 it does not come into play
        self.small_pop = small_pop  # smallest cluster population to be considered a community
        self.jac_weighted_edges = jac_weighted_edges #boolean. whether to partition using weighted graph
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden #the default is 5 in PARC
        self.random_seed = random_seed  # enable reproducible Leiden clustering
        self.num_threads = num_threads  # number of threads used in KNN search/construction
        self.distance = distance  # Euclidean distance "l2" by default; other options "ip" and "cosine"
        self.time_smallpop = time_smallpop #number of seconds trying to check an outlier
        self.partition_type = partition_type #default is the simple ModularityVertexPartition where resolution_parameter =1. In order to change resolution_parameter, we switch to RBConfigurationVP
        self.resolution_parameter = resolution_parameter # defaults to 1. expose this parameter in leidenalg
        self.knn_struct = knn_struct #the hnsw index of the KNN graph on which we perform queries
        self.neighbor_graph = neighbor_graph # CSR affinity matrix for pre-computed nearest neighbors
        self.hnsw_param_ef_construction = hnsw_param_ef_construction #set at 150. higher value increases accuracy of index construction. Even for several 100,000s of cells 150-200 is adequate

    def make_knn_struct(self, too_big=False, big_cluster=None):
        if self.knn > 190:
            logger.message(
                f"knn is {self.knn}, consider using a lower K_in for KNN graph construction"
            )
        ef_query = max(100, self.knn + 1)  # ef always should be > k. higher ef, more accurate query
        if not too_big:
            num_dims = self.data.shape[1]
            n_elements = self.data.shape[0]
            p = hnswlib.Index(space=self.distance, dim=num_dims)  # default to Euclidean distance
            p.set_num_threads(self.num_threads)  # set threads used in KNN construction
            if n_elements < 10000:
                ef_query = min(n_elements - 10, 500)
                ef_construction = ef_query
            else:
                ef_construction = self.hnsw_param_ef_construction
            if (num_dims > 30) & (n_elements <= 50000):
                # good for scRNA seq where dimensionality is high
                p.init_index(
                    max_elements=n_elements,
                    ef_construction=ef_construction,
                    M=48
                )
            else:
                p.init_index(
                    max_elements=n_elements,
                    ef_construction=ef_construction,
                    M=24  # 30
                )
            p.add_items(self.data)
        if too_big:
            num_dims = big_cluster.shape[1]
            n_elements = big_cluster.shape[0]
            p = hnswlib.Index(space="l2", dim=num_dims)
            p.init_index(max_elements=n_elements, ef_construction=200, M=30)
            p.add_items(big_cluster)
        p.set_ef(ef_query)  # ef should always be > k

        return p

    def knngraph_full(self):
        k_umap = 15
        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(k_umap+1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.data, k=k_umap)

        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]

        row_list.extend(
            list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten())
        )

        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:, np.newaxis])/row_sigma[:, np.newaxis]

        col_list = neighbor_array.flatten().tolist()
        distance_array = distance_array.flatten()
        distance_array = np.sqrt(distance_array)
        distance_array = distance_array * -1

        weight_list = np.exp(distance_array)

        threshold = np.mean(weight_list) + 2 * np.std(weight_list)

        weight_list[weight_list >= threshold] = threshold

        weight_list = weight_list.tolist()

        graph = csr_matrix((np.array(weight_list), (np.array(row_list), np.array(col_list))),
                           shape=(n_cells, n_cells))

        graph_transpose = graph.T
        prod_matrix = graph.multiply(graph_transpose)

        graph = graph_transpose + graph - prod_matrix
        return graph

    def make_csrmatrix_noselfloop(self, neighbor_array, distance_array):
        # neighbor array not listed in in any order of proximity
        row_list = []
        col_list = []
        weight_list = []

        n_neighbors = neighbor_array.shape[1]
        n_cells = neighbor_array.shape[0]
        rowi = 0
        discard_count = 0
        if not self.keep_all_local_dist:  # locally prune based on (squared) l2 distance

            logger.message(
                "Starting local pruning based on Euclidean distance metric at "
                f"{self.dist_std_local} standard deviations above the mean"
            )
            distance_array = distance_array + 0.1
            for row in neighbor_array:
                distlist = distance_array[rowi, :]
                to_keep = np.where(
                    distlist < np.mean(distlist) + self.dist_std_local * np.std(distlist)
                )[0]  # 0 * std
                updated_nn_ind = row[np.ix_(to_keep)]
                updated_nn_weights = distlist[np.ix_(to_keep)]
                discard_count = discard_count + (n_neighbors - len(to_keep))

                for ik in range(len(updated_nn_ind)):
                    if rowi != row[ik]:  # remove self-loops
                        row_list.append(rowi)
                        col_list.append(updated_nn_ind[ik])
                        dist = np.sqrt(updated_nn_weights[ik])
                        weight_list.append(1/(dist+0.1))

                rowi = rowi + 1

        if self.keep_all_local_dist:  # don't prune based on distance
            row_list.extend(
                list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten())
            )
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1. / (distance_array.flatten() + 0.1)).tolist()

        csr_graph = csr_matrix(
            (np.array(weight_list), (np.array(row_list), np.array(col_list))),
            shape=(n_cells, n_cells)
        )
        return csr_graph

    def func_mode(self, ll):  # return MODE of list
        # If multiple items are maximal, the function returns the first one encountered.
        return max(set(ll), key=ll.count)

    def run_toobig_subPARC(
        self,
        X_data,
        jac_std_toobig=0.3,
        jac_weighted_edges=True
    ):

        n_elements = X_data.shape[0]
        hnsw = self.make_knn_struct(too_big=True, big_cluster=X_data)
        if n_elements <= 10:
            logger.message("Consider increasing the too_big_factor")
        if n_elements > self.knn:
            knnbig = self.knn
        else:
            knnbig = int(max(5, 0.2 * n_elements))

        neighbor_array, distance_array = hnsw.knn_query(X_data, k=knnbig)
        csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources.tolist(), targets.tolist()))
        edgelist_copy = edgelist.copy()
        G = ig.Graph(edgelist, edge_attrs={"weight": csr_array.data.tolist()})
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)  # list of jaccard weights
        new_edgelist = []
        sim_list_array = np.asarray(sim_list)
        if jac_std_toobig == "median":
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_toobig * np.std(sim_list)

        logger.message(f"jac threshold {threshold:.3f}")
        logger.message(f"jac std {np.std(sim_list):.3f}")
        logger.message(f"jac mean {np.mean(sim_list):.3f}")

        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs:
            new_edgelist.append(edgelist_copy[ii])

        sim_list_new = list(sim_list_array[strong_locs])

        if jac_weighted_edges:
            G_sim = ig.Graph(
                n=n_elements,
                edges=list(new_edgelist),
                edge_attrs={"weight": sim_list_new}
            )
        else:
            G_sim = ig.Graph(n=n_elements, edges=list(new_edgelist))
        G_sim.simplify(combine_edges="sum")
        if jac_weighted_edges:
            if self.partition_type == "ModularityVP":
                logger.message("partition type MVP")
                partition = leidenalg.find_partition(
                    G_sim, leidenalg.ModularityVertexPartition,
                    weights="weight",
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed
                )
            else:
                logger.message("partition type RBC")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )
        else:
            if self.partition_type == "ModularityVP":
                logger.message("partition type MVP")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.ModularityVertexPartition,
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed
                )
            else:
                logger.message("partition type RBC")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.RBConfigurationVertexPartition,
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )
        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)[1]
        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])
            if population < small_pop:
                small_pop_exist = True
                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group

        time_smallpop_start = time.time()
        logger.message("Handling fragments...")
        while small_pop_exist & (time.time() - time_smallpop_start < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True

                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)[1]

        return PARC_labels_leiden

    def run_subPARC(self):

        X_data = self.data
        too_big_factor = self.too_big_factor
        small_pop = self.small_pop
        jac_std_global = self.jac_std_global
        jac_weighted_edges = self.jac_weighted_edges
        knn = self.knn
        n_elements = X_data.shape[0]

        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        else:
            if self.knn_struct is None:
                logger.message("knn struct was not available, creating new one")
                self.knn_struct = self.make_knn_struct()
            else:
                logger.message("knn struct already exists")
            neighbor_array, distance_array = self.knn_struct.knn_query(X_data, k=knn)
            csr_array = self.make_csrmatrix_noselfloop(neighbor_array, distance_array)

        sources, targets = csr_array.nonzero()

        edgelist = list(zip(sources, targets))

        edgelist_copy = edgelist.copy()

        G = ig.Graph(edgelist, edge_attrs={"weight": csr_array.data.tolist()})
        sim_list = G.similarity_jaccard(pairs=edgelist_copy)

        logger.message("Starting global pruning...")

        sim_list_array = np.asarray(sim_list)
        edge_list_copy_array = np.asarray(edgelist_copy)

        if jac_std_global == "median":
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - jac_std_global * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        new_edgelist = list(edge_list_copy_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])

        G_sim = ig.Graph(
            n=n_elements,
            edges=list(new_edgelist),
            edge_attrs={"weight": sim_list_new}
        )
        G_sim.simplify(combine_edges="sum")  # "first"
        logger.message("Starting Leiden community detection...")
        if jac_weighted_edges:
            if self.partition_type == "ModularityVP":
                logger.message("partition type MVP")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.ModularityVertexPartition,
                    weights="weight",
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed
                )
            else:
                logger.message("partition type RBC")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )

        else:
            if self.partition_type == "ModularityVP":
                logger.message("partition type MVP")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.ModularityVertexPartition,
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed
                )
            else:
                logger.message("partition type RBC")
                partition = leidenalg.find_partition(
                    G_sim,
                    leidenalg.RBConfigurationVertexPartition,
                    n_iterations=self.n_iter_leiden,
                    seed=self.random_seed,
                    resolution_parameter=self.resolution_parameter
                )

        PARC_labels_leiden = np.asarray(partition.membership)
        PARC_labels_leiden = np.reshape(PARC_labels_leiden, (n_elements, 1))

        too_big = False

        # The 0th cluster is the largest one.
        # So, if cluster 0 is not too big, then the others won't be too big either
        cluster_i_loc = np.where(PARC_labels_leiden == 0)[0]
        pop_i = len(cluster_i_loc)
        if pop_i > too_big_factor * n_elements:  # 0.4
            too_big = True
            cluster_big_loc = cluster_i_loc
            list_pop_too_bigs = [pop_i]

        while too_big:

            X_data_big = X_data[cluster_big_loc, :]
            PARC_labels_leiden_big = self.run_toobig_subPARC(X_data_big)
            PARC_labels_leiden_big = PARC_labels_leiden_big + 100000
            pop_list = []

            for item in set(list(PARC_labels_leiden_big.flatten())):
                pop_list.append([item, list(PARC_labels_leiden_big.flatten()).count(item)])

            logger.message(f"pop of big clusters {pop_list}")
            jj = 0
            logger.message(f"shape PARC_labels_leiden {PARC_labels_leiden.shape}")
            for j in cluster_big_loc:
                PARC_labels_leiden[j] = PARC_labels_leiden_big[jj]
                jj = jj + 1
            PARC_labels_leiden = np.unique(
                list(PARC_labels_leiden.flatten()), return_inverse=True
            )[1]
            logger.message(f"New set of labels {set(PARC_labels_leiden)}")
            too_big = False
            set_PARC_labels_leiden = set(PARC_labels_leiden)

            PARC_labels_leiden = np.asarray(PARC_labels_leiden)
            for cluster_ii in set_PARC_labels_leiden:
                cluster_ii_loc = np.where(PARC_labels_leiden == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_bigs
                if pop_ii > too_big_factor * n_elements and not_yet_expanded:
                    too_big = True
                    logger.message(f"Cluster {cluster_ii} is too big and has population {pop_ii}.")
                    cluster_big_loc = cluster_ii_loc
                    cluster_big = cluster_ii
                    big_pop = pop_ii
            if too_big:
                list_pop_too_bigs.append(big_pop)
                logger.message(
                    f"Cluster {cluster_big} is too big and has population {big_pop}."
                    "It will be expanded."
                )
        PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)[1]
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False

        for cluster in set(PARC_labels_leiden):
            population = len(np.where(PARC_labels_leiden == cluster)[0])

            if population < small_pop:  # 10
                small_pop_exist = True

                small_pop_list.append(list(np.where(PARC_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    PARC_labels_leiden[single_cell] = best_group
        time_smallpop_start = time.time()
        while small_pop_exist & ((time.time() - time_smallpop_start) < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(PARC_labels_leiden.flatten())):
                population = len(np.where(PARC_labels_leiden == cluster)[0])
                if population < small_pop:
                    small_pop_exist = True
                    logger.message(f"Cluster {cluster} has small population of {population}.")
                    small_pop_list.append(np.where(PARC_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = PARC_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    PARC_labels_leiden[single_cell] = best_group

        PARC_labels_leiden = np.unique(list(PARC_labels_leiden.flatten()), return_inverse=True)[1]
        PARC_labels_leiden = list(PARC_labels_leiden.flatten())
        pop_list = []
        for item in set(PARC_labels_leiden):
            pop_list.append((item, PARC_labels_leiden.count(item)))
        logger.message(f"Cluster labels and populations {len(pop_list)} {pop_list}")

        self.labels = PARC_labels_leiden
        return

    def accuracy(self, onevsall=1):

        true_labels = self.true_label
        Index_dict = {}
        PARC_labels = self.labels
        N = len(PARC_labels)
        n_cancer = list(true_labels).count(onevsall)
        n_pbmc = N - n_cancer

        for k in range(N):
            Index_dict.setdefault(PARC_labels[k], []).append(true_labels[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        pbmc_labels = []
        thp1_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = self.func_mode(vals)
            if majority_val == onevsall:
                logger.message(f"Cluster {kk} has majority {onevsall} with population {len(vals)}")
            if kk == -1:
                len_unknown = len(vals)
                logger.message(f"len unknown: {len_unknown}")
            if (majority_val == onevsall) and (kk != -1):
                thp1_labels.append(kk)
                fp = fp + len([e for e in vals if e != onevsall])
                tp = tp + len([e for e in vals if e == onevsall])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != onevsall) and (kk != -1):
                pbmc_labels.append(kk)
                tn = tn + len([e for e in vals if e != onevsall])
                fn = fn + len([e for e in vals if e == onevsall])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(PARC_labels)
        PARC_labels_array = np.array(PARC_labels)
        number_clusters_for_target = len(thp1_labels)
        for cancer_class in thp1_labels:
            predict_class_array[PARC_labels_array == cancer_class] = 1
        for benign_class in pbmc_labels:
            predict_class_array[PARC_labels_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / N
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_cancer
        tpr = tp / n_cancer
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0:
            recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)
        majority_truth_labels = np.empty((len(true_labels), 1), dtype=object)

        for cluster_i in set(PARC_labels):
            cluster_i_loc = np.where(np.asarray(PARC_labels) == cluster_i)[0]
            true_labels = np.asarray(true_labels)
            majority_truth = self.func_mode(list(true_labels[cluster_i_loc]))
            majority_truth_labels[cluster_i_loc] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def run_PARC(self):
        logger.message(
            f"Input data has shape {self.data.shape[0]} (samples) x {self.data.shape[1]} (features)"
        )
        if self.true_label is None:
            self.true_label = [1] * self.data.shape[0]
        list_roc = []

        time_start_total = time.time()

        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        self.run_subPARC()
        run_time = time.time() - time_start_total
        logger.message(f"Time elapsed to run PARC: {run_time:.1f} seconds")

        targets = list(set(self.true_label))
        N = len(list(self.true_label))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            "jac_std_global": [self.jac_std_global],
            "dist_std_local": [self.dist_std_local],
            "runtime(s)": [run_time]
        })
        self.majority_truth_labels = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for onevsall_val in targets:
                logger.message(f"Target is {onevsall_val}")
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = \
                    self.accuracy(onevsall=onevsall_val)
                f1_current = vals_roc[1]
                logger.message(f"Target {onevsall_val} has f1-score of {(f1_current * 100):.2f}")
                f1_accumulated = f1_accumulated + \
                    f1_current * (list(self.true_label).count(onevsall_val)) / N
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_global, self.dist_std_local, onevsall_val] +
                    vals_roc +
                    [numclusters_targetval] +
                    [run_time]
                )

            f1_mean = f1_acc_noweighting / len(targets)
            logger.message(f"f1-score (unweighted) mean: {(f1_mean * 100):.2f}")
            logger.message(f"f1-score weighted (by population): {(f1_accumulated * 100):.2f}")

            df_accuracy = pd.DataFrame(
                list_roc,
                columns=[
                    "jac_std_global", "dist_std_local", "onevsall-target", "error rate",
                    "f1-score", "tnr", "fnr", "tpr", "fpr", "precision", "recall", "num_groups",
                    "population of target", "num clusters", "clustering runtime"
                ]
            )

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels
        return

    def run_umap_hnsw(
            self,
            X_input,
            graph,
            n_components=2,
            alpha: float = 1.0,
            negative_sample_rate: int = 5,
            gamma: float = 1.0,
            spread=1.0,
            min_dist=0.1,
            n_epochs=0,
            init_pos="spectral",
            random_state_seed=1,
            densmap=False,
            densmap_kwds={},
            output_dens=False
    ):
        """Perform a fuzzy simplicial set embedding, using a specified initialisation method and
        then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and
        low-dimensional fuzzy simplicial sets.

        See `umap.umap_ simplicial_set_embedding
        <https://github.com/lmcinnes/umap/blob/master/umap/umap_.py>`__.

        Args:
            X_input: (array) an array containing the input data, with shape n_samples x n_features.
            graph: (array) the 1-skeleton of the high dimensional fuzzy simplicial set as
                represented by a graph for which we require a sparse matrix for the
                (weighted) adjacency matrix.
            n_components: (int) the dimensionality of the Euclidean space into which the data
                will be embedded.
            alpha: (float) the initial learning rate for stochastic gradient descent (SGD).
            negative_sample_rate: (int) the number of negative samples to select per positive
                sample in the optimization process. Increasing this value will result in
                greater repulsive force being applied, greater optimization cost, but
                slightly more accuracy.
            gamma: (float) weight to apply to negative samples.
            spread: (float) the upper range of the x-value to be used to fit a curve for the
                low-dimensional fuzzy simplicial complex construction. This curve should be
                close to an offset exponential decay. Must be greater than 0. See
                ``umap.umap_.find_ab_params``.
            min_dist: (float) See ``umap.umap_.find_ab_params``.
            n_epochs: (int) the number of training epochs to be used in optimizing the
                low-dimensional embedding. Larger values result in more accurate embeddings.
                If 0 is specified, a value will be selected based on the size of the input dataset
                (200 for large datasets, 500 for small).
            init_pos: (string) how to initialize the low-dimensional embedding. One of:
                `spectral`: use a spectral embedding of the fuzzy 1-skeleton
                `random`: assign initial embedding positions at random
                `pca`: use the first n components from Principal Component Analysis (PCA)
                    applied to the input data.
            random_state_seed: (int) an integer to pass as a seed for the Numpy RandomState.
            densmap: (bool) whether to use the density-augmented objective function to optimize
                the embedding according to the densMAP algorithm.
            densmap_kwds: (dict) keyword arguments to be used by the densMAP optimization.
            output_dens: (bool) whether to output local radii in the original data
                and the embedding.

        Returns:
            embedding: array of shape (n_samples, n_components)
                The optimized of ``graph`` into an ``n_components`` dimensional
                euclidean space.
        """

        a, b = find_ab_params(spread, min_dist)
        logger.message(f"a: {a}, b: {b}, spread: {spread}, dist: {min_dist}")

        X_umap = simplicial_set_embedding(
            data=X_input,
            graph=graph,
            n_components=n_components,
            initial_alpha=alpha,
            a=a,
            b=b,
            n_epochs=0,
            metric_kwds={},
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            init=init_pos,
            random_state=np.random.RandomState(random_state_seed),
            metric="euclidean",
            verbose=1,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            output_dens=output_dens
        )
        return X_umap[0]
