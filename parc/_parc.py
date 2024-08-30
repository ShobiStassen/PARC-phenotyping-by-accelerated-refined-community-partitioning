import numpy as np
import pandas as pd
import hnswlib
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg
import time
from umap.umap_ import find_ab_params, simplicial_set_embedding
from parc.utils import get_mode
from parc.logger import get_logger


logger = get_logger(__name__)


class PARC:
    """``PARC``: ``P``henotyping by ``A``ccelerated ``R``efined ``C``ommunity-partitioning.

    Attributes:
        x_data:
            An array of the input x data, with dimensions ``(n_samples, n_features)``.
        y_data_true:
            An array of the true output y labels.
        y_data_pred:
            An array of the predicted output y labels.
        knn:
            The number of nearest neighbors k for the k-nearest neighbours algorithm.
            Larger k means more neighbors in a cluster and therefore less clusters.
        n_iter_leiden:
            The number of iterations for the Leiden algorithm.
        random_seed:
            The random seed to enable reproducible Leiden clustering.
        distance_metric:
            The distance metric to be used in the KNN algorithm:

                * ``l2``: Euclidean distance L^2 norm:

                  .. code-block:: python

                    d = np.sum((x_i - y_i)**2)
                * ``cosine``: cosine similarity:

                  .. code-block:: python

                    d = 1.0 - np.sum(x_i*y_i) / np.sqrt(sum(x_i*x_i) * np.sum(y_i*y_i))
                * ``ip``: inner product distance:

                  .. code-block:: python

                    d = 1.0 - np.sum(x_i*y_i)
        n_threads:
            The number of threads used in the KNN algorithm.
        hnsw_param_ef_construction:
            A higher value increases accuracy of index construction.
            Even for O(100 000) cells, 150-200 is adequate.
        neighbor_graph:
            A sparse matrix with dimensions ``(n_samples, n_samples)``, containing the
            distances between nodes.
        knn_struct:
            The HNSW index of the KNN graph on which we perform queries.
        l2_std_factor:
            The multiplier used in calculating the Euclidean distance threshold for the distance
            between two nodes during local pruning:

            .. code-block:: python

                max_distance = np.mean(distances) + l2_std_factor * np.std(distances)

            Avoid setting both the ``jac_std_factor`` (global) and the ``l2_std_factor`` (local)
            to < 0.5 as this is very aggressive pruning.
            Higher ``l2_std_factor`` means more edges are kept.
        do_prune_local:
            Whether or not to do local pruning. If ``None`` (default), set to ``False`` if the
            number of samples is > 300 000, and set to ``True`` otherwise.
        jac_threshold_type:
            One of ``"median"`` or ``"mean"``. Determines how the Jaccard similarity threshold is
            calculated during global pruning.
        jac_std_factor:
            The multiplier used in calculating the Jaccard similarity threshold for the similarity
            between two nodes during global pruning for ``jac_threshold_type = "mean"``:

            .. code-block:: python

                threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

            Setting ``jac_std_factor = 0.15`` and ``jac_threshold_type="mean"`` performs empirically
            similar to ``jac_threshold_type="median"``, which does not use the ``jac_std_factor``.
            Generally values between 0-1.5 are reasonable. Higher ``jac_std_factor`` means more
            edges are kept.
        jac_weighted_edges:
            Whether to partition using the weighted graph.
        resolution_parameter:
            The resolution parameter to be used in the Leiden algorithm.
            In order to change ``resolution_parameter``, we switch to ``RBVP``.
        partition_type:
            The partition type to be used in the Leiden algorithm:

            * ``ModularityVP``: ModularityVertexPartition, ``resolution_parameter=1``
            * ``RBVP``: RBConfigurationVP, Reichardt and Bornholdt’s Potts model. Note that this
                is the same as ``ModularityVP`` when setting 𝛾 = 1 and normalising by 2m.
        large_community_factor:
            A factor used to determine if a community is too large.
            If the community size is greater than ``large_community_factor * n_samples``,
            then the community is too large and the ``PARC`` algorithm will be run on the single
            community to split it up. The default value of ``0.4`` ensures that all communities
            will be less than the cutoff size.
        small_community_size:
            The smallest population size to be considered a community.
        small_community_timeout:
            The maximum number of seconds trying to check an outlying small community.
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data_true: np.ndarray | None = None,
        knn: int = 30,
        n_iter_leiden: int = 5,
        random_seed: int = 42,
        distance_metric: str = "l2",
        n_threads: int = -1,
        hnsw_param_ef_construction: int = 150,
        neighbor_graph: csr_matrix | None = None,
        knn_struct: hnswlib.Index | None = None,
        l2_std_factor: float = 3,
        jac_threshold_type: str = "median",
        jac_std_factor: float = 0.15,
        jac_weighted_edges: bool = True,
        do_prune_local: bool | None = None,
        large_community_factor: float = 0.4,
        small_community_size: int = 10,
        small_community_timeout: float = 15,
        resolution_parameter: float = 1.0,
        partition_type: str = "ModularityVP"
    ):
        self.x_data = x_data
        self.y_data_true = y_data_true
        self.y_data_pred = None
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.distance_metric = distance_metric
        self.n_threads = n_threads
        self.hnsw_param_ef_construction = hnsw_param_ef_construction
        self.neighbor_graph = neighbor_graph
        self.knn_struct = knn_struct
        self.l2_std_factor = l2_std_factor
        self.jac_threshold_type = jac_threshold_type
        self.jac_std_factor = jac_std_factor
        self.jac_weighted_edges = jac_weighted_edges
        self.do_prune_local = do_prune_local
        self.large_community_factor = large_community_factor
        self.small_community_size = small_community_size
        self.small_community_timeout = small_community_timeout
        self.resolution_parameter = resolution_parameter
        self.partition_type = partition_type

    @property
    def y_data_true(self) -> np.ndarray:
        return self._y_data_true

    @y_data_true.setter
    def y_data_true(self, y_data_true: np.ndarray | None):
        if y_data_true is None:
            y_data_true = [1] * self.x_data.shape[0]
        self._y_data_true = y_data_true

    @property
    def do_prune_local(self) -> bool:
        return self._do_prune_local

    @do_prune_local.setter
    def do_prune_local(self, do_prune_local: bool | None):
        if do_prune_local is None:
            if self.x_data.shape[0] > 300000:
                logger.message(
                    f"Sample size is {self.x_data.shape[0]}, setting do_prune_local "
                    f"to False so that local pruning will be skipped and algorithm will be faster."
                )
                do_prune_local = False
            else:
                do_prune_local = True

        self._do_prune_local = do_prune_local

    @property
    def partition_type(self) -> str:
        return self._partition_type

    @partition_type.setter
    def partition_type(self, partition_type: str):
        if self.resolution_parameter != 1:
            self._partition_type = "RBVP"
        else:
            self._partition_type = partition_type

    def make_knn_struct(
        self,
        x_data: np.ndarray,
        knn: int,
        ef_query: int = 100,
        hnsw_param_m: int | None = None,
        hnsw_param_ef_construction: int | None = None,
        distance_metric: str = "l2",
        n_threads: int | None = None
    ) -> hnswlib.Index:
        """Create a KNN graph using the Hierarchical Navigable Small Worlds (HNSW) algorithm.

        See `hnswlib.Index
        <https://github.com/nmslib/hnswlib/blob/master/python_bindings/LazyIndex.py>`__.

        Args:
            x_data:
                An array of the input x data, with dimensions ``(n_samples, n_features)``.
            knn:
                The number of nearest neighbors k for the k-nearest neighbors algorithm.
            ef_query:
                The ``ef_query`` parameter corresponds to the ``hnswlib.Index`` parameter ``ef``.
                It determines the size of the dynamic list for the nearest neighbors
                (used during the search). Higher ``ef`` leads to more accurate but slower search.
                Must be a value in the interval ``(k, n_samples]``.
            hnsw_param_m:
                The ``hnsw_param_m`` parameter corresponds to the ``hnswlib.Index`` parameter ``M``.
                It corresponds to the number of bi-directional links created for every new element
                during the ``hnswlib.Index`` construction. Reasonable range for ``M`` is ``2-100``.
                Higher ``M`` works better on datasets with high intrinsic dimensionality and/or
                high recall, while lower ``M`` works better for datasets with low intrinsic
                dimensionality and/or low recall. The parameter also determines the algorithm's
                memory consumption, which is roughly ``M * 8-10 bytes`` per stored element.

                For example, for ``n_features=4`` random vectors, the optimal ``M`` for search
                is somewhere around ``6``, while for high dimensional datasets
                (word embeddings, good face descriptors, scRNA seq), higher values of ``M``
                are required (e.g. ``M=48-64``) for optimal performance at high recall.
                The range ``M=12-48`` is adequate for the most of the use cases.
                When ``M`` is changed, one has to update the other parameters.
                Nonetheless, ``ef`` and ``ef_construction`` parameters can be roughly estimated
                by assuming that ``M*ef_construction`` is a constant.
            hnsw_param_ef_construction:
                The ``hnsw_param_ef_construction`` parameter corresponds to the ``hnswlib.Index``
                parameter ``ef_construction``. It has the same meaning as ``ef_query``,
                but controls the index_time/index_accuracy. Higher values lead to longer
                construction, but better index quality. Even for ``O(100 000)`` cells,
                ``ef_construction ~ 150-200`` is adequate.

                At some point, increasing ``ef_construction`` does not improve the quality of
                the index. One way to check if the selection of ``ef_construction`` is
                appropriate is to measure a recall for ``M`` nearest neighbor search when
                ``ef = ef_construction``: if the recall is lower than ``0.9``, then there is room
                for improvement.
            distance_metric:
                The distance metric to be used in the KNN algorithm:

                * ``l2``: Euclidean distance L^2 norm:

                  .. code-block:: python

                    d = np.sum((x_i - y_i)**2)

                * ``cosine``: cosine similarity

                  .. code-block:: python

                    d = 1.0 - np.sum(x_i*y_i) / np.sqrt(np.sum(x_i*x_i) * np.sum(y_i*y_i))

            n_threads:
                The number of threads used in the KNN algorithm.

        Returns:
            The HNSW index of the k-nearest neighbors graph.
        """

        if knn > 190:
            logger.message(
                f"knn is {knn}, consider using a lower K_in for KNN graph construction"
            )

        n_features = x_data.shape[1]
        n_samples = x_data.shape[0]

        ef_query = min(max(ef_query, knn + 1), n_samples)
        logger.info(f"Setting ef_query to {ef_query}")

        knn_struct = hnswlib.Index(space=distance_metric, dim=n_features)

        if n_threads is not None:
            knn_struct.set_num_threads(n_threads)

        if hnsw_param_m is None:
            if n_features > 30 and n_samples <= 50000:
                hnsw_param_m = 48
            else:
                hnsw_param_m = 24

        if hnsw_param_ef_construction is None:
            if n_samples < 10000:
                hnsw_param_ef_construction = min(n_samples - 10, 500)
            else:
                hnsw_param_ef_construction = self.hnsw_param_ef_construction

        knn_struct.init_index(
            max_elements=n_samples,
            ef_construction=hnsw_param_ef_construction,
            M=hnsw_param_m
        )
        knn_struct.add_items(x_data)
        knn_struct.set_ef(ef_query)

        return knn_struct

    def create_knn_graph(self, knn: int = 15) -> csr_matrix:
        """Create a full k-nearest neighbors graph using the HNSW algorithm.

        Args:
            knn: The number of nearest neighbors k for the k-nearest neighbours algorithm.

        Returns:
            A compressed sparse row matrix with dimensions ``(n_samples, n_samples)``,
            containing the pruned distances.
        """

        # neighbors in array are not listed in in any order of proximity
        self.knn_struct.set_ef(knn + 1)
        neighbor_array, distance_array = self.knn_struct.knn_query(self.x_data, k=knn)

        row_list = []
        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]

        row_list.extend(
            list(np.transpose(np.ones((n_neighbors, n_samples)) * range(0, n_samples)).flatten())
        )

        row_min = np.min(distance_array, axis=1)
        row_sigma = np.std(distance_array, axis=1)

        distance_array = (distance_array - row_min[:, np.newaxis]) / row_sigma[:, np.newaxis]
        distance_array = np.sqrt(distance_array.flatten()) * -1

        col_list = neighbor_array.flatten().tolist()

        weight_list = np.exp(distance_array)
        threshold = np.mean(weight_list) + 2 * np.std(weight_list)
        weight_list[weight_list >= threshold] = threshold

        csr_array = csr_matrix(
            (weight_list, (np.array(row_list), np.array(col_list))),
            shape=(n_samples, n_samples)
        )
        prod_matrix = csr_array.multiply(csr_array.T)
        csr_array = csr_array.T + csr_array - prod_matrix
        return csr_array

    def prune_local(
        self,
        neighbor_array: np.ndarray,
        distance_array: np.ndarray,
        l2_std_factor: float | None = None
    ) -> csr_matrix:
        """Prune the nearest neighbors array.

        If ``do_prune_local==True``, remove any neighbors which are further away than
        the specified cutoff distance. Also, remove any self-loops. Return in the ``csr_matrix``
        format.

        If ``do_prune_local==False``, then don't perform any pruning and return the original
        arrays in the ``csr_matrix`` format.

        Args:
            neighbor_array: An array with dimensions ``(n_samples, k)`` listing the
                k nearest neighbors for each data point.

                .. note::
                    The neighbors in the array are not listed in any order of proximity.

            distance_array: An array with dimensions ``(n_samples, k)`` listing the
                distances to each of the k nearest neighbors for each data point.
            l2_std_factor: The multiplier used in calculating the Euclidean distance threshold
                for the distance between two nodes during local pruning. If ``None`` (default),
                then the value is set to the value of ``self.l2_std_factor``.

        Returns:
            A compressed sparse row matrix with dimensions ``(n_samples, n_samples)``,
            containing the pruned distances.
        """

        row_list = []
        col_list = []
        weight_list = []

        n_neighbors = neighbor_array.shape[1]
        n_samples = neighbor_array.shape[0]

        if l2_std_factor is None:
            l2_std_factor = self.l2_std_factor
        else:
            self.l2_std_factor = l2_std_factor

        if self.do_prune_local:
            logger.message(
                "Starting local pruning based on Euclidean distance metric at "
                f"{self.l2_std_factor} standard deviations above the mean"
            )
            distance_array = distance_array + 0.1
            for community_id, neighbors in zip(range(n_samples), neighbor_array):
                distances = distance_array[community_id, :]
                max_distance = np.mean(distances) + self.l2_std_factor * np.std(distances)
                to_keep = np.where(distances < max_distance)[0]
                updated_neighbors = neighbors[np.ix_(to_keep)]
                updated_distances = distances[np.ix_(to_keep)]

                # remove self-loops
                for index in range(len(updated_neighbors)):
                    if community_id != neighbors[index]:
                        row_list.append(community_id)
                        col_list.append(updated_neighbors[index])
                        distance = np.sqrt(updated_distances[index])
                        weight_list.append(1 / (distance + 0.1))
        else:
            row_list.extend(
                list(np.transpose(np.ones((n_neighbors, n_samples)) * range(n_samples)).flatten())
            )
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1.0 / (distance_array.flatten() + 0.1)).tolist()

        csr_array = csr_matrix(
            (np.array(weight_list), (np.array(row_list), np.array(col_list))),
            shape=(n_samples, n_samples)
        )
        return csr_array

    def prune_global(
        self,
        csr_array: csr_matrix,
        jac_threshold_type: str,
        jac_std_factor: float,
        jac_weighted_edges: bool,
        n_samples: int
    ) -> ig.Graph:
        """Prune the graph globally based on the Jaccard similarity measure.

        The ``csr_array`` contains the locally-pruned pairwise distances. From this, we can
        use the Jaccard similarity metric to compute the similarity score for each edge. We then
        remove any edges from the graph that do not meet a minimum similarity threshold.

        Args:
            csr_array: A sparse matrix with dimensions ``(n_samples, n_samples)``,
                containing the locally-pruned pair-wise distances.
            jac_threshold_type: One of ``"median"`` or ``"mean"``. Determines how the
                Jaccard similarity threshold is calculated during global pruning.
            jac_std_factor: The multiplier used in calculating the Jaccard similarity
                threshold for the similarity between two nodes during global pruning for
                ``jac_threshold_type = "mean"``:

                .. code-block:: python

                    threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

                Setting ``jac_std_factor = 0.15`` and ``jac_threshold_type="mean"`` performs
                empirically similar to ``jac_threshold_type="median"``, which does not use the
                ``jac_std_factor``. Generally values between 0-1.5 are reasonable. Higher
                ``jac_std_factor`` means more edges are kept.
            jac_weighted_edges: Whether to weight the pruned graph. This is always ``True`` for
                the top-level ``PARC`` run, but can be changed when pruning the large communities.
            n_samples: The number of samples in the data.

        Returns:
            A ``Graph`` object which has now been locally and globally pruned.
        """

        logger.message("Starting global pruning...")

        input_nodes, output_nodes = csr_array.nonzero()
        edges = list(zip(input_nodes.tolist(), output_nodes.tolist()))
        edges_copy = edges.copy()

        logger.info(f"Creating graph with {len(edges)} edges and {n_samples} nodes...")

        graph = ig.Graph(edges, edge_attrs={"weight": csr_array.data.tolist()})
        similarities = np.asarray(graph.similarity_jaccard(pairs=edges_copy))

        if jac_threshold_type == "median":
            threshold = np.median(similarities)
        else:
            threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)

        indices_similar = np.where(similarities > threshold)[0]

        logger.info(
            f"Pruning {len(edges) - len(indices_similar)} edges based on Jaccard similarity "
            f"threshold of {threshold:.3f} "
            f"(mean = {np.mean(similarities):.3f}, std = {np.std(similarities):.3f}) "
        )
        logger.message(f"Creating graph with {len(indices_similar)} edges and {n_samples} nodes...")

        if jac_weighted_edges:
            graph_pruned = ig.Graph(
                n=n_samples,
                edges=list(np.asarray(edges_copy)[indices_similar]),
                edge_attrs={"weight": list(similarities[indices_similar])}
            )
        else:
            graph_pruned = ig.Graph(
                n=n_samples,
                edges=list(np.asarray(edges_copy)[indices_similar])
            )

        graph_pruned.simplify(combine_edges="sum")  # "first"
        return graph_pruned

    def get_leiden_partition(
        self,
        graph: ig.Graph,
        jac_weighted_edges: bool = True
    ) -> leidenalg.VertexPartition:
        """Partition the graph using the Leiden algorithm.

        A partition is a set of communities.

        Args:
            graph: A ``Graph`` object which has been locally and globally pruned.
            jac_weighted_edges: Whether to partition using the weighted graph.

        Returns:
            A partition object.
            See `leidenalg.VertexPartition on GitHub
            <https://github.com/vtraag/leidenalg/blob/main/src/leidenalg/VertexPartition.py>`_.
        """

        if jac_weighted_edges:
            weights = "weight"
        else:
            weights = None

        if self.partition_type == "ModularityVP":
            logger.message(
                "Leiden algorithm find partition: partition type = ModularityVertexPartition"
            )
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.ModularityVertexPartition,
                weights=weights,
                n_iterations=self.n_iter_leiden,
                seed=self.random_seed
            )
        else:
            logger.message(
                "Leiden algorithm find partition: partition type = RBConfigurationVertexPartition"
            )
            partition = leidenalg.find_partition(
                graph=graph,
                partition_type=leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                n_iterations=self.n_iter_leiden,
                seed=self.random_seed,
                resolution_parameter=self.resolution_parameter
            )
        return partition

    def run_toobig_subPARC(
        self,
        x_data,
        jac_threshold_type: str = "mean",
        jac_std_factor: float = 0.3,
        jac_weighted_edges=True
    ):

        n_samples = x_data.shape[0]
        knn_struct = self.make_knn_struct(
            x_data=x_data,
            knn=self.knn,
            hnsw_param_m=30,
            hnsw_param_ef_construction=200,
            distance_metric="l2"
        )
        if n_samples <= 10:
            logger.message(
                f"Large community is small with only {n_samples} nodes. "
                f"Consider increasing the large_community_factor = {self.large_community_factor}."
            )
        if n_samples > self.knn:
            knnbig = self.knn
        else:
            knnbig = int(max(5, 0.2 * n_samples))

        neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knnbig)
        csr_array = self.prune_local(neighbor_array, distance_array)

        graph_pruned = self.prune_global(
            csr_array=csr_array,
            jac_threshold_type=jac_threshold_type,
            jac_std_factor=jac_std_factor,
            jac_weighted_edges=jac_weighted_edges,
            n_samples=n_samples
        )

        partition = self.get_leiden_partition(
            graph=graph_pruned,
            jac_weighted_edges=jac_weighted_edges
        )

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_samples, 1))
        small_pop_list = []
        small_cluster_list = []
        small_community_exists = False
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        logger.message("Stating small community detection...")
        for cluster in set(node_communities):
            population = len(np.where(node_communities == cluster)[0])
            if population < self.small_community_size:
                small_community_exists = True
                small_pop_list.append(list(np.where(node_communities == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell, :]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group

        time_start = time.time()
        while small_community_exists and (time.time() - time_start < self.small_community_timeout):
            small_pop_list = []
            small_community_exists = False
            for cluster in set(list(node_communities.flatten())):
                population = len(np.where(node_communities == cluster)[0])
                if population < self.small_community_size:
                    small_community_exists = True

                    small_pop_list.append(np.where(node_communities == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell, :]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        return node_communities

    def run_parc(self):
        time_start = time.time()
        x_data = self.x_data
        n_samples = x_data.shape[0]
        n_features = x_data.shape[1]
        logger.message(
            f"Input data has shape {n_samples} (samples) x {n_features} (features)"
        )

        large_community_factor = self.large_community_factor
        small_community_size = self.small_community_size
        jac_threshold_type = self.jac_threshold_type
        jac_std_factor = self.jac_std_factor
        jac_weighted_edges = self.jac_weighted_edges
        knn = self.knn

        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(csr_array.indices, csr_array.indptr)[1:-1]
        else:
            if self.knn_struct is None:
                logger.message("Creating knn_struct...")
                if n_samples < 10000:
                    ef_query = min(n_samples - 10, 500)
                else:
                    ef_query = 100
                self.knn_struct = self.make_knn_struct(
                    x_data=x_data,
                    knn=knn,
                    ef_query=ef_query,
                    distance_metric=self.distance_metric,
                    n_threads=self.n_threads
                )
            else:
                logger.message("knn struct already exists")
            neighbor_array, distance_array = self.knn_struct.knn_query(x_data, k=knn)
            csr_array = self.prune_local(neighbor_array, distance_array)

        graph_pruned = self.prune_global(
            csr_array=csr_array,
            jac_threshold_type=jac_threshold_type,
            jac_std_factor=jac_std_factor,
            jac_weighted_edges=True,
            n_samples=n_samples
        )

        logger.message("Starting Leiden community detection...")
        partition = self.get_leiden_partition(
            graph=graph_pruned,
            jac_weighted_edges=jac_weighted_edges
        )

        node_communities = np.asarray(partition.membership)
        node_communities = np.reshape(node_communities, (n_samples, 1))

        too_big = False

        # The 0th cluster is the largest one.
        # So, if cluster 0 is not too big, then the others won't be too big either
        large_community_id = 0
        community_indices = np.where(node_communities == large_community_id)[0]
        community_size = len(community_indices)

        if community_size > large_community_factor * n_samples:
            logger.message(
                f"\nCommunity 0 is too large and has size:\n"
                f"{community_size} > large_community_factor * n_samples = "
                f"{large_community_factor} * {n_samples} = {large_community_factor * n_samples}\n"
                f"Starting large community expansion..."
            )
            too_big = True
            large_community_indices = community_indices
            list_pop_too_bigs = [community_size]
        else:
            logger.message(
                f"\nCommunity 0 is not too large and has size:\n"
                f"{community_size} <= large_community_factor * n_samples = "
                f"{large_community_factor} * {n_samples} = {large_community_factor * n_samples}\n"
                "Skipping large community expansion."
            )

        while too_big:
            logger.message(f"Expanding large community {large_community_id}...")
            node_communities_big = self.run_toobig_subPARC(
                x_data=x_data[large_community_indices, :]
            )
            node_communities_big = node_communities_big + 100000

            jj = 0
            for j in large_community_indices:
                node_communities[j] = node_communities_big[jj]
                jj = jj + 1
            node_communities = np.unique(
                list(node_communities.flatten()), return_inverse=True
            )[1]

            too_big = False
            node_communities = np.asarray(node_communities)
            for community_id in set(node_communities):
                community_indices = np.where(node_communities == community_id)[0]
                community_size = len(community_indices)
                not_yet_expanded = community_size not in list_pop_too_bigs
                if community_size > large_community_factor * n_samples and not_yet_expanded:
                    too_big = True
                    logger.message(
                        f"Community {community_id} is too big and has population {community_size}."
                    )
                    large_community_indices = community_indices
                    large_community_id = community_id
                    large_community_size = community_size
            if too_big:
                list_pop_too_bigs.append(large_community_size)
                logger.message(
                    f"Community {large_community_id} is too big and has population "
                    f"{large_community_size}. It will be expanded."
                )
        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]

        logger.message("Starting small community detection...")
        small_pop_list = []
        small_cluster_list = []
        small_community_exists = False

        for cluster in set(node_communities):
            population = len(np.where(node_communities == cluster)[0])
            if population < small_community_size:
                logger.message(
                    f"Community {cluster} is a small community with population {population}"
                )
                small_community_exists = True
                small_pop_list.append(list(np.where(node_communities == cluster)[0]))
                small_cluster_list.append(cluster)

        for small_cluster in small_pop_list:

            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = node_communities[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [value for value in group_of_old_neighbors if
                                                 value in list(available_neighbours)]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    node_communities[single_cell] = best_group
        time_start_sc = time.time()
        while small_community_exists and (time.time() - time_start_sc) < self.small_community_timeout:
            small_pop_list = []
            small_community_exists = False
            for cluster in set(list(node_communities.flatten())):
                population = len(np.where(node_communities == cluster)[0])
                if population < small_community_size:
                    logger.info(
                        f"Community {cluster} is a small community with population {population}"
                    )
                    small_community_exists = True
                    small_pop_list.append(np.where(node_communities == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = node_communities[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    node_communities[single_cell] = best_group

        node_communities = np.unique(list(node_communities.flatten()), return_inverse=True)[1]
        node_communities = list(node_communities.flatten())
        pop_list = []
        for item in set(node_communities):
            pop_list.append((int(item), node_communities.count(item)))
        logger.message(f"Community labels and sizes: {pop_list}")

        self.y_data_pred = node_communities
        run_time = time.time() - time_start
        logger.message(f"Time elapsed to run PARC: {run_time:.1f} seconds")
        self.compute_performance_metrics(run_time)

    def accuracy(self, target=1):

        y_data_true = self.y_data_true
        Index_dict = {}
        y_data_pred = self.y_data_pred
        n_samples = len(y_data_pred)
        n_target = list(y_data_true).count(target)
        n_pbmc = n_samples - n_target

        for k in range(n_samples):
            Index_dict.setdefault(y_data_pred[k], []).append(y_data_true[k])
        num_groups = len(Index_dict)
        sorted_keys = list(sorted(Index_dict.keys()))
        error_count = []
        negative_labels = []
        positive_labels = []
        fp, fn, tp, tn, precision, recall, f1_score = 0, 0, 0, 0, 0, 0, 0

        for kk in sorted_keys:
            vals = [t for t in Index_dict[kk]]
            majority_val = get_mode(vals)
            if majority_val == target:
                logger.info(f"Cluster {kk} has majority {target} with population {len(vals)}")
            if kk == -1:
                len_unknown = len(vals)
                logger.info(f"len unknown: {len_unknown}")
            if (majority_val == target) and (kk != -1):
                positive_labels.append(kk)
                fp = fp + len([e for e in vals if e != target])
                tp = tp + len([e for e in vals if e == target])
                list_error = [e for e in vals if e != majority_val]
                e_count = len(list_error)
                error_count.append(e_count)
            elif (majority_val != target) and (kk != -1):
                negative_labels.append(kk)
                tn = tn + len([e for e in vals if e != target])
                fn = fn + len([e for e in vals if e == target])
                error_count.append(len([e for e in vals if e != majority_val]))

        predict_class_array = np.array(y_data_pred)
        y_data_pred_array = np.array(y_data_pred)
        number_clusters_for_target = len(positive_labels)
        for cancer_class in positive_labels:
            predict_class_array[y_data_pred_array == cancer_class] = 1
        for benign_class in negative_labels:
            predict_class_array[y_data_pred_array == benign_class] = 0
        predict_class_array.reshape((predict_class_array.shape[0], -1))
        error_rate = sum(error_count) / n_samples
        n_target = tp + fn
        tnr = tn / n_pbmc
        fnr = fn / n_target
        tpr = tp / n_target
        fpr = fp / n_pbmc

        if tp != 0 or fn != 0:
            recall = tp / (tp + fn)  # ability to find all positives
        if tp != 0 or fp != 0:
            precision = tp / (tp + fp)  # ability to not misclassify negatives as positives
        if precision != 0 or recall != 0:
            f1_score = precision * recall * 2 / (precision + recall)
        majority_truth_labels = np.empty((len(y_data_true), 1), dtype=object)

        for community_id in set(y_data_pred):
            community_indices = np.where(np.asarray(y_data_pred) == community_id)[0]
            y_data_true = np.asarray(y_data_true)
            majority_truth = get_mode(list(y_data_true[community_indices]))
            majority_truth_labels[community_indices] = majority_truth

        majority_truth_labels = list(majority_truth_labels.flatten())
        accuracy_val = [error_rate, f1_score, tnr, fnr, tpr, fpr, precision,
                        recall, num_groups, n_target]

        return accuracy_val, predict_class_array, majority_truth_labels, number_clusters_for_target

    def compute_performance_metrics(self, run_time: float):
        """Compute performance metrics for the PARC algorithm.

        Args:
            run_time: (float) the time taken to run the PARC algorithm.
        """
        list_roc = []
        targets = list(set(self.y_data_true))
        n_samples = len(list(self.y_data_true))
        self.f1_accumulated = 0
        self.f1_mean = 0
        self.stats_df = pd.DataFrame({
            "jac_std_factor": [self.jac_std_factor],
            "l2_std_factor": [self.l2_std_factor],
            "runtime(s)": [run_time]
        })
        self.majority_truth_labels = []
        if len(targets) > 1:
            f1_accumulated = 0
            f1_acc_noweighting = 0
            for target in targets:
                logger.info(f"Target is {target}")
                vals_roc, predict_class_array, majority_truth_labels, numclusters_targetval = \
                    self.accuracy(target=target)
                f1_current = vals_roc[1]
                logger.info(f"Target {target} has f1-score of {(f1_current * 100):.2f}")
                f1_accumulated = f1_accumulated + \
                    f1_current * (list(self.y_data_true).count(target)) / n_samples
                f1_acc_noweighting = f1_acc_noweighting + f1_current

                list_roc.append(
                    [self.jac_std_factor, self.l2_std_factor, target] +
                    vals_roc +
                    [numclusters_targetval] +
                    [run_time]
                )

            f1_mean = f1_acc_noweighting / len(targets)

            df_accuracy = pd.DataFrame(
                list_roc,
                columns=[
                    "jac_std_factor", "l2_std_factor", "target", "error rate",
                    "f1-score", "tnr", "fnr", "tpr", "fpr", "precision", "recall", "num_groups",
                    "target population", "num clusters", "clustering runtime"
                ]
            )

            logger.message(f"f1-score (unweighted) mean: {(f1_mean * 100):.2f}")
            logger.message(f"f1-score weighted (by population): {(f1_accumulated * 100):.2f}")
            logger.message(
                f"\n{df_accuracy[['target', 'f1-score', 'target population', 'num clusters']]}"
            )

            self.f1_accumulated = f1_accumulated
            self.f1_mean = f1_mean
            self.stats_df = df_accuracy
            self.majority_truth_labels = majority_truth_labels

    def run_umap_hnsw(
            self,
            x_data,
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
            x_data: (array) an array containing the input data, with shape n_samples x n_features.
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
            data=x_data,
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
