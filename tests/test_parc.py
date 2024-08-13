import pytest
from sklearn import datasets
import igraph as ig
import numpy as np
import scipy
import igraph
import hnswlib
from parc._parc import PARC
from parc.logger import get_logger
from tests.variables import NEIGHBOR_ARRAY_L2, NEIGHBOR_ARRAY_COSINE

logger = get_logger(__name__, 20)


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target
    return x_data, y_data


def test_parc_run_umap_hnsw():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.run_PARC()

    graph = parc_model.create_knn_graph()
    x_umap = parc_model.run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)


@pytest.mark.parametrize("knn", [5, 10, 20, 50])
@pytest.mark.parametrize("jac_weighted_edges", [True, False])
def test_parc_get_leiden_partition(iris_data, knn, jac_weighted_edges):
    x_data = iris_data[0]
    y_data = iris_data[1]

    parc_model = PARC(x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct()
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    csr_array = parc_model.make_csrmatrix_noselfloop(neighbor_array, distance_array)

    input_nodes, output_nodes = csr_array.nonzero()

    edges = list(zip(input_nodes, output_nodes))

    graph = ig.Graph(edges, edge_attrs={"weight": csr_array.data.tolist()})

    leiden_partition = parc_model.get_leiden_partition(
        graph=graph, jac_weighted_edges=jac_weighted_edges
    )

    assert len(leiden_partition.membership) == len(y_data)
    assert len(leiden_partition) <= len(y_data)
    assert len(leiden_partition) >= 1


@pytest.mark.parametrize(
    "knn_hnsw, knn_query, distance_metric, expected_neighbor_array",
    [
        (30, 2, "l2", NEIGHBOR_ARRAY_L2),
        (30, 2, "cosine", NEIGHBOR_ARRAY_COSINE)
    ]
)
def test_parc_make_knn_struct(
    iris_data, knn_hnsw, knn_query, distance_metric, expected_neighbor_array
):
    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct(
        x_data=x_data,
        knn=knn_hnsw,
        distance_metric=distance_metric
    )
    assert isinstance(knn_struct, hnswlib.Index)
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn_query)
    assert neighbor_array.shape == (x_data.shape[0], knn_query)
    assert distance_array.shape == (x_data.shape[0], knn_query)
    n_diff = np.count_nonzero(expected_neighbor_array != neighbor_array)
    assert n_diff / x_data.shape[0] < 0.1


@pytest.mark.parametrize("knn", [5, 10, 20, 50])
def test_parc_create_knn_graph(iris_data, knn):
    x_data = iris_data[0]
    y_data = iris_data[1]

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.knn_struct = parc_model.make_knn_struct()
    csr_array = parc_model.create_knn_graph(knn=knn)
    nn_collection = np.split(csr_array.indices, csr_array.indptr)[1:-1]
    assert len(nn_collection) == y_data.shape[0]


@pytest.mark.parametrize(
    "knn, l2_std_factor, n_edges",
    [
        (100, 3.0, 14625),
        (100, 100.0, 14850),
        (100, -100.0, 0)
    ]
)
def test_parc_prune_local(
    iris_data, knn, l2_std_factor, n_edges
):
    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct()
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    csr_array = parc_model.prune_local(neighbor_array, distance_array, l2_std_factor)
    input_nodes, output_nodes = csr_array.nonzero()
    edges = list(zip(input_nodes, output_nodes))
    assert isinstance(csr_array, scipy.sparse.csr_matrix)
    assert len(edges) == n_edges


@pytest.mark.parametrize(
    "knn, jac_threshold_type, jac_std_factor, jac_weighted_edges, n_edges",
    [
        (100, "median", 0.3, True, 3679),
        (100, "mean", 0.3, False, 3865),
        (100, "mean", -1000, False, 0),
        (100, "mean", 1000, False, 8558)
    ]
)
def test_parc_prune_global(
    iris_data, knn, jac_threshold_type, jac_std_factor, jac_weighted_edges, n_edges
):
    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    knn_struct = parc_model.make_knn_struct()
    neighbor_array, distance_array = knn_struct.knn_query(x_data, k=knn)
    csr_array = parc_model.prune_local(neighbor_array, distance_array)
    graph_pruned = parc_model.prune_global(
        csr_array=csr_array,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        n_samples=x_data.shape[0]
    )
    assert isinstance(graph_pruned, igraph.Graph)
    assert graph_pruned.ecount() == n_edges
    assert graph_pruned.vcount() == x_data.shape[0]
