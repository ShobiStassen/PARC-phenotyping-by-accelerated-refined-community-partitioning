import pytest
from sklearn import datasets
import igraph as ig
import numpy as np
from parc._parc import PARC
from parc.logger import get_logger

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


@pytest.mark.parametrize("knn", [5, 10, 20, 50])
def test_parc_create_knn_graph(iris_data, knn):
    x_data = iris_data[0]
    y_data = iris_data[1]

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.knn_struct = parc_model.make_knn_struct()
    csr_array = parc_model.create_knn_graph(knn=knn)
    nn_collection = np.split(csr_array.indices, csr_array.indptr)[1:-1]
    assert len(nn_collection) == y_data.shape[0]
