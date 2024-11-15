import pytest
from sklearn import datasets
import igraph as ig
import numpy as np
import pathlib
import json
import scipy
import igraph
import hnswlib
from parc._parc import PARC
from parc.logger import get_logger
from tests.variables import NEIGHBOR_ARRAY_L2, NEIGHBOR_ARRAY_COSINE
from tests.utils import __tmp_dir__, create_tmp_dir, remove_tmp_dir

logger = get_logger(__name__, 20)


def setup_function():
    create_tmp_dir()


@pytest.fixture
def iris_data():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target
    return x_data, y_data


@pytest.fixture
def forest_data():
    forests = datasets.fetch_covtype()
    x_data = forests.data[list(range(0, 30000)), :]
    y_data = forests.target[list(range(0, 30000))]
    return x_data, y_data


def test_parc_run_umap_hnsw():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.fit_predict()

    graph = parc_model.create_knn_graph()
    x_umap = parc_model.run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)


@pytest.mark.parametrize("knn", [5, 10, 20, 50])
@pytest.mark.parametrize("jac_weighted_edges", [True, False])
def test_parc_get_leiden_partition(iris_data, knn, jac_weighted_edges):
    x_data = iris_data[0]
    y_data = iris_data[1]

    parc_model = PARC(x_data, y_data_true=y_data)
    hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
    neighbor_array, distance_array = hnsw_index.knn_query(x_data, k=knn)
    csr_array = parc_model.prune_local(neighbor_array, distance_array)

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
def test_parc_create_hnsw_index(
    iris_data, knn_hnsw, knn_query, distance_metric, expected_neighbor_array
):
    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    hnsw_index = parc_model.create_hnsw_index(
        x_data=x_data,
        knn=knn_hnsw,
        distance_metric=distance_metric
    )
    assert isinstance(hnsw_index, hnswlib.Index)
    neighbor_array, distance_array = hnsw_index.knn_query(x_data, k=knn_query)
    assert neighbor_array.shape == (x_data.shape[0], knn_query)
    assert distance_array.shape == (x_data.shape[0], knn_query)
    n_diff = np.count_nonzero(expected_neighbor_array != neighbor_array)
    assert n_diff / x_data.shape[0] < 0.1


@pytest.mark.parametrize("knn", [5, 10, 20, 50])
def test_parc_create_knn_graph(iris_data, knn):
    x_data = iris_data[0]
    y_data = iris_data[1]

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
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
    hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
    neighbor_array, distance_array = hnsw_index.knn_query(x_data, k=knn)
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
    hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
    neighbor_array, distance_array = hnsw_index.knn_query(x_data, k=knn)
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


@pytest.mark.parametrize(
    "large_community_factor",
    [
        (1.0),
        (0.5),
        (0.08)
    ]
)
def test_parc_large_community_expansion(
    iris_data, large_community_factor
):
    x_data = iris_data[0]
    y_data = iris_data[1]
    node_communities = np.random.randint(0, 10, x_data.shape[0])
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    node_communities_expanded = parc_model.large_community_expansion(
        x_data=x_data,
        node_communities=node_communities.copy(),
        large_community_factor=large_community_factor,
    )
    if large_community_factor == 1.0:
        assert np.all(node_communities_expanded == node_communities)
    else:
        assert len(np.unique(node_communities_expanded)) >= len(np.unique(node_communities))
        np.testing.assert_array_equal(
            np.unique(node_communities_expanded),
            range(len(np.unique(node_communities_expanded)))
        )


@pytest.mark.parametrize(
    "dataset_name, node_communities, small_community_size, expected_small_communities",
    [
        ("iris_data", np.random.choice([0], 150), 10, {}),
        ("iris_data", np.array([0] * 130 + [1] * 20), 50, {1: np.array(range(130, 150))}),
        ("iris_data", np.array([0] * 130 + [1] * 20), 10, {}),
        (
            "iris_data", np.array([0] * 50 + [1] * 50 + [2] * 50), 60,
            {0: np.array(range(0, 50)), 1: np.array(range(50, 100)), 2: np.array(range(100, 150))}
        ),
    ]
)
def test_parc_get_small_communities(
    request, dataset_name, node_communities, small_community_size, expected_small_communities,
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    small_communities = parc_model.get_small_communities(
        node_communities=node_communities.copy(),
        small_community_size=small_community_size
    )
    assert len(small_communities) == len(expected_small_communities)
    for community_id, small_community in small_communities.items():
        assert len(small_community) <= small_community_size
        assert np.all(small_community == expected_small_communities[community_id])


@pytest.mark.parametrize(
    (
        "dataset_name, node_communities, knn, small_communities,"
        "allow_small_to_small, expected_node_communities"
    ),
    [
        (
            "iris_data",
            np.array([0] * 130 + [1] * 20),
            5,
            {1: np.array(range(130, 150))},
            False,
            np.array([0] * 150)
        ),
        (
            "iris_data",
            np.array([0] * 50 + [1] * 50 + [2] * 50),
            5,
            {0: np.array(range(0, 50)), 1: np.array(range(50, 100)), 2: np.array(range(100, 150))},
            False,
            np.array([0] * 50 + [1] * 50 + [2] * 50)
        ),
    ]
)
def test_parc_reassign_small_communities(
    request, dataset_name, node_communities, knn, small_communities,
    allow_small_to_small, expected_node_communities
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
    neighbor_array, _ = hnsw_index.knn_query(x_data, k=knn)
    node_communities_reassigned = parc_model.reassign_small_communities(
        node_communities=node_communities.copy(),
        small_communities=small_communities,
        neighbor_array=neighbor_array,
        allow_small_to_small=allow_small_to_small
    )
    np.testing.assert_array_equal(node_communities_reassigned, expected_node_communities)


@pytest.mark.parametrize(
    "dataset_name, node_communities, small_community_size, expected_node_communities",
    [
        ("iris_data", np.random.choice([0], 150), 10, np.random.choice([0], 150)),
        ("iris_data", np.array([0] * 130 + [1] * 20), 50, np.array([0] * 150)),
        ("iris_data", np.array([0] * 130 + [1] * 20), 10, np.array([0] * 130 + [1] * 20)),
        ("iris_data", np.array([0] * 50 + [1] * 50 + [2] * 50), 60, None),
    ]
)
@pytest.mark.parametrize(
    "small_community_timeout",
    [15]
)
@pytest.mark.parametrize(
    "knn",
    [5, 10]
)
def test_parc_small_community_merging(
    request, dataset_name, node_communities, small_community_size, expected_node_communities,
    small_community_timeout, knn
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    parc_model = PARC(x_data=x_data, y_data_true=y_data)
    hnsw_index = parc_model.create_hnsw_index(x_data=x_data, knn=knn)
    neighbor_array, _ = hnsw_index.knn_query(x_data, k=knn)
    node_communities_merged = parc_model.small_community_merging(
        node_communities=node_communities.copy(),
        small_community_size=small_community_size,
        small_community_timeout=small_community_timeout,
        neighbor_array=neighbor_array
    )
    assert len(np.unique(node_communities_merged)) <= len(np.unique(node_communities))
    assert set(np.unique(node_communities_merged)) <= set(np.unique(node_communities))
    if expected_node_communities is not None:
        np.testing.assert_array_equal(node_communities_merged, expected_node_communities)


@pytest.mark.parametrize(
    (
        "dataset_name, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,"
        "l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,"
        "large_community_factor, small_community_size, small_community_timeout,"
        "resolution_parameter, partition_type,"
        "f1_mean, f1_accumulated"
    ),
    [
        (
            "iris_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.4, 10, 15, 1.0,
            "ModularityVP", 0.9, 0.9
        ),
        (
            "iris_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.15, 10, 15, 1.0,
            "ModularityVP", 0.9, 0.9
        )
    ]
)
@pytest.mark.parametrize(
    "targets_exist",
    [
        (True),
        (False)
    ]
)
def test_parc_fit_predict_fast(
    request, dataset_name, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,
    l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,
    large_community_factor, small_community_size, small_community_timeout,
    resolution_parameter, partition_type, f1_mean, f1_accumulated, targets_exist
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    if not targets_exist:
        y_data = None

    parc_model = PARC(
        x_data=x_data,
        y_data_true=y_data,
        knn=knn,
        n_iter_leiden=n_iter_leiden,
        distance_metric=distance_metric,
        hnsw_param_ef_construction=hnsw_param_ef_construction,
        l2_std_factor=l2_std_factor,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        do_prune_local=do_prune_local,
        large_community_factor=large_community_factor,
        small_community_size=small_community_size,
        small_community_timeout=small_community_timeout,
        resolution_parameter=resolution_parameter,
        partition_type=partition_type
    )

    parc_model.fit_predict()
    if targets_exist:
        assert parc_model.f1_mean >= f1_mean
        assert parc_model.f1_accumulated >= f1_accumulated
    else:
        assert parc_model.f1_mean == 0
        assert parc_model.f1_accumulated == 0
    assert len(parc_model.y_data_pred) == x_data.shape[0]


@pytest.mark.parametrize(
    (
        "dataset_name, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,"
        "l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,"
        "large_community_factor, small_community_size, small_community_timeout,"
        "resolution_parameter, partition_type,"
        "f1_mean, f1_accumulated"
    ),
    [
        (
            "iris_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.4, 10, 15, 1.0,
            "ModularityVP", 0.9, 0.9
        ),
        (
            "iris_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.15, 10, 15, 1.0,
            "ModularityVP", 0.9, 0.9
        ),
        (
            "iris_data", 2, 10, "l2", 150, 3.0, "median", 0.15, True, True, 0.4, 10, 15, 1.0,
            "ModularityVP", 0.9, 0.9
        ),
        (
            "forest_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.019, 10, 15, 1.0,
            "ModularityVP", 0.6, 0.7
        ),
        (
            "forest_data", 30, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.4, 10, 15, 1.0,
            "ModularityVP", 0.6, 0.7
        ),
        (
            "forest_data", 100, 5, "l2", 150, 3.0, "median", 0.15, True, True, 0.4, 10, 15, 1.0,
            "ModularityVP", 0.5, 0.6
        )
    ]
)
@pytest.mark.parametrize(
    "targets_exist",
    [
        (True),
        (False)
    ]
)
def test_parc_fit_predict_full(
    request, dataset_name, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,
    l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,
    large_community_factor, small_community_size, small_community_timeout,
    resolution_parameter, partition_type, f1_mean, f1_accumulated, targets_exist
):
    x_data, y_data = request.getfixturevalue(dataset_name)
    if not targets_exist:
        y_data = None

    parc_model = PARC(
        x_data=x_data,
        y_data_true=y_data,
        knn=knn,
        n_iter_leiden=n_iter_leiden,
        distance_metric=distance_metric,
        hnsw_param_ef_construction=hnsw_param_ef_construction,
        l2_std_factor=l2_std_factor,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        do_prune_local=do_prune_local,
        large_community_factor=large_community_factor,
        small_community_size=small_community_size,
        small_community_timeout=small_community_timeout,
        resolution_parameter=resolution_parameter,
        partition_type=partition_type
    )

    parc_model.fit_predict()
    if targets_exist:
        assert parc_model.f1_mean >= f1_mean
        assert parc_model.f1_accumulated >= f1_accumulated
    else:
        assert parc_model.f1_mean == 0
        assert parc_model.f1_accumulated == 0
    assert len(parc_model.y_data_pred) == x_data.shape[0]


@pytest.mark.parametrize(
    (
        "file_path, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,"
        "l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,"
        "large_community_factor, small_community_size, small_community_timeout,"
        "resolution_parameter, partition_type"
    ),
    [
        (
            pathlib.Path(__tmp_dir__, "parc_model.json"), 30, 5, "l2", 150, 3.0,
            "median", 0.15, True, True, 0.4, 10, 15, 1.0, "ModularityVP"
        )
    ]
)
def test_parc_save(
    iris_data, file_path, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,
    l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,
    large_community_factor, small_community_size, small_community_timeout,
    resolution_parameter, partition_type
):

    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(
        x_data=x_data,
        y_data_true=y_data,
        knn=knn,
        n_iter_leiden=n_iter_leiden,
        distance_metric=distance_metric,
        hnsw_param_ef_construction=hnsw_param_ef_construction,
        l2_std_factor=l2_std_factor,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        do_prune_local=do_prune_local,
        large_community_factor=large_community_factor,
        small_community_size=small_community_size,
        small_community_timeout=small_community_timeout,
        resolution_parameter=resolution_parameter,
        partition_type=partition_type
    )
    parc_model.fit_predict()
    parc_model.save(file_path)
    assert file_path.exists()
    with open(file_path) as file:
        model_dict = json.load(file)

    for key, value in model_dict.items():
        assert getattr(parc_model, key) == value


@pytest.mark.parametrize(
    (
        "file_path, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,"
        "l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,"
        "large_community_factor, small_community_size, small_community_timeout,"
        "resolution_parameter, partition_type"
    ),
    [
        (
            pathlib.Path(__tmp_dir__, "parc_model.json"), 30, 5, "l2", 150, 3.0,
            "median", 0.15, True, True, 0.4, 10, 15, 1.0, "ModularityVP"
        )
    ]
)
def test_parc_load(
    iris_data, file_path, knn, n_iter_leiden, distance_metric, hnsw_param_ef_construction,
    l2_std_factor, jac_threshold_type, jac_std_factor, jac_weighted_edges, do_prune_local,
    large_community_factor, small_community_size, small_community_timeout,
    resolution_parameter, partition_type
):

    x_data = iris_data[0]
    y_data = iris_data[1]
    parc_model = PARC(
        x_data=x_data,
        y_data_true=y_data,
        knn=knn,
        n_iter_leiden=n_iter_leiden,
        distance_metric=distance_metric,
        hnsw_param_ef_construction=hnsw_param_ef_construction,
        l2_std_factor=l2_std_factor,
        jac_threshold_type=jac_threshold_type,
        jac_std_factor=jac_std_factor,
        jac_weighted_edges=jac_weighted_edges,
        do_prune_local=do_prune_local,
        large_community_factor=large_community_factor,
        small_community_size=small_community_size,
        small_community_timeout=small_community_timeout,
        resolution_parameter=resolution_parameter,
        partition_type=partition_type
    )
    parc_model.fit_predict()
    parc_model.save(file_path)

    parc_model_saved = PARC(
        x_data=x_data,
        y_data_true=y_data,
        file_path=file_path
    )

    with open(file_path) as file:
        model_dict = json.load(file)

    for key, value in model_dict.items():
        assert getattr(parc_model_saved, key) == value



def teardown_function():
    remove_tmp_dir()
