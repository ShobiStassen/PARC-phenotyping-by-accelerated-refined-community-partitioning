import pytest
from sklearn import datasets
from parc._parc import PARC
from parc.umap_hnsw import run_umap_hnsw
from parc.logger import get_logger
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


def test_parc_run_umap_hnsw():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.fit_predict()

    graph = parc_model.create_knn_graph()
    x_umap = run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)


def teardown_function():
    remove_tmp_dir()