from sklearn import datasets
from parc._parc import PARC


def test_parc_run_umap_hnsw():
    iris = datasets.load_iris()
    x_data = iris.data
    y_data = iris.target

    parc_model = PARC(x_data, y_data_true=y_data)
    parc_model.run_PARC()

    graph = parc_model.knngraph_full()
    x_umap = parc_model.run_umap_hnsw(x_data, graph)
    assert x_umap.shape == (150, 2)
