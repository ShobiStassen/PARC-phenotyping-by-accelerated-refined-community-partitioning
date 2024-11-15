import numpy as np
from scipy.sparse import csr_matrix
from umap.umap_ import simplicial_set_embedding, find_ab_params
from parc.logger import get_logger

logger = get_logger(__name__)


def run_umap_hnsw(
    x_data: np.ndarray,
    graph: csr_matrix,
    n_components: int = 2,
    alpha: float = 1.0,
    negative_sample_rate: int = 5,
    gamma: float = 1.0,
    spread: float = 1.0,
    min_dist: float = 0.1,
    n_epochs: int = 0,
    init_pos: str = "spectral",
    random_state_seed: int = 1,
    densmap: bool = False,
    densmap_kwds: dict = {},
    output_dens: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """Perform a fuzzy simplicial set embedding, using a specified initialisation method and
    then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and
    low-dimensional fuzzy simplicial sets.

    See `umap.umap_ simplicial_set_embedding
    <https://github.com/lmcinnes/umap/blob/master/umap/umap_.py>`__.

    Args:
        x_data: an array containing the input data, with shape n_samples x n_features.
        graph: the 1-skeleton of the high dimensional fuzzy simplicial set as
            represented by a graph for which we require a sparse matrix for the
            (weighted) adjacency matrix.
        n_components: the dimensionality of the Euclidean space into which the data
            will be embedded.
        alpha: the initial learning rate for stochastic gradient descent (SGD).
        negative_sample_rate: the number of negative samples to select per positive
            sample in the optimization process. Increasing this value will result in
            greater repulsive force being applied, greater optimization cost, but
            slightly more accuracy.
        gamma: weight to apply to negative samples.
        spread: the upper range of the x-value to be used to fit a curve for the
            low-dimensional fuzzy simplicial complex construction. This curve should be
            close to an offset exponential decay. Must be greater than 0. See
            ``umap.umap_.find_ab_params``.
        min_dist: See ``umap.umap_.find_ab_params``.
        n_epochs: the number of training epochs to be used in optimizing the
            low-dimensional embedding. Larger values result in more accurate embeddings.
            If 0 is specified, a value will be selected based on the size of the input dataset
            (200 for large datasets, 500 for small).
        init_pos: how to initialize the low-dimensional embedding. One of:
            `spectral`: use a spectral embedding of the fuzzy 1-skeleton
            `random`: assign initial embedding positions at random
            `pca`: use the first n components from Principal Component Analysis (PCA)
                applied to the input data.
        random_state_seed: an integer to pass as a seed for the Numpy RandomState.
        densmap: whether to use the density-augmented objective function to optimize
            the embedding according to the densMAP algorithm.
        densmap_kwds: keyword arguments to be used by the densMAP optimization.
        output_dens: whether to output local radii in the original data
            and the embedding.

    Returns:
        An array of shape ``(n_samples, n_components)``. This array gives the
        coordinates of the data in the embedding space. The embedding space is
        a Euclidean space with dimensionality equal to ``n_components``.
    """

    a, b = find_ab_params(spread, min_dist)
    logger.message(f"a: {a}, b: {b}, spread: {spread}, dist: {min_dist}")

    umap_embedding = simplicial_set_embedding(
        data=x_data,
        graph=graph,
        n_components=n_components,
        initial_alpha=alpha,
        a=a,
        b=b,
        n_epochs=n_epochs,
        metric_kwds={},
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init=init_pos,
        random_state=np.random.RandomState(random_state_seed),
        metric="euclidean",
        verbose=verbose,
        densmap=densmap,
        densmap_kwds=densmap_kwds,
        output_dens=output_dens
    )
    return umap_embedding[0]