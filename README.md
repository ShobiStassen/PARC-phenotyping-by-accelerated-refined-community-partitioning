# PARC
PARC, “phenotyping by accelerated refined community-partitioning” - is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm. [PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells](https://academic.oup.com/bioinformatics/article/36/9/2778/5714737).

Check out **[Readthedocs](https://parc.readthedocs.io/en/latest/index.html)** for An [installation guide](https://parc.readthedocs.io/en/latest/Installation.html), [examples](https://parc.readthedocs.io/en/latest/Examples.html) on different data and [tutorials](https://parc.readthedocs.io/en/latest/Notebook-covid19.html).  

:eight_spoked_asterisk: **PARC** forms the clustering basis for our new Trajectory Inference (TI) method **VIA** available on [Github](https://github.com/ShobiStassen/VIA) or [readthedocs](https://parc.readthedocs.io/en/latest/index.html). VIA is a single-cell TI method that offers topology construction and visualization, pseudotimes, automated prediction of terminal cell fates and temporal gene dynamics along detected lineages.
**VIA can also be used to topologically visualize the graph-based connectivity of clusters found by PARC in a non-TI context.**



### Installation
**Using pip**
```
conda create --name ParcEnv pip
pip install parc // tested on linux
```
**Cloning repository and running setup.py** (ensure dependencies are installed)
```
git clone https://github.com/ShobiStassen/PARC.git
python3 setup.py install // cd into the directory of the cloned PARC folder containing setup.py and issue this command
```

**install dependencies separately if needed (linux)**
If the pip install doesn't work, it usually suffices to first install all the requirements (using pip) and subsequently install parc (also using pip)
```
pip install igraph, leidenalg, hnswlib, umap-learn
pip install parc
```

**Windows installation**
Once you have Visual Studio installed it should be smooth sailing (sometimes requires a restart after intalling VS). It might be easier to install dependences using either pip or conda -c conda-forge install. If this doesn't work then you might need to consider using binaries to install igraph and leidenalg.

python-igraph: download the python36 Windows Binaries by [Gohlke](http://www.lfd.uci.edu/~gohlke/pythonlibs)
leidenalg: depends on python-igraph. download [windows binary](https://pypi.org/project/leidenalg/#files) available for python3.6 only

```
conda create --name parcEnv python=3.6 pip
pip install igraph #(or install python_igraph-0.7.1.post6-cp36-cp36m-win_amd64.whl )
pip install leidenalg #(or install leidenalg-0.7.0-cp36-cp36m-win_amd64.whl)
pip install hnswlib
pip install parc
```
## Example Usage on Covid-19 scRNA-seq data

Check out the [Jupyter Notebook](https://parc.readthedocs.io/en/latest/Notebook-covid19.html) for how to pre-process and PARC cluster the new Covid-19 BALF dataset by [Liao et. al 2020](https://www.nature.com/articles/s41591-020-0901-9).
We also show how to integrate UMAP with HNSW such that the embedding in UMAP is constructed using the HNSW graph built in PARC, enabling a very fast and memory efficient viusalization (particularly noticeable when n_cells > 1 Million)

### PARC Cluster-level average gene expression
![](https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_matrixplot.png)

### PARC visualizes cells by integrating UMAP embedding on the HNSW graph
![](https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_hnsw_umap.png)

## Example Usage 1. (small test sets) - IRIS and Digits dataset from sklearn

```
import parc
from parc.umap_hnsw import run_umap_hnsw
import matplotlib.pyplot as plt
from sklearn import datasets

// load sample IRIS data
//data (n_obs x k_dim, 150x4)
iris = datasets.load_iris()
X = iris.data
y=iris.target

plt.scatter(X[:,0],X[:,1], c = y) // colored by 'ground truth'
plt.show()

Parc1 = parc.PARC(X,true_label=y) // instantiate PARC
//Parc1 = parc.PARC(X) // when no 'true labels' are available
Parc1.run_PARC() // run the clustering
parc_labels = Parc1.labels
# View scatterplot colored by PARC labels
plt.scatter(X[:, 0], X[:, 1], c=parc_labels, cmap='rainbow')
plt.show()

# Run umap on the HNSW knngraph already built in PARC (more time and memory efficient for large datasets)
// Parc1.hnsw_index = p1.create_hnsw_index() // if you choose to visualize before running PARC clustering. then you need to include this line
graph = Parc1.knngraph_full()
X_umap = run_umap_hnsw(X, graph)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Parc1.labels)
plt.show()





// load sample digits data
digits = datasets.load_digits()
X = digits.data // (n_obs x k_dim, 1797x64)
y = digits.target
Parc2 = parc.PARC(X,true_label=y, jac_std_global='median') // 'median' is default pruning level
Parc2.run_PARC()
parc_labels = Parc2.labels

```
## Example Usage 2. (mid-scale scRNA-seq): 10X PBMC (Zheng et al., 2017)
[pre-processed datafile](https://drive.google.com/file/d/1H4gOZ09haP_VPCwsYxZt4vf3hJ1GZj3b/view?usp=sharing)

[annotations](https://github.com/ShobiStassen/PARC/blob/master/Datasets/zheng17_annotations.txt)

```
import parc
import csv
import numpy as np
import pandas as pd

## load data (50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)

X = csv.reader(open("./pca50_pbmc68k.txt", 'rt'),delimiter = ",")
X = np.array(list(X)) // (n_obs x k_dim, 68579 x 50)
X = X.astype("float")
// OR with pandas as: X = pd.read_csv("'./pca50_pbmc68k.txt", header=None).values.astype("float")

y = [] // annotations
with open('./zheng17_annotations.txt', 'rt') as f:
    for line in f: y.append(line.strip().replace('\"', ''))
// OR with pandas as: y =  list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])   

// setting small_pop to 50 cleans up some of the smaller clusters, but can also be left at the default 10
parc1 = parc.PARC(X,true_label=y,jac_std_global=0.15, random_seed =1, small_pop = 50) // instantiate PARC
parc1.run_PARC() // run the clustering
parc_labels = parc1.labels
```
![](Images/10X_PBMC_PARC_andGround.png) tsne plot of annotations and PARC clustering

## Example Usage 3. 10X PBMC (Zheng et al., 2017) integrating Scanpy pipeline

[raw datafile 68K pbmc from github page](https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis)

[10X compressed file "filtered genes"](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz)

```
pip install scanpy
```

```
import scanpy.api as sc
import pandas as pd
//load data
path = './data/filtered_matrices_mex/hg19/'
adata = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]

// annotations as per correlation with pure samples
annotations = list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])
adata.obs['annotations'] = pd.Categorical(annotations)

//pre-process as per Zheng et al., and take first 50 PCs for analysis
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, n_comps=50)
// setting small_pop to 50 cleans up some of the smaller clusters, but can also be left at the default 10
parc1 = parc.PARC(adata.obsm['X_pca'], true_label = annotations, jac_std_global=0.15, random_seed =1, small_pop = 50)  
parc1.run_PARC() // run the clustering
parc_labels = parc1.labels
adata.obs["PARC"] = pd.Categorical(parc_labels)

//visualize
sc.settings.n_jobs=4
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, color='annotations')
sc.pl.umap(adata, color='PARC')
```
## Example Usage 4. Large-scale (70K subset and 1.1M cells) Lung Cancer cells (multi-ATOM imaging cytometry based features)

[normalized image-based feature matrix 70K cells](https://drive.google.com/open?id=1LeFjxGlaoaZN9sh0nuuMFBK0bvxPiaUz)

[Lung Cancer cells annotation 70K cells](https://drive.google.com/open?id=1iwXQkdwEwplhZ1v0jYWnu2CHziOt_D9C)

[Lung Cancer Digital Spike Test of n=100 H1975 cells on N281604 ](https://drive.google.com/open?id=1kWtx3j1ixua4nQt1HFHlwzCHnOr7gvKm)

[1.1M cell features and annotations](https://data.mendeley.com/datasets/nnbfwjvmvw/draft?a=dae895d4-25cd-4bdf-b3e4-57dd31c11e37)

```
import parc
import pandas as pd

// load data: digital mix of 7 cell lines from 7 sets of pure samples (1.1M cells)
X = pd.read_csv("'./LungData.txt", header=None).values.astype("float")
y = list(pd.read_csv('./LungData_annotations.txt', header=None)[0]) // list of cell-type annotations

// run PARC on 1.1M cells
// jac_weighted_edges can be set to false which provides an unweighted graph to leiden and offers some speedup
parc1 = parc.PARC(X, true_label=y,jac_weighted_edges = False)
parc1.run_PARC() // run the clustering
parc_labels = parc1.labels

// run PARC on H1975 spiked cells
parc2 = parc.PARC(X, true_label=y, jac_std_global = 0.15, jac_weighted_edges = False) // 0.15 corresponds to pruning ~60% edges and can be effective for rarer populations than the default 'median'
parc2.run_PARC() // run the clustering
parc_labels_rare = parc2.labels

```
![](Images/70K_Lung_github_overview.png) tsne plot of annotations and PARC clustering, heatmap of features

## Parameters and Attributes

### Parameter tuning
For a more detailed explanation of the impact of tuning key parameters please see the Supplementary Analysis in our paper.
[PARC Supplementary Analysis](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/PAP/10.1093_bioinformatics_btaa042/1/btaa042_supplementary-data.pdf?Expires=1583098421&Signature=R1gJB7MebQjg7t9Mp-MoaHdubyyke4OcoevEK5817el27onwA7TlU-~u7Ug1nOUFND2C8cTnwBle7uSHikx7BJ~SOAo6xUeniePrCIzQBi96MvtoL674C8Pd47a4SAcHqrA2R1XMLnhkv6M8RV0eWS-4fnTPnp~lnrGWV5~mdrvImwtqKkOyEVeHyt1Iajeb1W8Msuh0I2y6QXlLDU9mhuwBvJyQ5bV8sD9C-NbdlLZugc4LMqngbr5BX7AYNJxvhVZMSKKl4aMnIf4uMv4aWjFBYXTGwlIKCjurM2GcHK~i~yzpi-1BMYreyMYnyuYHi05I9~aLJfHo~Qd3Ux2VVQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

| Input Parameter | Description |
| ---------- |----------|
| `data` | (numpy.ndarray) n_samples x n_features |
| `true_label` | (numpy.ndarray) (optional)|
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. the higher the parameter, the more edges are retained|
| `jac_std_global` |  (optional, default = 'median') global level  graph pruning. This threshold can also be set as the number of standard deviations below the network's mean-jaccard-weighted edges. 0.1-1 provide reasonable pruning. higher value means less pruning. e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained. We find both 0.15 and 'median' to yield good results resulting in pruning away ~ 50-60% edges |
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. higher value means less pruning|
| `random_seed` |  (optional, default = 42) The random seed to pass to Leiden|
| `resolution_parameter` |  (optional, default = 1) Uses ModuliartyVP and RBConfigurationVertexPartition|
| `jac_weighted_edges` |  (optional, default = True) Uses Jaccard weighted pruned graph as input to community detection. For very large datasets set this to False to observe a speed-up with negligble impact on accuracy |

| Attributes | Description |
| ---------- |----------|
| `labels` | (list) length n_samples of corresponding cluster labels |
| `f1_mean` | (list) f1 score (not weighted by population). For details see supplementary section of [paper](https://doi.org/10.1101/765628) |
| `stats_df` | (DataFrame) stores parameter values and performance metrics |

## Developers

### Installation

To install a development version:

```
git clone https://github.com/ShobiStassen/PARC.git
cd PARC
pip install -e ".[extra]"
```

### Testing

To run the test suite for the package:

```
cd PARC
pytest tests/
```

## References to dependencies
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- hsnwlib Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small   World graphs." TPAMI, preprint: https://arxiv.org/abs/1603.09320
- igraph (igraph.org/python/)

## Citing
If you find this code useful in your work, please consider citing this paper [PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa042/5714737)
