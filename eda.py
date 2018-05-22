import plot
import decomposition
import load
import manifold
import pandas as pd
import clustering
from scipy import sparse as sp
import numpy as np

if __name__ == "__main__":
    sparse = load.load_covariates("covariates.csv")

    pc = decomposition.decompose_svd(sparse, n_pc=100)
    np.savetxt("pc_100.csv", pc, delimiter=",")

    svd = decomposition.get_svd()
    plot.plot_svd_redundancy(svd)

    pc = pd.read_csv("pc_100.csv")

    outcome = load.load_outcome("outcome.csv")

    manifold_df = manifold.fit_manifold(pc, outcome, technique="isomap", dim=3)
    manifold_df.to_csv("isomap_pc_100_3d.csv", sep=",")

    manifold_df = pd.read_csv("isomap_pc_100_3d.csv")
    plot.plot_manifold(manifold_df)
    cluster_out = clustering.fit_cluster(manifold_df, n_clusters=5,
                                         technique="agglomerative")
    plot.plot_cluster(cluster_out)
