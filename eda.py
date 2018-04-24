import plot
import decomposition
import load
import manifold
import pandas as pd
import clustering

# sparse_no_redundancy = load.load_covariates(
#     "hf_in_t2dm/standard_covariates.csv",
#                                          remove_redundancy=True)
#
sparse = load.load_covariates("hf_in_t2dm/standard_covariates.csv",
                              remove_redundancy=True)
#
# plot.plot_sparsity(sparse_no_redundancy, 2500, paper=True)
# plot.plot_sparsity(sparse)
#
# pc = decomposition.decompose_svd(sparse, n_pc=1000)
# svd = decomposition.get_svd()
# plot.plot_svd_redundancy(svd)
#
outcome = load.load_outcome("hf_in_t2dm/outcome.csv")

pc = sparse.toarray()

manifold_df = manifold.fit_manifold(pc, outcome, technique="isomap", dim=3)
manifold_df.to_csv("iso_original_3d.csv", sep=",")
# # # manifold_df.to_csv("lle_400.csv", sep=",")
# #
# manifold_df = pd.read_csv("isomap_1000.csv")
# plot.plot_manifold(manifold_df)
# cluster_out = clustering.fit_cluster(manifold_df, n_clusters=5,
#                                      technique="agglomerative")
# plot.plot_cluster(cluster_out)
