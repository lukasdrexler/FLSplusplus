import numpy as np
from sklearn import cluster

X = np.genfromtxt('datasets/pr91.txt')  # Padberg Rinaldi Datensatz (Bohrplatten)
# X = np.loadtxt('rectangles.txt')                   # Rechteckige Cluster


if __name__ == '__main__':
    n_clusters = 50

    # Running normal kmeans++ in current version:
    normal_k_means = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1)
    normal_k_means.fit(X)
    print("Clustering cost normal k_means++: {}".format(normal_k_means.inertia_))

    # Local search strategy
    local_search_k_means = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, algorithm='ls++', z=20)
    local_search_k_means.fit_new(X)
    print("Clustering cost ls++: {}".format(local_search_k_means.inertia_))

    # Running FLS++ using some arbitrary values for depth.
    # During testing we also analyzed the algorithm with respect to the additional value "norm_it" (first introduced in paper)
    # in chapter 4.1 Extensive search. There it represents parameter N
    # "norm_it" allows to either sample more times between future vision based on depth (depth > norm_it) or to skip more iterations and to
    # continue sampling using depth in a later iteration (depth < norm_it). For depth == norm_it we get the original procedure described
    # in the paper
    fls_pp = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, algorithm='fls++', depth=10, norm_it=3, n_local_trials=1)
    fls_pp.fit_new(X)
    print("Clustering cost fls++: {}".format(fls_pp.inertia_))


    # documentation of the heuristics:
    # - first_improve:              The first time we find an exchange of a center with the sampled point we greedily take this solution
    # - increasing_clustercosts:    We consider swaps of centers with the sampled point using the order of the corresponding clustercosts, i.e.,
    #                               we first check the swap with the center with smallest clustercost and so forth till the center with the largest clustercost
    # - increasing_distancesLog_clustercosts: First check the first log(k) centers sorted by their distance to the sampled candidate, then continue
    #                                         with the remaining sorted clustercosts like in the strategy "increasing_clustercosts"
    # - early_stop_exchanges:       If "increasing_distancesLog_clustercosts" is also True, we check the first log(k) centers sorted by their distance and then
    #                               log(k) centers with the largest clustercost and log(k) centers with the smallest clustercost
    # - early abort:                We wish to stop using the sampling-strategy, if often enough we did not find any improvement.
    #                               This breakpoint is reached, if the number of overall times we did not find an improvement/did not
    #                               exchange any center with a sampled point is at least the value saved in "early_abort_number"
    heuristics = {"first_improve": False, "increasing_clustercosts": False, "increasing_distancesLog_clustercosts": True, "early_abort": False, "early_abort_number": 4,
                  "early_stop_exchanges": True}
    fls_pp_heuristics = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, algorithm='fls++', depth=10, norm_it=3, heuristics=heuristics)
    fls_pp_heuristics.fit_new(X)
    print("Clustering cost fls++ using some heuristics: {}".format(fls_pp_heuristics.inertia_))