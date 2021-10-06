import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster


def test_instances(instances, configurations, number_runs, heuristic_adaptations=None):
    # instances: {"points": , n_clusters: }

    # my_points = instances["points"][i]
    # my_k = instances['n_clusters'][i][J]

    # Kann cluster.KMeans die Zeiten selber messen?

    time_normal = 0
    for i in range(number_runs):
        start = time.time()
        normal_k_means = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, random_state=randomSeeds[i])
        normal_k_means.fit(X)
        time_normal += (time.time() - start)
        inertias_normal[i] = normal_k_means.inertia_

    depths = configurations["depths"]
    norm_its = configurations["norm_its"]
    searchsteps = configurations["searchsteps"]

    # configurations: {"norm_it: ", "searchstep: ", "depth: "}
    # heuristic_adaptations: {"test_bestworst_logn": True / False, "stop_early": True / False}
    for depth in np.arange(depths.shape[0]):
        for norm_it in np.arange(norm_its.shape[0]):
            for step in np.arange(search_steps.shape[0]):
                # print("current depth: {} , current search steps: {}".format(depths[depth], search_steps[step]))

                inertias_current = 0
                for i in range(n_runs):
                    # First we run our algorithm in the current configuration without any heuristic
                    # adaptation. Then we include one heuristic adaptation one at a time and finally we add
                    # all heuristic adaptations and compare all results
                    return None

    # return inertias (dictionary) and times (dictionary)


n_runs = 10  # Anzahl der Durchläufe
# n_clusters = 8  # Anzahl der Cluster
n_clusters = 8  # Anzahl der Cluster

time_file = "current_times.txt"


# Verschiedene Datensätze

# data = np.genfromtxt('mall.txt', delimiter=',')
# data = pd.read_csv('chipotle.txt', delimiter=',', quotechar='"')
# X = data[data.columns[-2:]].to_numpy()
# X[:,[0,1]] = X[:,[1,0]]

# X = np.genfromtxt('s2.txt')

def prepare_instances():
    # initialize dictionary instances: corresponding point sets and possible value for k
    # return instances
    return None


X = np.delete(np.genfromtxt('datasets/pr91.txt'), 0, 1)  # Padberg Rinaldi Datensatz (Bohrplatten)
# X = np.loadtxt('rectangles.txt')                   # Rechteckige Cluster


# Datensatz plotten (derzeit nur für 2D)
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()


# Kram für den Plot der Cluster später
fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = cm.rainbow(np.linspace(0, 1, n_clusters))

# Optimale Lösung des Padberg-Rinaldi-Datensatzes
opt = 0.701338e+10

# Variablen für minimale/maximale/durchschnittliche Approximationsratios über alle runs
k_means_min_ratio = np.inf
mini_batch_min_ratio = np.inf

k_means_max_ratio = 0
mini_batch_max_ratio = 0

k_means_sum = 0
als_sum = 0

# depths = np.array([2, 3, 5, 7, 10])
depths = np.array([5, 7, 10])
# depths = np.array([10, 11, 12])
# depths = np.array([10, 11, 12])
# search_steps = np.arange(1, np.log2(n_clusters) + 1, 2)
search_steps = np.arange(1, 4)
search_steps = search_steps = search_steps.astype('int64')
# norm_its = np.arange(2,3)
norm_its = np.array([2, 3, 5, 7, 10])

minimum_ratios = np.ones((depths.shape[0], search_steps.shape[0])) * -1
maximum_ratios = np.ones((depths.shape[0], search_steps.shape[0])) * -1
avg_ratios = np.zeros((depths.shape[0], search_steps.shape[0]))

avg_normal = 0
best_normal = -1
worst_normal = 0

times = np.zeros((depths.shape[0], norm_its.shape[0], search_steps.shape[0]))

time_normal = 0
inertias_normal = np.zeros(n_runs)

# we fix some specific Seeds as the random initialisation for every run
randomSeeds = np.random.randint(0, 1000000, n_runs)

for i in range(n_runs):
    start = time.time()
    normal_k_means = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, random_state=randomSeeds[i])
    normal_k_means.fit(X)
    time_normal += (time.time() - start)
    inertias_normal[i] = normal_k_means.inertia_

time_normal /= n_runs
quality_normal = (sum(inertias_normal) / n_runs) / time_normal
print("normal elkan: inertia = {}   time = {}   quality = {}".format(sum(inertias_normal) / n_runs, time_normal, quality_normal))

# Eigentliches Clustering
for depth in np.arange(depths.shape[0]):
    for norm_it in np.arange(norm_its.shape[0]):
        for step in np.arange(search_steps.shape[0]):
            # print("current depth: {} , current search steps: {}".format(depths[depth], search_steps[step]))

            inertias_current = 0
            for i in range(n_runs):
                if depth == 0 and step == 0 and norm_it == 0:
                    normal_k_means = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, random_state=randomSeeds[i])
                    normal_k_means.fit(X)
                    avg_normal = avg_normal + normal_k_means.inertia_
                    if best_normal == -1 or best_normal > normal_k_means.inertia_:
                        best_normal = normal_k_means.inertia_
                    if worst_normal < normal_k_means.inertia_:
                        worst_normal = normal_k_means.inertia_

                # Lloyd's und Mini-Batch, jeweils mit D^2-Sampling zur Initialiserung
                # als_pp = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1)
                als_pp = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=1, algorithm='als++',
                                        depth=depths[depth],
                                        search_steps=search_steps[step], norm_it=norm_its[norm_it], random_state=randomSeeds[i])

                # als_pp.fit(X)
                als_pp.fit_new(X)

                inertias_current += als_pp.inertia_

                fp = open(time_file, "r")
                o = fp.readline()
                while True:
                    output = o.split()
                    if output[0] == "overall_time":
                        times[depth][norm_it][step] += float(output[1]) / n_runs
                        break
                    o = fp.readline()

                # Infos für den Plot
                # km_centers = k_means.cluster_centers_
                # k_means_labels = pairwise_distances_argmin(X, km_centers)

                # mb_centers = als_pp.cluster_centers_
                # mb_labels = pairwise_distances_argmin(X, mb_centers)

                # Ratios ggf aktualisieren
                # if k_means.inertia_/opt < k_means_min_ratio:
                #    k_means_min_ratio = k_means.inertia_/opt

                if minimum_ratios[depth][step] == -1:
                    minimum_ratios[depth][step] = als_pp.inertia_ / opt
                else:
                    if als_pp.inertia_ / opt < minimum_ratios[depth][step]:
                        minimum_ratios[depth][step] = als_pp.inertia_ / opt

                # if k_means.inertia_/opt > k_means_max_ratio:
                #    k_means_max_ratio = k_means.inertia_/opt

                if maximum_ratios[depth][step] == -1:
                    maximum_ratios[depth][step] = als_pp.inertia_ / opt
                else:
                    if als_pp.inertia_ / opt > maximum_ratios[depth][step]:
                        maximum_ratios[depth][step] = als_pp.inertia_ / opt

                # k_means_sum = k_means_sum + k_means.inertia_
                avg_ratios[depth][step] = avg_ratios[depth][step] + (als_pp.inertia_ / n_runs) / opt
                # als_sum = als_sum + als_pp.inertia_

            inertias_current /= n_runs

            print("new elkan for depth={}, norm_it={}, ss={}: inertia = {:.5f}   time = {:.5f}   quality = {:10}".format(depths[depth], norm_its[norm_it], search_steps[step],
                                                                                                                         inertias_current, times[depth][norm_it][step],
                                                                                                                         inertias_current / times[depth][norm_it][step]))

    print("current depth: " + str(depths[depth]))

# ax = fig.add_subplot(1, 2, 1)

# Plots (derzeit einfach nur von den Lösungen, die im letzten Run berechnet wurden)

# for k, col in zip(range(n_clusters), colors):
#     cluster_members = X[np.where(k_means_labels == k)]
#     cluster_center = km_centers[k]
#     ax.scatter(cluster_members[:, 0], cluster_members[:, 1], linestyle='None', marker='.', color=col)
#     ax.scatter(cluster_center[0], cluster_center[1], marker='s', color=col, edgecolor="black")
#     #plt.text('inertia: %f' % k_means.inertia_)
#     plt.text(0.05, 0.05, 'inertia: %f' % k_means.inertia_, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top')
#
# ax = fig.add_subplot(1, 2, 2)
#
# for k, col in zip(range(n_clusters), colors):
#     cluster_members = X[np.where(mb_labels == k)]
#     cluster_center = mb_centers[k]
#     ax.scatter(cluster_members[:, 0], cluster_members[:, 1], linestyle='None', marker='.', color=col)
#     ax.scatter(cluster_center[0], cluster_center[1], marker='s', color=col, edgecolor="black")
#     #plt.text('inertia: %f' % mini_batch.inertia_)
#     plt.text(0.05, 0.05, 'inertia: %f' % als_pp.inertia_, transform=ax.transAxes, fontsize=14,
#              verticalalignment='top')
# plt.show()


# print('k-means:' + str(k_means_min_ratio) + ' ' + str(k_means_max_ratio))
# print('mini_batch:' + str(mini_batch_min_ratio) + ' ' + str(mini_batch_max_ratio))
#
# print('avg k-means:' + str(k_means_sum / (n_runs * opt)))
# print('avg mini_batch:' + str(als_sum / (n_runs * opt)))

print("")
print("best ratio normal k-means: {}".format(best_normal / opt))
print("average ratio normal k-means: {}".format(avg_normal / (opt * n_runs)))
print("worst ratio normal k-means: {}".format(worst_normal / opt))
print("")

for depth in np.arange(depths.shape[0]):
    for search_step in np.arange(search_steps.shape[0]):
        print("depth: " + str(depths[depth]) + " , seachstep: " + str(search_steps[search_step]))
        print("minimum ratio: " + str(minimum_ratios[depth][search_step]))
        print("maximum ratio: " + str(maximum_ratios[depth][search_step]))
        print("average ratio: " + str(avg_ratios[depth][search_step]))
        print("")
