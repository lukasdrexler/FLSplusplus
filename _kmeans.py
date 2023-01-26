"""K-means clustering."""

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
# License: BSD 3 clause
import math
import warnings

import numpy as np
import scipy.sparse as sp
from numpy.random.mtrand import sample
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info

from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import _euclidean_distances
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils import check_array
from ..utils import check_random_state
from ..utils import deprecated
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._readonly_array_wrapper import ReadonlyArrayWrapper
from ..exceptions import ConvergenceWarning
from ._k_means_common import CHUNK_SIZE
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse
from ._k_means_minibatch import _minibatch_update_dense
from ._k_means_minibatch import _minibatch_update_sparse
from ._k_means_lloyd import lloyd_iter_chunked_dense
from ._k_means_lloyd import lloyd_iter_chunked_sparse
from ._k_means_elkan import init_bounds_dense
from ._k_means_elkan import init_bounds_sparse
from ._k_means_elkan import elkan_iter_chunked_dense
from ._k_means_elkan import elkan_iter_chunked_sparse

from ..metrics.pairwise import paired_euclidean_distances

###############################################################################
# Initialization heuristic


def kmeans_plusplus(
    X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None
):
    """Init n_clusters seeds according to k-means++

    .. versionadded:: 0.24

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.

    n_clusters : int
        The number of centroids to initialize

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.

    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  4],
           [ 1,  0]])
    >>> indices
    array([4, 2])
    """

    # Check data
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])

    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError(
            f"n_local_trials is set to {n_local_trials} but should be an "
            "integer value greater than zero."
        )

    random_state = check_random_state(random_state)

    # Call private k-means++
    centers, indices = _kmeans_plusplus(
        X, n_clusters, x_squared_norms, random_state, n_local_trials
    )

    return centers, indices


def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="auto",
    return_n_iter=False,
):
    """K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in X. If None, all observations
        are assigned equal weight.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).

        For now "auto" (kept for backward compatibility) chooses "elkan" but it
        might change in the future for a better heuristic.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_


def _kmeans_single_elkan(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means elkan, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    x_squared_norms : array-like, default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """


    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(
        np.asarray(center_half_distances), kth=1, axis=0
    )[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(X, centers, center_half_distances, labels, upper_bounds, lower_bounds)

    strict_convergence = False

    for i in range(max_iter):
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
        )

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0
        )[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


#################################################################################
# Start of newly added content related to FLS++
#################################################################################

def _kmeans_als_plusplus(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1, depth=3, search_steps=3, norm_it=2):

    random_state = check_random_state(None)

    # centers_init is either some predefined set of centers or the output of D2-Sampling
    centers = centers_init

    n_clusters = centers_init.shape[0]
    k = n_clusters
    n_samples = X.shape[0]

    # initialize set of labels. We test for equality as break-condition
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()

    # additional measurements for tesing depths: we compare multiple values for depth until nothing changes
    best_depths = np.zeros(search_steps)
    best_depths = best_depths[np.newaxis, ...]

    # sanity checking iterations
    actual_iterations = 0

    # We iterate to at most max_iter iterations with norm_it steps (or labels stay the same)
    for iteration in range(0, max_iter, norm_it):
        actual_iterations += 1

        # calculate for sampling the potential and minimum distances:
        # closest_dist_sq: kxn, current_pot: sum over min_distances: nx1
        closest_dist_sq, current_pot, min_distances = calculate_potential(X, centers, x_squared_norms)


        # the best exchange found over the searchsteps
        best_depths_column = np.zeros(search_steps)

        exchange = True
        for i in range(search_steps):

            # draw according to probability distribution D2
            rand_val = random_state.random_sample() * current_pot  # draw random candidate proportional to its cost

            # find location where sampled point should be in sorted array (sum over distances)
            candidate_id = np.searchsorted(stable_cumsum(min_distances), rand_val)

            # XXX: numerical imprecision can result in a candidate_id out of range
            # np.clip(candidate_id, None, closest_dist_sq.size - 1, out=candidate_id)
            candidate_id = min(candidate_id, closest_dist_sq.size - 1)

            # calculate how much we have to pay for every cluster
            clustercosts = np.zeros(k)
            for l in range(X.shape[0]):
                center = np.argmin(closest_dist_sq[:, l])
                clustercosts[center] += closest_dist_sq[center, l]

            # ----------------------------------------------------------------------------------------------------------------------
            # main function: exchange every center with candidate, perform loyd/elkan for depth many steps, save best found solution
            # ----------------------------------------------------------------------------------------------------------------------
            best_index, solutions, start = exchange_solutions(X, clustercosts, assertions, plot_save, candidate_id, centers, depth, n_clusters, n_threads, norm_it, sample_weight, start, time_check, times,
                                                              tol, verbose, x_squared_norms, max_iter, search_steps)


            # check if some exchange was the best option
            if best_index != -1:

                # make exchange: replace center with new candidate
                centers[best_index] = X[candidate_id]  # make final change

                # replace distance-array-row of exchanged center with new row for new center
                closest_dist_sq[best_index] = euclidean_distances(
                    centers[best_index].reshape(1, X[0].shape[0]), X, Y_norm_squared=x_squared_norms,
                    squared=True)

                # recalculate (maybe faster) min_distances and pot
                min_distances = closest_dist_sq.min(axis=0)  # Distanzen zu den closest centers
                current_pot = min_distances.sum()  # Summe der quadrierten Abstände zu closest centers

            # no exchange option found
            if best_index == -1:
                exchange = False


        # depending on relation of depth and norm_it we need to make more iterations to find level of norm_it with new solution (norm_it > depth)
        # or can simply reuse precalculated solution (norm_it <= depth)
        if depth == norm_it:
            labels, inertia, centers = solutions['depth']['labels'][best_index + 1].copy(), solutions['depth']['inertia'][best_index + 1], \
                                       solutions['depth']['centers'][best_index + 1].copy()

        elif depth > norm_it:
            labels, inertia, centers = solutions['norm_it']['labels'][best_index + 1].copy(), solutions['norm_it']['inertia'][best_index + 1], \
                                       solutions['norm_it']['centers'][best_index + 1].copy()

        elif depth < norm_it:
            labels, inertia, centers, _ = _kmeans_single_elkan(X, sample_weight, solutions['depth']['centers'][best_index + 1].copy(), max_iter=norm_it - depth, verbose=verbose, tol=tol,
                                                               x_squared_norms=x_squared_norms, n_threads=n_threads)
            if inertia > solutions['depth']['inertia'][best_index + 1]:
                centers, labels, inertia = solutions['depth']['centers'][best_index + 1].copy(), solutions['depth']['labels'][best_index + 1].copy(), \
                                           solutions['depth']['inertia'][best_index + 1]


        if np.array_equal(labels, labels_old):
            # print("Labels are equal!")
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break

        labels_old[:] = labels

    return labels, inertia, centers, iteration + 1



"""
def _kmeans_fls_plusplus(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1):


    random_state = check_random_state(None)

    # centers_init is either some predefined set of centers or the output of D2-Sampling
    centers = centers_init

    n_clusters = centers_init.shape[0]
    k = n_clusters
    n_samples = X.shape[0]

    # initialize set of labels. We test for equality as break-condition
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()


    # We iterate to at most max_iter iterations with norm_it steps (or labels stay the same)
    for iteration in range(0, max_iter):
        # calculate for sampling the potential and minimum distances:
        # closest_dist_sq: kxn, current_pot: sum over min_distances: nx1
        closest_dist_sq, current_pot, min_distances = calculate_potential(X, centers, x_squared_norms)


        exchange = True
        # draw according to probability distribution D2
        rand_val = random_state.random_sample() * current_pot  # draw random candidate proportional to its cost

        # sanity check
        if not np.allclose(current_pot, sum(min_distances)):
            error()
            print('Sums do not match for random value: difference = {}'.format(current_pot - sum(min_distances)))

        # find location where sampled point should be in sorted array (sum over distances)
        candidate_id = np.searchsorted(stable_cumsum(min_distances), rand_val)

        # XXX: numerical imprecision can result in a candidate_id out of range
        # np.clip(candidate_id, None, closest_dist_sq.size - 1, out=candidate_id)
        candidate_id = min(candidate_id, closest_dist_sq.size - 1)


        clustercosts = np.zeros(k)
            for l in range(X.shape[0]):
                center = np.argmin(closest_dist_sq[:, l])
                clustercosts[center] += closest_dist_sq[center, l]

            # ----------------------------------------------------------------------------------------------------------------------
            # main function: exchange every center with candidate, perform loyd/elkan for depth many steps, save best found solution
            # ----------------------------------------------------------------------------------------------------------------------
            best_index, solutions, start = exchange_solutions(X, clustercosts, assertions, plot_save, candidate_id, centers, depth, n_clusters, n_threads, norm_it, sample_weight, start, time_check, times,
                                                              tol, verbose, x_squared_norms, max_iter, search_steps)

            # check if some exchange was the result / best option
            if best_index != -1:
                # make exchange: replace center with new candidate
                centers[best_index] = X[candidate_id]  # make final change

                # replace distance-array-row of exchanged center with new row for new center
                closest_dist_sq[best_index] = euclidean_distances(
                    centers[best_index].reshape(1, X[0].shape[0]), X, Y_norm_squared=x_squared_norms,
                    squared=True)

                # recalculate (maybe faster) min_distances and pot
                min_distances = closest_dist_sq.min(axis=0)  # Distanzen zu den closest centers
                current_pot = min_distances.sum()  # Summe der quadrierten Abstände zu closest centers

            # no exchange option found
            if best_index == -1:
                exchange = False

        if np.array_equal(labels, labels_old):
            # print("Labels are equal!")
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True

        labels_old[:] = labels

        return labels, inertia, centers, iteration + 1
"""

def _localSearchPP(X, sample_weight, centers_init, max_iter=300, verbose=False, x_squared_norms=None, tol=1e-4, n_threads=1, z=None):
    centers = centers_init
    n = len(X)
    k = len(centers)

    labels = None

    random_state = check_random_state(None)

    if z==None:
        z = math.floor(100000*k*np.log2(np.log2(k)))

    for i in range(z):

        if labels is None:
            closest_dist_sq = euclidean_distances(
                centers, X, Y_norm_squared=x_squared_norms,
                squared=True)

            labels = np.argpartition(closest_dist_sq, 1, axis=0)[:2]

            min_distances = closest_dist_sq[labels, np.arange(closest_dist_sq.shape[1])]

            current_pot = min_distances[0,:].sum()

        rand_val = random_state.random_sample() * current_pot  # draw random candidate proportional to its cost

        # find location where sampled point should be in sorted array (sum over distances)
        candidate_id = np.searchsorted(stable_cumsum(min_distances), rand_val)

        # XXX: numerical imprecision can result in a candidate_id out of range
        # np.clip(candidate_id, None, closest_dist_sq.size - 1, out=candidate_id)
        candidate_id = min(candidate_id, min_distances.size - 1)

        # calculate all distances between the candidate and the other points
        # assume we remove center i:
        # if we have points with label i: recalculate the distance to its closest center
        # otherwise: check if the distance decreases if we assign the point to the new candidate
        candidate_distances = euclidean_distances(X[candidate_id].reshape((1,len(X[candidate_id]))), X, Y_norm_squared=x_squared_norms, squared=True)

        min_pot = current_pot
        found_exchange = False
        best_exchange = 0

        for j in range(k):
            same_labels = np.ma.masked_where(labels[0] != j, labels[0])
            A = np.minimum(min_distances[0], candidate_distances[0], where=same_labels.mask)[same_labels.mask].sum()
            B = np.minimum(min_distances[1], candidate_distances[0], where=~same_labels.mask)[~same_labels.mask].sum()
            newpot = A + B

            if newpot < min_pot:
                found_exchange = True
                best_exchange = j
                min_pot = newpot

        if found_exchange:
            j = best_exchange

            new_labels = labels.copy()
            new_min_distances = min_distances.copy()

            for point in range(n):
                if labels[0,point] != j and labels[1,point] != j:
                    if new_min_distances[0,point] > candidate_distances[0,point]:
                        new_labels[1,point] = new_labels[0,point]
                        new_labels[0,point] = j
                        new_min_distances[1, point] = new_min_distances[0, point]
                        new_min_distances[0,point] = candidate_distances[0,point]

                    elif new_min_distances[0,point] <= candidate_distances[0,point] < new_min_distances[1,point]:
                        new_labels[1, point] = j
                        new_min_distances[1, point] = candidate_distances[0,point]

                # we remove for a point its second-closest
                elif labels[1,point] == j:
                    if new_min_distances[0,point] > candidate_distances[0,point]:
                        new_labels[1,point] = new_labels[0,point]
                        new_labels[0,point] = j
                        new_min_distances[1, point] = new_min_distances[0, point]
                        new_min_distances[0,point] = candidate_distances[0,point]
                    else:
                        # find the third-closest center which is now (without the new sampled point) the second closest center
                        labels_third_closest = np.argpartition(closest_dist_sq[:, point], 2, axis=0)
                        label_third_closest = labels_third_closest[2]
                        if label_third_closest == j:
                            if closest_dist_sq[labels_third_closest[0], point] <= closest_dist_sq[labels_third_closest[1], point]:
                                label_second_closest = labels_third_closest[1]
                            else:
                                label_second_closest = labels_third_closest[0]
                            label_third_closest = label_second_closest
                            assert label_third_closest != j


                        assert label_third_closest == np.argsort(closest_dist_sq[:,point])[2] or \
                               closest_dist_sq[label_third_closest,point] == closest_dist_sq[np.argsort(closest_dist_sq[:,point])[2], point]

                        distance_third_closest = closest_dist_sq[label_third_closest, point]
                        # case where new sampled point becomes second closest
                        if distance_third_closest > candidate_distances[0,point]:
                            new_labels[1, point] = j
                            new_min_distances[1, point] = candidate_distances[0, point]
                        # case where third-closest becomes second-closest
                        else:
                            new_labels[1, point] = label_third_closest
                            new_min_distances[1, point] = distance_third_closest

                    #new_pot += new_min_distances[0,point]

                # we remove the closest point
                elif labels[0,point] == j:
                    if new_min_distances[1,point] > candidate_distances[0,point]:
                        # we only need to update the closest center
                        new_labels[0,point] = j
                        new_min_distances[0,point] =  candidate_distances[0,point]
                    else:
                        # make the second closest center now the closest center
                        new_labels[0,point] = new_labels[1,point]
                        new_min_distances[0,point] = new_min_distances[1,point]

                        # find the third-closest center which is now (without the new sampled point) the second closest center

                        # Problem: If multiple centers have same distance this can have unexpected results (f.e. now the
                        # 2-closest center is the 3-closest). In this case we need to select the second closest in the left partition
                        # (which is also a valid "3-closest")

                        labels_third_closest = np.argpartition(closest_dist_sq[:, point], 2, axis=0)
                        label_third_closest = labels_third_closest[2]
                        if label_third_closest == j:
                            if closest_dist_sq[labels_third_closest[0], point] <= closest_dist_sq[labels_third_closest[1], point]:
                                label_second_closest = labels_third_closest[1]
                            else:
                                label_second_closest = labels_third_closest[0]
                            label_third_closest = label_second_closest
                            assert label_third_closest != j

                            #################################

                        distance_third_closest = closest_dist_sq[label_third_closest, point]
                        # case where new sampled point becomes second closest
                        if distance_third_closest > candidate_distances[0, point]:
                            new_labels[1, point] = j
                            new_min_distances[1, point] = candidate_distances[0, point]
                        # case where third-closest becomes second-closest
                        else:
                            new_labels[1, point] = label_third_closest
                            new_min_distances[1, point] = distance_third_closest


        if found_exchange:
            if verbose:
                print("Decrease in inertia: old inertia = {}, new inertia = {}".format(current_pot, min_pot))

            labels = new_labels
            current_pot = min_pot
            centers[best_exchange] = X[candidate_id]
            closest_dist_sq[best_exchange] = candidate_distances[0]
            min_distances = new_min_distances

    # run elkan for max_iter iterations, using the current best found centers

    labels, inertia, centers, n_iter = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=max_iter, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms, n_threads=n_threads)

    return labels, inertia, centers, z + n_iter + 1


def _fls_d_one(X, sample_weight, centers_init, max_iter=300, verbose=False, x_squared_norms=None, tol=1e-4, n_threads=1, z=None, random_state=None):

    centers = centers_init
    n = len(X)
    k = len(centers)

    labels = None

    if z==None:
        z = math.floor(100000*k*np.log2(np.log2(k)))

    for i in range(z):

        centers_copy = centers.copy()

        if labels is None:
            # (k x n) matrix of all distance pairs
            closest_dist_sq = euclidean_distances(
                centers, X, Y_norm_squared=x_squared_norms,
                squared=True)

            # we find for each point its closest and secondclosest center
            labels = np.argpartition(closest_dist_sq, 1, axis=0)[:2]
            min_distances = closest_dist_sq[labels, np.arange(closest_dist_sq.shape[1])]

            # for each point we add the sum of (squared) distances to its closest center
            _, current_pot, _, _ = _kmeans_single_elkan(X, sample_weight, centers, max_iter=1, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                                                    n_threads=n_threads)

        rand_val = random_state.random_sample() * current_pot  # draw random candidate proportional to its cost
        # find location where sampled point should be in sorted array (sum over distances)
        candidate_id = np.searchsorted(stable_cumsum(min_distances), rand_val)

        # XXX: numerical imprecision can result in a candidate_id out of range
        # np.clip(candidate_id, None, closest_dist_sq.size - 1, out=candidate_id)
        candidate_id = min(candidate_id, min_distances.size - 1)

        # calculate all distances between the candidate and the other points
        # assume we remove center i:
        # if we have points with label i: recalculate the distance to its closest center
        # otherwise: check if the distance decreases if we assign the point to the new candidate
        candidate_distances = euclidean_distances(X[candidate_id].reshape((1,len(X[candidate_id]))), X, Y_norm_squared=x_squared_norms, squared=True)

        min_pot = current_pot
        found_exchange = False
        best_exchange = 0
    
        for j in range(k):
            # all points which have NOT the currently exchanged center j as their respective closest center
            same_labels = np.ma.masked_where(labels[0] != j, labels[0])
            idx_closest = np.where(same_labels.mask)[0]

            # First consider all points for which removed center is NOT the closest one. Then A contains for each such point the information whether label needs
            # to change.
            A = np.argmin(np.r_[ [min_distances[0][same_labels.mask]], [candidate_distances[0][same_labels.mask]]], axis=0)

            new_labels = labels.copy()

            # for each point, where A==1 we know that the new candidate is the new closest center
            new_labels[0][idx_closest[np.where(A==1)[0]]] = j

            # for the remaining points we compare the secondclosest center to the candidate since the closest center got removed (inverted mask)
            B = np.argmin(np.r_[ [min_distances[1][~same_labels.mask]], [candidate_distances[0][~same_labels.mask]]], axis=0)

            idx_secondclosest = np.where(~same_labels.mask)[0]
            new_labels[0][idx_secondclosest[np.where(B == 1)[0]]] = j
            new_labels[0][idx_secondclosest[np.where(B == 0)[0]]] = new_labels[1][idx_secondclosest[np.where(B == 0)[0]]]

            newpot = 0
            centroids = np.zeros((k,X.shape[1]))

            Y = [np.mean(X[np.where(new_labels[0] == l)[0]], axis=0) if len(X[np.where(new_labels[0] == l)[0]])>0 else centers[l] for l in range(k)]
            # if number of relevant centers reduces we sample random point

            newpot = euclidean_distances(Y, X, Y_norm_squared=x_squared_norms, squared=True)[new_labels[0], np.arange(n)].sum()

            if newpot < min_pot:
                found_exchange = True
                best_exchange = j
                min_pot = newpot
                same_labels_min = same_labels.copy()
                idx_closest_min = idx_closest.copy()
                idx_secondclosest_min = idx_secondclosest.copy()
                A_closest = A.copy()
                B_secondclosest = B.copy()
    
            # if found_exchange = True we need to make an update to our set of centers, labels and min_distances. The closest center distance
            # is already updated so we only need to update the secondclosest center distance and label
            if found_exchange:
                j = best_exchange

                # for each of the following points we set the new minimum distance as the distance to the candidate
                min_distances[0][idx_closest_min[np.where(A_closest==1)[0]]] = candidate_distances[0][idx_closest_min[np.where(A_closest==1)[0]]]
                min_distances[0][idx_secondclosest_min[np.where(B_secondclosest==1)[0]]] = candidate_distances[0][idx_secondclosest_min[np.where(B_secondclosest==1)[0]]]
                min_distances[0][idx_secondclosest_min[np.where(B_secondclosest == 0)[0]]] = min_distances[1][idx_secondclosest_min[np.where(B_secondclosest == 0)[0]]]
                A_counter = 0
                B_counter = 0
                for point in range(n):
                    # check if for point neither closest nor secondclosest center got removed and continue accordingly
                    if labels[0, point] != j and labels[1, point] != j:
                        assert point == idx_closest_min[A_counter]
                        # in the first case the new candidate is the closest center to the point, so the secondclosest becomes the old closest center
                        if A_closest[A_counter] == 1:
                            new_labels[1, point] = labels[0, point]
                            min_distances[1, point] = closest_dist_sq[labels[0, point], point]
                        # in the second case the closest center stays the same. If the old secondclosest center is still the secondclosest we
                        # do nothing, otherwise the candidate center becomes the secondclosest center
                        ###################
                        # I think we can save time in further iterations by saving which center is the thirdclosest
                        ###################
                        else:
                            if min_distances[1, point] > candidate_distances[0][point]:
                                new_labels[1, point] = j
                                min_distances[1, point] = candidate_distances[0][point]
                        A_counter += 1
                    # We remove the old closest center to the point.
                    elif labels[0, point] == j:
                        assert point == idx_secondclosest_min[B_counter]
                        # We already checked if now the secondclosest or the candidate becomes closest center
                        # If the secondclosest becomes closest we need to compare the thirdclosest to the candidate
                        if B_secondclosest[B_counter] == 0:
                            label_third_closest = np.argpartition(closest_dist_sq[:,point], 2, axis=0)[2]
                            if candidate_distances[0][point] < closest_dist_sq[label_third_closest, point]:
                                new_labels[1, point] = j
                                min_distances[1, point] = candidate_distances[0][point]
                            else:
                                new_labels[1, point] = label_third_closest
                                min_distances[1, point] = closest_dist_sq[label_third_closest, point]
                        B_counter += 1
                    # We removed the secondclosest center.
                    elif labels[1, point] == j:
                        assert point == idx_closest_min[A_counter]
                        # If the candidate is now closest center than the old closest becomes secondclosest
                        if A_closest[A_counter] == 1:
                            new_labels[1, point] = labels[0, point]
                            min_distances[1, point] = closest_dist_sq[labels[0, point], point]
                        # If the old closest stays closest we need to compare the thirdclosest with the candidate
                        else:
                            label_third_closest = np.argpartition(closest_dist_sq[:, point], 2, axis=0)[2]
                            if candidate_distances[0][point] < closest_dist_sq[label_third_closest, point]:
                                new_labels[1, point] = j
                                min_distances[1, point] = candidate_distances[0][point]
                            else:
                                new_labels[1, point] = label_third_closest
                                min_distances[1, point] = closest_dist_sq[label_third_closest, point]
                        A_counter += 1
    
                        # we check if the mindistances are correct according to the current centers:

                # We already updated the mindistances, so we only update labels and pot
                print("found improvement: old pot = {} , new pot = {}".format(current_pot, min_pot))
                # consecutive_print = 0
                labels = new_labels.copy()
                current_pot = min_pot
                centers[best_exchange] = X[candidate_id]
                closest_dist_sq[best_exchange] = candidate_distances

    
    labels, inertia, centers, n_iter = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=max_iter, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                                            n_threads=n_threads)

    return labels, inertia, centers, z + n_iter + 1




def _kmeans_als_plusplus_fast(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1, depth=3, norm_it=2, heuristics={}, random_state=None):

    # If no heristics were provided we dont use any
    if len(heuristics) == 0:
        heuristics = {"first_improve": False, "increasing_clustercosts": False, "increasing_distancesLog_clustercosts": False, "early_abort": False, "early_abort_number": 4,
                      "early_stop_exchanges": False}

    # controls if we make in one step multiple sampling steps to simulate the average output in current step
    n_runs_average_improvement = 100
    no_improvement_counter = 0

    if verbose:
        print("Starting alspp: depth {} norm_it {} search_step {}".format(depth, norm_it, search_steps))
        print("heuristics:")
        for key, value in heuristics.items():
            print(f"{key}: {value}")

    # centers_init is either some predefined set of centers or the output of D2-Sampling
    centers = centers_init

    n_clusters = centers_init.shape[0]
    n_samples = X.shape[0]

    # initialize set of labels. We test for equality as break-condition
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()

    # We iterate to at most max_iter iterations with norm_it steps (or labels stay the same)
    for iteration in range(0, max_iter, norm_it):
        if verbose:
            print(f"Iteration {iteration} in main loop")

        if iteration == 0:
            # calculate for sampling the potential and minimum distances:
            closest_dist_sq, current_pot, min_distances, clustercosts = calculate_potential_clustercosts(X, centers, x_squared_norms=x_squared_norms)
        else:
            # If labels is already defined we compute the new min_distances and potential using the labels
            closest_dist_sq, current_pot, min_distances, clustercosts = calculate_potential_clustercosts(X, centers, labels=labels)


        # draw according to probability distribution D2
        rand_val = random_state.random_sample() * current_pot  # draw random candidate proportional to its cost

        # find location where sampled point should be in sorted array (sum over distances)
        candidate_id = np.searchsorted(stable_cumsum(min_distances), rand_val)

        # XXX: numerical imprecision can result in a candidate_id out of range
        # np.clip(candidate_id, None, closest_dist_sq.size - 1, out=candidate_id)
        candidate_id = min(candidate_id, min_distances.size - 1)

        best_index, best_centers_min, best_labels_min, best_inertia_compare, best_inertia_min, inertia_unchanged_solution, best_found_pos = exchange_solutions_faster(X, candidate_id, centers, clustercosts, depth, n_clusters, n_threads, norm_it,
                                                                                                         sample_weight, tol, verbose, x_squared_norms, max_iter, heuristics)

        # check if some exchange was the best option
        if best_index != -1:
            # make exchange: replace center with new candidate
            centers[best_index] = X[candidate_id]  # make final change


            # compute the new distances of the sampled point to the other candidates
            new_distances_candidate = euclidean_distances(
                 centers[best_index].reshape(1, X[0].shape[0]), X, Y_norm_squared=x_squared_norms,
                 squared=True).reshape(X.shape[0])

            # recalculate min_distances and potential
            min_distances = np.minimum(min_distances, new_distances_candidate)
            current_pot = min_distances.sum()

        elif best_index == -1:
            if heuristics["early_abort"] == True:
                no_improvement_counter += 1

        old_centers = centers.copy()

        # depending on relation of depth and norm_it we need to make more iterations to find level of norm_it with new solution (norm_it > depth)
        # or can simply reuse precalculated solution (norm_it <= depth)

        if depth == norm_it:
            labels, centers, inertia = best_labels_min, best_centers_min, best_inertia_min

        elif depth > norm_it:
            labels, centers, inertia = best_labels_min, best_centers_min, best_inertia_min

        elif depth < norm_it:
            labels, inertia, centers, _ = _kmeans_single_elkan(X, sample_weight, best_centers_min.copy(), max_iter=norm_it - depth, verbose=verbose, tol=tol,
                                                               x_squared_norms=x_squared_norms, n_threads=n_threads)
            if inertia > best_inertia_min:
                labels, centers, inertia = best_labels_min, best_centers_min, best_inertia_min


        if np.array_equal(labels, labels_old):
            # print("Labels are equal!")
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {iteration}: strict convergence.")
            strict_convergence = True
            break
        else:
            center_shift = paired_euclidean_distances(centers, old_centers)
            center_shift_tot = (center_shift ** 2).sum() / norm_it
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {iteration}: center shift difference below tolerance "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

        if heuristics["early_abort"] and no_improvement_counter >= heuristics["early_abort_number"]:
            if verbose:
                print("Terminated because of early abort.")
            final_labels, final_inertia, final_centers, final_iteration = _kmeans_single_elkan(X, sample_weight, centers, max_iter=max(max_iter - iteration - norm_it, 1), verbose=verbose, x_squared_norms=x_squared_norms, tol=tol, n_threads=n_threads)
            labels = final_labels
            inertia = final_inertia
            centers = final_centers
            iteration += final_iteration
            break

    return labels, inertia, centers, iteration

def exchange_solutions_fast(X, candidate_id, centers, clustercosts, depth, n_clusters, n_threads, norm_it, sample_weight, tol, verbose, x_squared_norms, max_iter,
                       search_steps):

    verbose_als = verbose
    verbose=False

    # first calculate appropriate solution where no exchange happend to level depth
    if depth <= norm_it:

        labels_depth, inertia, centers_depth, _ =  _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms, n_threads=n_threads)
        # inertia_normit = inertia  # inertia_normit should be evaluated later, except if depth==inertia
        if depth == norm_it:
            inertia_depth = inertia

    elif depth > norm_it:
        labels_depth, inertia_depth, centers_depth, _ = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=norm_it, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms, n_threads=n_threads)

        _, inertia, _, _ = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth - norm_it, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                 n_threads=n_threads)

        # in some cases a single iteration of single_elkan can make the solution worse
        # we then just return the better solution of the two
        if inertia > inertia_depth:
            inertia = inertia_depth

    best_value = inertia

    if depth >= norm_it:
        inertia_normit = inertia_depth
    else:
        inertia_normit = None

    best_centers = centers_depth.copy()
    best_labels = labels_depth.copy()
    best_index = -1  # index of best found swap candidate (if any)
    # comparision of sampled point to old centers

    if verbose_als:
        print(f"starting with inertia {best_value}")
        centers_candidate_distances = euclidean_distances(centers, X[candidate_id].reshape(1,-1), squared=True).reshape(n_clusters)

    # now we exchange our centers with possible candidate, run to level depth (or less) and compare output inertias to find best candidate
    for j in range(n_clusters):
        old_center = centers[j].copy()  # store old center (candidate for swapping)
        centers[j] = X[candidate_id]  # swap current centers with candidate
        if depth <= norm_it:
            labels_depth, inertia, centers_depth, _ =_kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms, n_threads=n_threads)


        elif depth > norm_it:
            labels_depth, inertia_depth, centers_depth, _ = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=norm_it, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms, n_threads=n_threads)

            _, inertia, _, _ = _kmeans_single_elkan(X, sample_weight, centers_depth.copy(), max_iter=depth - norm_it, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                     n_threads=n_threads)

            # in some cases a single iteration of single_elkan can make the solution worse
            # we then just return the better solution of the two
            if inertia > inertia_depth:
                inertia = inertia_depth

        centers[j] = old_center.copy()  # undo swap

        if best_value > inertia:
            #calculate the shift in centers (only evaluated for the centers after min(depth, norm_it) steps)
            center_shift = paired_euclidean_distances(best_centers, centers_depth)
            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot > tol:
                best_index = j
                best_value = inertia
                best_centers = best_centers.copy()
                best_labels = labels_depth

                if depth == norm_it:
                    inertia_depth = inertia
                if depth >= norm_it:
                    inertia_normit = inertia_depth
                else:
                    inertia_normit = None


    return best_index, best_centers, best_labels, best_value, inertia_normit


def exchange_solutions_faster(X, candidate_id, centers, clustercosts, depth, n_clusters, n_threads, norm_it, sample_weight, tol, verbose, x_squared_norms, max_iter,
                       heuristics):
    verbose_als = verbose
    verbose=False

    first_improve = False
    increasing_clustercosts = False
    increasing_distancesLog_clustercosts = False
    early_stop_exchanges = False
    if "first_improve" in heuristics:
        first_improve = heuristics["first_improve"]
    if "increasing_clustercosts" in heuristics and heuristics["increasing_clustercosts"] == True:
        increasing_clustercosts = True
    elif "increasing_distancesLog_clustercosts" in heuristics:
        increasing_distancesLog_clustercosts = heuristics["increasing_distancesLog_clustercosts"]
        if "early_stop_exchanges" in heuristics and heuristics["early_stop_exchanges"] == True:
            early_stop_exchanges = True

    # first calculate appropriate solution where no exchange happend to level depth

    centers_min, labels_min, centers_depth, inertia_min, inertia_compare = future_vision(X, centers.copy(), depth, n_threads, norm_it, sample_weight, tol,
                                                                                                                             verbose, x_squared_norms)
    inertia_unchanged_solution = inertia_compare
    best_centers_min = centers_min.copy()
    best_labels_min = labels_min.copy()
    best_inertia_compare = inertia_compare
    best_inertia_min = inertia_min
    best_index = -1  # index of best found swap candidate (if any)
    best_found_pos = -1 # position in enumeration at which we found our improvement (if any)
    # comparision of sampled point to old centers

    centers_candidate_distances = None

    if verbose_als:
        print(f"starting with inertia {best_inertia_compare}")
        centers_candidate_distances = euclidean_distances(centers, X[candidate_id].reshape(1,-1), squared=True).reshape(n_clusters)

    if increasing_clustercosts: # check centers by their increasing clustercosts
        possible_exchange_centers = np.argsort(clustercosts)
    elif increasing_distancesLog_clustercosts: # first check the first log(k) centers sorted by their distance to candidate, then continue with sorted clustercosts
        possible_exchange_centers = np.argsort(clustercosts)
        if centers_candidate_distances is None:
            centers_candidate_distances = euclidean_distances(centers, X[candidate_id].reshape(1, -1), squared=True).reshape(n_clusters)
        #centers_candidate_distances = centers_candidate_distances[np.argsort(centers_candidate_distances)][np.log(n_clusters)]
        centers_candidate_distances_sorted = np.argsort(centers_candidate_distances)
        centers_candidate_distances_firstLog = centers_candidate_distances_sorted[:(int)(np.ceil(np.log2(n_clusters)))]
        centers_to_include = possible_exchange_centers[~np.in1d(possible_exchange_centers, centers_candidate_distances_firstLog)]
        if early_stop_exchanges:
            # we add the first "cheapest cluster centers" and the "most expensive cluster centers"
            if 2*(int)(np.ceil(np.log2(n_clusters))) >= len(centers_to_include):
                # the remaining parts are larger together so we add them all
                possible_exchange_centers = np.append(centers_candidate_distances_firstLog, centers_to_include)
            else:
                possible_exchange_centers = np.append(centers_candidate_distances_firstLog, centers_to_include[:(int)(np.ceil(np.log2(n_clusters)))])
                possible_exchange_centers = np.append(possible_exchange_centers, centers_to_include[-(int)(np.ceil(np.log2(n_clusters))):])
        else:
            possible_exchange_centers = np.append(centers_candidate_distances_firstLog, centers_to_include)
    else:
        possible_exchange_centers = np.arange(0, n_clusters)

    for index, j in enumerate(possible_exchange_centers):

        old_center = centers[j].copy()  # store old center (candidate for swapping)
        centers[j] = X[candidate_id]  # swap current centers with candidate

        centers_min, labels_min, centers_depth, inertia_min, inertia_compare = future_vision(X, centers.copy(), depth, n_threads, norm_it, sample_weight, tol, verbose, x_squared_norms)
        # centers_min: centers after min(depth, norm_it) many steps
        # labels_min: labels after min(depth, norm_it) many steps
        # centers_depth: centers after depth many steps
        # inertia_min: inertia after min(depth, norm_it) many steps
        # inertia_compare: inertia after depth many steps


        if best_inertia_compare > inertia_compare:
            #calculate the shift in centers (only evaluated for the centers after min(depth, norm_it) steps)

            if depth == norm_it:
                center_shift = paired_euclidean_distances(centers, centers_min)
            elif depth < norm_it:
                center_shift = paired_euclidean_distances(centers, centers_min)
            elif depth > norm_it:
                center_shift = paired_euclidean_distances(centers, centers_depth)


            center_shift_tot = (center_shift ** 2).sum()
            if center_shift_tot > tol:
                if verbose_als:
                   print(f"found improvement in exchange {index}: old inertia = {best_inertia_compare}, new inertia = {inertia_compare}")
                   print("The clustercost of removed center is {}-smallest value".format(np.where(np.sort(clustercosts)==clustercosts[j])[0][0]))
                   print("The removed center is {}-closest to candidate".format(np.where(np.sort(centers_candidate_distances)== centers_candidate_distances[j])[0][0]))

                best_index = j
                best_centers_min = centers_min.copy()
                best_labels_min = labels_min.copy()
                best_inertia_compare = inertia_compare
                best_inertia_min = inertia_min
                best_found_pos = index

                if first_improve:
                    # need to reverse exchange in this case for consistency
                    centers[j] = old_center.copy()
                    break

        centers[j] = old_center.copy()  # undo swap


    return best_index, best_centers_min, best_labels_min, best_inertia_compare, best_inertia_min, inertia_unchanged_solution, best_found_pos


def future_vision(X, centers, depth, n_threads, norm_it, sample_weight, tol, verbose, x_squared_norms):
    centers_depth = centers

    if depth == norm_it:
        labels, inertia, centers, n_iter = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                                                n_threads=n_threads)
        inertia_compare = inertia
        best_centers = centers.copy()
        best_labels = labels.copy()

    elif depth < norm_it:
        labels, inertia, centers, n_iter = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                                                n_threads=n_threads)
        inertia_compare = inertia
        best_centers = centers.copy()
        best_labels = labels.copy()

    elif depth > norm_it:
        labels, inertia, centers, n_iter = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=norm_it, verbose=verbose, tol=tol, x_squared_norms=x_squared_norms,
                                                                n_threads=n_threads)
        labels_depth, inertia_depth, centers_depth, n_iter_depth = _kmeans_single_elkan(X, sample_weight, centers.copy(), max_iter=depth - norm_it, verbose=verbose, tol=tol,
                                                                                        x_squared_norms=x_squared_norms, n_threads=n_threads)

        # in some cases a single iteration of single_elkan can make the solution worse
        # we then just return the better solution of the two
        if inertia_depth > inertia:
            inertia_depth = inertia
        inertia_compare = inertia_depth
        best_centers = centers.copy()
        best_labels = labels.copy()
    return best_centers, best_labels, centers_depth, inertia, inertia_compare


def calculate_exchanged_solution(X, candidate_id, centers, centers_new_solution, depth, inertia, j, labels_new,
                                 n_threads, norm_it, sample_weight, tol, verbose, x_squared_norms):
    old_center = centers[j].copy()  # store old center (candidate for swapping)
    centers[j] = X[candidate_id]  # swap current centers with candidate
    if depth <= norm_it:
        labels_new, inertia, centers_new_solution, n_iter_ = _kmeans_single_elkan(X, sample_weight,
                                                                                  centers.copy(),
                                                                                  max_iter=depth,
                                                                                  verbose=verbose, tol=tol,
                                                                                  x_squared_norms=x_squared_norms,
                                                                                  n_threads=n_threads)
        inertia_normit = inertia  # inertia_normit should be evaluated later

    elif depth > norm_it:
        labels_new, inertia_normit, centers_new_solution, n_iter_ = _kmeans_single_elkan(X, sample_weight,
                                                                                         centers.copy(),
                                                                                         max_iter=norm_it,
                                                                                         verbose=verbose, tol=tol,
                                                                                         x_squared_norms=x_squared_norms,
                                                                                         n_threads=n_threads)
        # if n_iter_ < norm_it:
        #   inertia = inertia_normit
        # else:
        labels_later, inertia, _, n_iter_ = _kmeans_single_elkan(X, sample_weight,
                                                                 centers_new_solution.copy(),
                                                                 max_iter=depth - norm_it,
                                                                 verbose=verbose, tol=tol,
                                                                 x_squared_norms=x_squared_norms,
                                                                 n_threads=n_threads)
    centers[j] = old_center.copy()  # undo swap

    # in some cases a single iteration of single_elkan can make the solution worse
    # we then just return the better solution of the two
    if inertia > inertia_normit:
        inertia = inertia_normit

    return centers_new_solution, inertia, inertia_normit, labels_new


def calculate_potential(X, centers, x_squared_norms=None, labels=None):

    if labels is not None:
        min_distances = paired_euclidean_distances(X, centers[labels])
        closest_dist_sq = None
    else:
        closest_dist_sq = euclidean_distances(
        centers, X, Y_norm_squared=x_squared_norms,
        squared=True)
        # closest_centers = np.argmin(closest_dist_sq, axis=0)    # Indizes der closest centers
        min_distances = closest_dist_sq.min(axis=0)  # Distanzen zu den closest centers
    current_pot = min_distances.sum()  # Summe der quadrierten Abstände zu closest centers

    return closest_dist_sq, current_pot, min_distances

def calculate_potential_clustercosts(X, centers, x_squared_norms=None, labels=None):

    if labels is not None:
        min_distances = paired_euclidean_distances(X, centers[labels])
        closest_dist_sq = None
    else:
        closest_dist_sq = euclidean_distances(
        centers, X, Y_norm_squared=x_squared_norms,
        squared=True)
        # closest_centers = np.argmin(closest_dist_sq, axis=0)    # Indizes der closest centers
        #min_distances = closest_dist_sq.min(axis=0)  # Distanzen zu den closest centers
        labels = np.argmin(closest_dist_sq, axis=0)
        min_distances = closest_dist_sq[labels, np.arange(closest_dist_sq.shape[1])]

    # collect each labeled point weighted by its distance in bins (so each bin is the clustercost)
    clustercosts = np.bincount(labels, weights=min_distances, minlength=centers.shape[0])
    current_pot = clustercosts.sum()  # Summe der quadrierten Abstände zu closest centers


    return closest_dist_sq, current_pot, min_distances, clustercosts

#################################################################################
# End of newly added content related to als++
#################################################################################


def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    x_squared_norms=None,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : ndarray of shape (n_samples,), default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(
                X,
                sample_weight,
                x_squared_norms,
                centers,
                centers_new,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
            )

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift ** 2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(
                            f"Converged at iteration {i}: center shift "
                            f"{center_shift_tot} within tolerance {tol}."
                        )
                    break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(
                X,
                sample_weight,
                x_squared_norms,
                centers,
                centers,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
                update_centers=False,
            )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _labels_inertia(X, sample_weight, x_squared_norms, centers, n_threads=1):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    weight_in_clusters = np.zeros(n_clusters, dtype=centers.dtype)
    center_shift = np.zeros_like(weight_in_clusters)

    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense
        X = ReadonlyArrayWrapper(X)

    _labels(
        X,
        sample_weight,
        x_squared_norms,
        centers,
        centers,
        weight_in_clusters,
        labels,
        center_shift,
        n_threads,
        update_centers=False,
    )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia


def _labels_inertia_threadpool_limit(
    X, sample_weight, x_squared_norms, centers, n_threads=1
):
    """Same as _labels_inertia but in a threadpool_limits context."""
    with threadpool_limits(limits=1, user_api="blas"):
        labels, inertia = _labels_inertia(
            X, sample_weight, x_squared_norms, centers, n_threads
        )

    return labels, inertia


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).

        For now "auto" (kept for backward compatibility) chooses "elkan" but it
        might change in the future for a better heuristic.

        .. versionchanged:: 0.18
            Added Elkan algorithm

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
        depth=3,
        norm_it=2,
        heuristics={},
        z=None,
        n_local_trials=None
    ):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        # new parameters for fls++
        self.heuristics = heuristics
        self.depth = depth
        self.norm_it = norm_it
        # parameter for local search
        self.z = z
        # number of trials for every iteration in D2-sampling
        self.n_local_trials = n_local_trials # greedy D2 Sampling number of trials, default = 2+log(k)

    def _check_params(self, X):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = _tolerance(X, self.tol)

        # algorithm
        if self.algorithm not in ("auto", "full", "elkan", "fls++", "ls++"):
            raise ValueError(
                "Algorithm must be 'auto', 'full', 'elkan', 'fls++', or 'ls++', "
                f"got {self.algorithm} instead."
            )

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            self._algorithm = "full" if self.n_clusters == 1 else "elkan"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                "algorithm='elkan' doesn't make sense for a single "
                "cluster. Using 'full' instead.",
                RuntimeWarning,
            )
            self._algorithm = "full"

        # init
        if not (
            hasattr(self.init, "__array__")
            or callable(self.init)
            or (isinstance(self.init, str) and self.init in ["k-means++", "random"])
        ):
            raise ValueError(
                "init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead."
            )

        if hasattr(self.init, "__array__") and self._n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _check_mkl_vcomp(self, X, n_samples):
        """Warns when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if active_threads < self._n_threads:
            modules = threadpool_info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            if has_vcomp and has_mkl:
                if not hasattr(self, "batch_size"):  # KMeans
                    warnings.warn(
                        "KMeans is known to have a memory leak on Windows "
                        "with MKL, when there are less chunks than available "
                        "threads. You can avoid it by setting the environment"
                        f" variable OMP_NUM_THREADS={active_threads}."
                    )
                else:  # MiniBatchKMeans
                    warnings.warn(
                        "MiniBatchKMeans is known to have a memory leak on "
                        "Windows with MKL, when there are less chunks than "
                        "available threads. You can prevent it by setting "
                        f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
                        "setting the environment variable "
                        f"OMP_NUM_THREADS={active_threads}"
                    )

    def _init_centroids(self, X, x_squared_norms, init, random_state, init_size=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, "__array__"):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def _init_centroids_trials(self, X, x_squared_norms, init, random_state, init_size=None, n_local_trials=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                n_local_trials=n_local_trials,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, "__array__"):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            kmeans_single = _kmeans_single_elkan

        best_inertia = None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                x_squared_norms=x_squared_norms,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # allow small tolerance on the inertia to accommodate for
            # non-deterministic rounding errors due to parallel computation
            if best_inertia is None or inertia < best_inertia * (1 - 1e-6):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
    
    #################################################################
    # fit_new works as fit, but for testing fls++ it can also take more
    # newly created parameters
    #################################################################
    


    def fit_new(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        elif self._algorithm == "fls++":
            kmeans_single = _kmeans_als_plusplus_fast
            self._check_mkl_vcomp(X, X.shape[0])
        elif self._algorithm == "ls++":
            kmeans_single = _localSearchPP
            self._check_mkl_vcomp(X, X.shape[0])

        else:
            kmeans_single = _kmeans_single_elkan

        n_local_trials = self.n_local_trials

        best_inertia = None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids_trials(
                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state, n_local_trials=n_local_trials
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            if kmeans_single == _kmeans_single_elkan or kmeans_single == _kmeans_single_lloyd:
                labels, inertia, centers, n_iter_ = kmeans_single(
                    X,
                    sample_weight,
                    centers_init,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    tol=self._tol,
                    x_squared_norms=x_squared_norms,
                    n_threads=self._n_threads,
                )
            elif kmeans_single == _localSearchPP:
                labels, inertia, centers, n_iter_ = kmeans_single(
                    X,
                    sample_weight,
                    centers_init,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    tol=self._tol,
                    x_squared_norms=x_squared_norms,
                    n_threads=self._n_threads,
                    z=self.z,
                )
            elif kmeans_single == _kmeans_als_plusplus_fast:
                labels, inertia, centers, n_iter_ = kmeans_single(
                    X,
                    sample_weight,
                    centers_init,
                    max_iter=self.max_iter,
                    verbose=self.verbose,
                    tol=self._tol,
                    x_squared_norms=x_squared_norms,
                    n_threads=self._n_threads,
                    depth=self.depth,
                    norm_it=self.norm_it,
                    heuristics=self.heuristics,
                    random_state=random_state,
                )

            # determine if these results are the best so far
            # allow small tolerance on the inertia to accommodate for
            # non-deterministic rounding errors due to parallel computation
            if best_inertia is None or inertia < best_inertia * (1 - 1e-6):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return _labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[0]

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[1]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
        }


def _mini_batch_step(
    X,
    x_squared_norms,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # Perform label assignment to nearest centers
    # For better efficiency, it's better to run _mini_batch_step in a
    # threadpool_limit context than using _labels_inertia_threadpool_limit here
    labels, inertia = _labels_inertia(
        X, sample_weight, x_squared_norms, centers, n_threads=n_threads
    )

    # Update centers according to the labels
    if sp.issparse(X):
        _minibatch_update_sparse(
            X, sample_weight, centers, centers_new, weight_sums, labels, n_threads
        )
    else:
        _minibatch_update_dense(
            ReadonlyArrayWrapper(X),
            sample_weight,
            centers,
            centers_new,
            weight_sums,
            labels,
            n_threads,
        )

    # Reassign clusters that have very low weight
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]) :]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(
                X.shape[0], replace=False, size=n_reassigns
            )
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

            if sp.issparse(X):
                assign_rows_csr(
                    X,
                    new_centers.astype(np.intp, copy=False),
                    np.where(to_reassign)[0].astype(np.intp, copy=False),
                    centers_new,
                )
            else:
                centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia


class MiniBatchKMeans(KMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster compuations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    counts_ : ndarray of shape (n_clusters,)
        Weight sum of each cluster.

        .. deprecated:: 0.24
           This attribute is deprecated in 0.24 and will be removed in
           1.1 (renaming of 0.26).

    init_size_ : int
        The effective number of samples used for the initialization.

        .. deprecated:: 0.24
           This attribute is deprecated in 0.24 and will be removed in
           1.1 (renaming of 0.26).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6)
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[2. , 1. ],
           [3.5, 4.5]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10).fit(X)
    >>> kmeans.cluster_centers_
    array([[1.19..., 1.22...],
           [4.03..., 2.46...]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
    ):

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio

    @deprecated(  # type: ignore
        "The attribute `counts_` is deprecated in 0.24"
        " and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def counts_(self):
        return self._counts

    @deprecated(  # type: ignore
        "The attribute `init_size_` is deprecated in "
        "0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def init_size_(self):
        return self._init_size

    @deprecated(  # type: ignore
        "The attribute `random_state_` is deprecated "
        "in 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def random_state_(self):
        return getattr(self, "_random_state", None)

    def _check_params(self, X):
        super()._check_params(X)

        # max_no_improvement
        if self.max_no_improvement is not None and self.max_no_improvement < 0:
            raise ValueError(
                "max_no_improvement should be >= 0, got "
                f"{self.max_no_improvement} instead."
            )

        # batch_size
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be > 0, got {self.batch_size} instead."
            )
        self._batch_size = min(self.batch_size, X.shape[0])

        # init_size
        if self.init_size is not None and self.init_size <= 0:
            raise ValueError(f"init_size should be > 0, got {self.init_size} instead.")
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                f"init_size={self._init_size} should be larger than "
                f"n_clusters={self.n_clusters}. Setting it to "
                "min(3*n_clusters, n_samples)",
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]
        x_squared_norms_valid = x_squared_norms[validation_indices]

        # perform several inits with random subsets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
            )

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid,
                sample_weight_valid,
                x_squared_norms_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size

        with threadpool_limits(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.randint(0, n_samples, self._batch_size)

                # Perform the actual update step on the minibatch data
                batch_inertia = _mini_batch_step(
                    X=X[minibatch_indices],
                    x_squared_norms=x_squared_norms[minibatch_indices],
                    sample_weight=sample_weight[minibatch_indices],
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                    i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

        self.cluster_centers_ = centers

        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                x_squared_norms,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
        else:
            self.inertia_ = self._ewa_inertia * n_samples

        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
        """
        has_centers = hasattr(self, "cluster_centers_")

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )

        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, "n_steps_", 0)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if not has_centers:
            # this instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            self._n_threads = _openmp_effective_n_threads()

            # Validate init array
            init = self.init
            if hasattr(init, "__array__"):
                init = check_array(init, dtype=X.dtype, copy=True, order="C")
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size,
            )

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            # Initialize number of samples seen since last reassignment
            self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            _mini_batch_step(
                X,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
                centers=self.cluster_centers_,
                centers_new=self.cluster_centers_,
                weight_sums=self._counts,
                random_state=self._random_state,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
                n_threads=self._n_threads,
            )

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                x_squared_norms,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )

        self.n_steps_ += 1

        return self

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        labels, _ = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            x_squared_norms,
            self.cluster_centers_,
            n_threads=self._n_threads,
        )

        return labels

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
