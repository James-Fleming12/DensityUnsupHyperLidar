"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""

# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

# Based on: Conrad Lee <conradlee@gmail.com>
#           Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
#           Martino Sorbaro <martino.sorbaro@ed.ac.uk>

import time
import numpy as np
import math

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state, gen_batches, check_array
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin

from random import shuffle

import math

import numpy as np

import torch
from torch import exp

#seeds number intital
SEED_NUM = 128
L=2
H=8

def mean_shift_binary(X, bandwidth=None, seeds=None, cluster_all=True, GPU=True):
    X = (X > 0.5).astype(np.uint8)

    if bandwidth is None:
        bandwidth = estimate_bandwidth_binary(X)
    if not (0 < bandwidth <= 1):
        raise ValueError("bandwidth must be in (0,1] for Hamming distance")

    if seeds is None:
        seeds = gpu_seed_generator_binary(X)

    centers, density = meanshift_torch_binary(X, seeds, bandwidth)

    nbrs = NearestNeighbors(radius=bandwidth, metric="hamming")
    nbrs.fit(centers)

    unique = np.ones(len(centers), dtype=bool)
    for i, c in enumerate(centers):
        if unique[i]:
            idxs = nbrs.radius_neighbors([c], return_distance=False)[0]
            unique[idxs] = False
            unique[i] = True

    cluster_centers = centers[unique]

    nbrs = NearestNeighbors(n_neighbors=1, metric="hamming")
    nbrs.fit(cluster_centers)
    distances, labels = nbrs.kneighbors(X)

    if not cluster_all:
        labels[distances.flatten() > bandwidth] = -1

    return cluster_centers, labels.flatten()

def meanshift_torch_binary(data, seed, bandwidth, max_iter=300):
    """
    Binary mean-shift using normalized Hamming distance and majority vote.
    """
    X = torch.from_numpy(data).to(torch.uint8).cuda()
    S = torch.from_numpy(seed).to(torch.uint8).cuda()

    D = X.shape[1]
    tol = 1e-3

    for _ in range(max_iter):
        hamming = torch.bitwise_xor(
            S[:, None, :], X[None, :, :]
        ).sum(dim=2).float() / D

        weight = (hamming <= bandwidth).float()

        wsum = weight.sum(dim=1, keepdim=True)
        wsum = torch.clamp(wsum, min=1.0)

        avg = (weight @ X.float()) / wsum
        S_new = (avg > 0.5).to(torch.uint8)

        flip_frac = (
            torch.bitwise_xor(S, S_new).sum(dim=1).float() / D
        ).mean()

        S = S_new
        if flip_frac < tol:
            break

    hamming = torch.bitwise_xor(S[:, None, :], X[None, :, :]).sum(dim=2).float() / D

    p_num = (hamming <= bandwidth).sum(dim=1).tolist()

    return S.cpu().numpy(), p_num

def gpu_seed_generator_binary(codes):
    idx = np.random.permutation(len(codes))[:SEED_NUM]
    return (codes[idx] > 0.5).astype(np.uint8)

def get_binary_density_centroids(binary_vectors, bandwidth=0.2):
    """Get density centroids for binary hypervectors"""
    binary_vectors = (binary_vectors > 0.5).astype(np.uint8)

    cluster_centers, labels = mean_shift_binary(
        X=binary_vectors,
        bandwidth=bandwidth,
        GPU=True
    )
    return cluster_centers, labels

def estimate_bandwidth_binary(X, quantile=0.3, n_samples=500):
    X = (X > 0.5).astype(np.uint8)

    n = min(n_samples, X.shape[0])
    idx = np.random.choice(X.shape[0], n, replace=False)
    Xs = X[idx]

    k = max(1, int(n * quantile))

    nbrs = NearestNeighbors(n_neighbors=k, metric="hamming")
    nbrs.fit(Xs)

    distances, _ = nbrs.kneighbors(Xs)

    bandwidth = np.median(distances[:, -1])

    return float(np.clip(bandwidth, 1e-3, 1.0))

def cos_batch(a, b):
    #return sqrt(((a[None,:] - b[:,None]) ** 2).sum(2))

    num = a@b.T
    denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
    return num / denom

def get_weight(sim, bandwidth):

    thr = 1-bandwidth
    #max = torch.tensor(1.0e+10).double().cuda()
    max = torch.tensor(1.0).double().cuda()
    min = torch.tensor(0.0).double().cuda()
    #dis=torch.where(sim>thr, 1-sim, max)
    dis=torch.where(sim>thr, max, min)

    return dis

def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))

def meanshift_torch(data, seed , bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter=0

    X = torch.from_numpy(np.copy(data)).double().cuda()
    S = torch.from_numpy(np.copy(seed)).double().cuda()
    B = torch.tensor(bandwidth).double().cuda()
    
    while True:
        #cosine = cos_batch(S, X)

        weight = get_weight(cos_batch(S, X),B)

        #torch.where(distances>(1-bandwidth))
        #weight = gaussian(distances, B)
        num = (weight[:, :, None] * X).sum(dim=1)
        S_old = S
        S = num / weight.sum(1)[:, None]
        #cosine2 = torch.norm(S - S_old, dim=1).mean()
        iter+=1

        if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
            break
    
    p_num=[]
    for line in weight:
        p_num.append(line[line==1].size()[0])

    my_mean = S.cpu().numpy()

    return my_mean, p_num

def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.

    That this function takes time at least quadratic in n_samples. For large
    datasets, it's wise to set that parameter to a small value.

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Input points.

    quantile : float, default 0.3
        should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.

    n_samples : int, optional
        The number of samples to use. If not given, all samples are used.

    random_state : int, RandomState instance or None (default)
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    bandwidth : float
        The bandwidth parameter.
    """
    X = check_array(X)

    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            n_jobs=n_jobs)
    nbrs.fit(X)

    bandwidth = 0.
    for batch in gen_batches(len(X), 500):
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()

    return bandwidth / X.shape[0]


def gpu_seed_generator(codes):

     seed_indizes = list(range(codes.shape[0]))
     shuffle(seed_indizes)
     seed_indizes = seed_indizes[:SEED_NUM]
     seeds = codes[seed_indizes]
     
     return seeds

def gpu_seed_adjust(codes):
    global SEED_NUM
    SEED_NUM *= 2
    
    return gpu_seed_generator(codes)

def get_N(P,r,I):

    #There is no foreground instances
    if r<0.1:
        return 32 #Allocated some seeds at least

    lnp = math.log(P,math.e)
    num=math.log(1-math.e**(lnp/I),math.e)
    den = math.log(1-r/I,math.e)
    result = num/den

    if result<32:
        result =32 #Allocated some seeds at least
    elif result>256:
        result =256 #Our GPU memory's max limitation, you can higher it.

    return int(result)


def mean_shift_cosine(X, bandwidth=None, seeds=None, 
                      cluster_all=True, GPU=True):
    """Perform mean shift clustering of data using a flat kernel.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------

    X : array-like, shape=[n_samples, n_features]
        Input data.

    bandwidth : float, optional
        Kernel bandwidth.

        If bandwidth is not given, it is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like, shape=[n_seeds, n_features] or None
        Point used as initial kernel locations. 

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    GPU : bool, default True
        Using GPU-based faster mean-shift


    Returns
    -------

    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.

    labels : array, shape=[n_samples]
        Cluster labels for each point.
    """

    if bandwidth is None:
        bandwidth = estimate_bandwidth(X)
    elif bandwidth <= 0:
        raise ValueError("bandwidth needs to be greater than zero or None,\
            got %f" % bandwidth)
    if seeds is None:
        if GPU == True:
            seeds = gpu_seed_generator(X)
            
    
    #adjusted=False
    n_samples, n_features = X.shape
    center_intensity_dict = {}
    nbrs = NearestNeighbors(radius=bandwidth, metric='cosine').fit(X)
    #NearestNeighbors(radius=bandwidth, n_jobs=n_jobs, metric='cosine').radius_neighbors()

    global SEED_NUM
    if GPU == True:
        #GPU ver
        while True:
            labels, number = meanshift_torch(X, seeds, bandwidth)#gpu calculation
            for i in range(len(number)):
                if number[i] is not None:
                    center_intensity_dict[tuple(labels[i])] = number[i]#find out cluster

            if not center_intensity_dict:
                # nothing near seeds
                raise ValueError("No point was within bandwidth=%f of any seed."
                            " Try a different seeding strategy \
                             or increase the bandwidth."
                            % bandwidth)

            # POST PROCESSING: remove near duplicate points
            # If the distance between two kernels is less than the bandwidth,
            # then we have to remove one because it is a duplicate. Remove the
            # one with fewer points.

            sorted_by_intensity = sorted(center_intensity_dict.items(),
                                        key=lambda tup: (tup[1], tup[0]),
                                        reverse=True)
            sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
            unique = np.ones(len(sorted_centers), dtype=np.bool)
            nbrs = NearestNeighbors(radius=bandwidth, metric='cosine').fit(sorted_centers)
            for i, center in enumerate(sorted_centers):
                if unique[i]:
                    neighbor_idxs = nbrs.radius_neighbors([center],
                                                    return_distance=False)[0]
                    unique[neighbor_idxs] = 0
                    unique[i] = 1  # leave the current point as unique
            cluster_centers = sorted_centers[unique]


            # assign labels
            nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(cluster_centers)
            labels = np.zeros(n_samples, dtype=np.int)
            distances, idxs = nbrs.kneighbors(X)
            if cluster_all:
                labels = idxs.flatten()
            else:
                labels.fill(-1)
                bool_selector = distances.flatten() <= bandwidth
                labels[bool_selector] = idxs.flatten()[bool_selector]

            #Test
            #break

            bg_num = np.sum(labels==0)
            r = 1-bg_num/labels.size
            #seed number adjust
            dict_len = len(cluster_centers)#cluster number

            N= get_N(0.95,r,dict_len)

            
            if L*N <= SEED_NUM: #safety area
                #SEED_NUM -= 200#test
                if H*N  <= SEED_NUM:
                    SEED_NUM -= N #seeds are too much, adjsut
                
                break
            else:
                seeds = gpu_seed_adjust(X)#seeds are too few, adjsut
        
        return cluster_centers, labels

class MeanShiftCosine(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).

    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    GPU : bool, default True
        Using GPU-based faster mean-shift


    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point.

    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering # doctest: +NORMALIZE_WHITESPACE
    MeanShift(bandwidth=2, cluster_all=True, seeds=None)

    References
    ----------

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    """
    def __init__(self, bandwidth=None, seeds=None, cluster_all=True, GPU=True):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.cluster_all = cluster_all
        self.GPU = GPU

    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.

        y : Ignored

        """
        X = check_array(X)
        self.cluster_centers_, self.labels_ = \
            mean_shift_cosine(X, bandwidth=self.bandwidth, seeds=self.seeds,
                       cluster_all=self.cluster_all, GPU=self.GPU)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)
    
def test_binary_mean_shift():
    """Test function with binary hypervectors"""

    print("=" * 60)
    print("Testing Binary Mean Shift with 20 Hypervectors")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! This test requires GPU.")
        return

    # Reproducibility
    np.random.seed(42)

    n_bits = 64

    # -------------------------
    # Create clean base clusters
    # -------------------------
    base1 = np.random.randint(0, 2, (1, n_bits), dtype=np.uint8)
    base2 = np.random.randint(0, 2, (1, n_bits), dtype=np.uint8)
    base3 = np.random.randint(0, 2, (1, n_bits), dtype=np.uint8)

    def noisy_copies(base, n, flip_prob):
        noise = (np.random.rand(n, n_bits) < flip_prob).astype(np.uint8)
        return np.bitwise_xor(base.repeat(n, axis=0), noise)

    # Three clusters with different noise levels
    cluster1 = noisy_copies(base1, 7, flip_prob=0.10)
    cluster2 = noisy_copies(base2, 8, flip_prob=0.20)
    cluster3 = noisy_copies(base3, 5, flip_prob=0.40)

    # Combine
    binary_vectors = np.vstack([cluster1, cluster2, cluster3]).astype(np.uint8)

    print(f"Generated {binary_vectors.shape[0]} binary vectors")
    print(f"Vector shape: {binary_vectors.shape}")
    print(f"Sample vector (first 10 bits): {binary_vectors[0, :10]}")

    # -------------------------
    # Test 1: Auto bandwidth
    # -------------------------
    print("\n1. Testing with auto bandwidth estimation:")
    try:
        bandwidth = estimate_bandwidth_binary(binary_vectors)
        print(f"   Estimated bandwidth: {bandwidth:.3f}")

        start = time.time()
        centroids, labels = get_binary_density_centroids(
            binary_vectors, bandwidth=bandwidth
        )
        elapsed = time.time() - start

        print(f"   Found {len(centroids)} clusters")
        print(f"   Labels: {labels}")
        print(f"   Label histogram: {np.bincount(labels + 1)}")
        print(f"   Time: {elapsed:.4f} seconds")

        for i, c in enumerate(centroids):
            print(f"   Centroid {i} (first 10 bits): {c[:10]}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # -------------------------
    # Test 2: Moderate bandwidth
    # -------------------------
    print("\n2. Testing with bandwidth = 0.25:")
    try:
        centroids, labels = get_binary_density_centroids(
            binary_vectors, bandwidth=0.25
        )
        print(f"   Found {len(centroids)} clusters")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # -------------------------
    # Test 3: Small bandwidth
    # -------------------------
    print("\n3. Testing with small bandwidth = 0.1:")
    try:
        centroids, labels = get_binary_density_centroids(
            binary_vectors, bandwidth=0.1
        )
        print(f"   Found {len(centroids)} clusters (expected more)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

def main():
    """Main entry point"""
    print("Binary Hypervector Density Centroid Finder")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    test_binary_mean_shift()

if __name__ == "__main__":
    main()