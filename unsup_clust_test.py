import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset.kitti.parser import Parser
from faster_mean_shift.mean_shift_cosine_gpu import mean_shift_binary

from sklearn.cluster import AgglomerativeClustering
import hdbscan
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from modules.HDC_utils import DensityModel

MODEL_DIR = "logs"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_EPOCHS = 10
MAX_HDC_EPOCHS = 10

HD_DIM = 5000

HDC_SUB_PATH = "logs/hdc_sub.pth"

def binary_mean_shift_wrapper(X, bandwidth=0.10, max_samples=2000, device="cuda"):
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    Xb = np.sign(X)
    Xb[Xb == 0] = 1

    Xb = torch.from_numpy(Xb).to(device).float()

    with torch.no_grad():
        centers, _ = mean_shift_binary(
            X_binary=Xb,
            bandwidth=bandwidth,
        )

    return centers.cpu().numpy()

def spherical_kmeans(X, k=8):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(X)
    C = km.cluster_centers_
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

def kmedoids_cosine(X, k=8):
    km = KMedoids(
        n_clusters=k,
        metric="cosine",
        init="k-medoids++"
    )
    km.fit(X)
    return X[km.medoid_indices_]

def hdbscan_cluster(X, min_cluster_size=30):
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size
    )
    labels = clusterer.fit_predict(X)

    centers = []
    for c in np.unique(labels):
        if c == -1:
            continue
        centers.append(X[labels == c].mean(axis=0))

    if len(centers) == 0:
        return X[np.random.choice(len(X), 1)]
    return np.stack(centers)

def agglomerative_cosine(X, k=8):
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average"
    )
    labels = model.fit_predict(X)
    return np.stack([X[labels == i].mean(axis=0) for i in range(k)])

def random_prototypes(X, k=8):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def evaluate_centroids(X, centers):
    C = torch.tensor(centers, device=X.device, dtype=X.dtype)

    sims = X @ C.T
    max_sim = sims.max(dim=1).values

    bipolarity = C.abs().mean().item()

    coverage = (max_sim > max_sim.quantile(0.1)).float().mean()

    if C.shape[0] > 1:
        sep = torch.pdist(C, p=2).mean().item()
    else:
        sep = 0.0

    return {
        "num_centroids": C.shape[0],
        "mean_max_sim": max_sim.mean().item(),
        "p05_sim": max_sim.quantile(0.05).item(),
        "coverage@0.3": coverage,
        "bipolarity": bipolarity,
        "separation": sep,
    }

def run_clustering_benchmark(
    embeddings,
    labels,
    class_id,
    clusterers,
    max_samples=5000,
    device="cuda",
):
    mask = labels == class_id
    X = embeddings[mask]

    if X.shape[0] == 0:
        print("  (no samples)")
        return

    if X.shape[0] > max_samples:
        idx = torch.randperm(X.shape[0])[:max_samples]
        X = X[idx]

    X_np = X.cpu().numpy()

    for name, fn in clusterers.items():
        try:
            centers = fn(X_np)
            metrics = evaluate_centroids(X, centers)

            print(f"[{name}]")
            for k, v in metrics.items():
                print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        except Exception as e:
            print(f"[{name}] FAILED: {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_mini.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    DATA['split']['train'] = [61, 103, 553, 655, 757, 796, 916, 1077, 1094, 1100]
    ARCH["train"]["batch_size"] = 2

    parser = Parser(root=DATA_DIR,
                    train_sequences=DATA["split"]["train"],
                    valid_sequences=DATA["split"]["valid"],
                    test_sequences=None,
                    labels=DATA["labels"],
                    color_map=DATA["color_map"],
                    learning_map=DATA["learning_map"],
                    learning_map_inv=DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=False)
    
    dataloader = parser.get_train_set()

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    model.eval()

    all_embeddings = []
    all_labels = []

    print("Collecting embeddings for benchmark...")
    with torch.no_grad():
        for proj_in, _, labels, *_ in dataloader:
            proj_in = proj_in.to(device)
            labels = labels.to(device)

            enc, _, _ = model.encode(proj_in)
            enc = F.normalize(enc, dim=1)

            all_embeddings.append(enc.cpu())
            all_labels.append(labels.view(-1).cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Collected {all_embeddings.shape[0]} embeddings")

    clusterers = {
        "spherical_kmeans": lambda X: spherical_kmeans(X, k=8),
        "kmedoids": lambda X: kmedoids_cosine(X, k=8),
        "hdbscan": lambda X: hdbscan_cluster(X, min_cluster_size=50),
        "agglomerative": lambda X: agglomerative_cosine(X, k=8),
        "binary_mean_shift": lambda X: binary_mean_shift_wrapper(X, bandwidth=0.06),
        "random": lambda X: random_prototypes(X, k=8),
    }

    for c in range(model.num_classes):
        print(f"\n===== Class {c} =====")
        run_clustering_benchmark(
            embeddings=all_embeddings,
            labels=all_labels,
            class_id=c,
            clusterers=clusterers,
        )

if __name__=="__main__":
    main()