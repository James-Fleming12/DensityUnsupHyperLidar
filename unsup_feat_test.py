import torch
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings

from dataset.kitti.parser import Parser
from modules.network.ResNet import ResNet_34
from unsup_main import train_dglss
from modules.trainer import DGLSSTrainer

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17

FEATURE_EXTRACTOR_EPOCHS = 400

def subsample_per_class(feats: np.ndarray, labels: np.ndarray, points_per_class: int) -> tuple:
    keep_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > points_per_class:
            cls_idx = np.random.choice(cls_idx, points_per_class, replace=False)
        keep_idx.append(cls_idx)
    keep_idx = np.concatenate(keep_idx)
    return feats[keep_idx], labels[keep_idx]

def profile_dataset(train_loader, probe_batches: int = 50) -> dict:
    from collections import defaultdict

    counts: dict = defaultdict(list)
    n_seen = 0
    for i, (_, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=probe_batches, desc="Profiling dataset (labels only)"):
        labels_flat = proj_labels.reshape(-1).numpy()
        labels_flat = labels_flat[labels_flat != 0] # exclude background
        for cls in np.unique(labels_flat):
            counts[cls].append(int((labels_flat == cls).sum()))
        n_seen += 1
        if n_seen >= probe_batches:
            break
    median_per_class = {cls: int(np.median(v)) for cls, v in counts.items()}
    p10_per_class = {cls: int(np.percentile(v, 10)) for cls, v in counts.items()}
    return {
        "per_class_per_batch": dict(counts),
        "median_per_class":    median_per_class,
        "p10_per_class":       p10_per_class,
        "n_batches_total":     len(train_loader),
        "probe_batches":       n_seen,
    }
def validate_points_per_class(points_per_class_per_batch: int, profile: dict) -> int:
    median = profile["median_per_class"]
    p10 = profile["p10_per_class"]
    if not median:
        print("  [profile] No non-background classes found in probe — skipping validation.")
        return points_per_class_per_batch
    rarest_cls = min(median, key=median.get)
    most_common_cls = max(median, key=median.get)
    rarest_median = median[rarest_cls]
    most_common_median = median[most_common_cls]
    most_common_p10 = p10[most_common_cls]
    print(f"\n--- points_per_class_per_batch calibration ---")
    print(f"  Requested value      : {points_per_class_per_batch}")
    print(f"  Rarest class  (cls {rarest_cls}) median pts/batch  : {rarest_median}")
    print(f"  Most common (cls {most_common_cls}) median pts/batch: {most_common_median}")
    adjusted = points_per_class_per_batch

    if points_per_class_per_batch > rarest_median:
        print(f"  WARNING OVERSHOOT: value ({points_per_class_per_batch}) exceeds the rarest class "
              f"median ({rarest_median}). The cap will not fire for that class — it will always "
              f"contribute all its points, which is fine.")
    elif points_per_class_per_batch < most_common_p10:
        print(f"  WARNING UNDERSHOOT: value ({points_per_class_per_batch}) is below the 10th-percentile count of the most common class ({most_common_p10}).\n     You may be discarding too aggressively — consider raising it if RAM allows.")
    else:
        print(f"  OK: Value looks well-calibrated. No adjustment needed.")

    print(f"  Final value          : {adjusted}\n")
    return adjusted

def test_features(ARCH, DATA, net, points_per_class_per_batch: int = 200):
    parser = Parser(
        root=DATA_DIR,
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
        shuffle_train=True,
    )

    train_loader = parser.get_train_set()

    profile = profile_dataset(train_loader, probe_batches=50)
    points_per_class_per_batch = validate_points_per_class(points_per_class_per_batch, profile)

    net.eval()

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Extracting features"):

            feats = net(in_vol, only_feat=True)

            B, C, H, W = feats.shape
            feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_flat = proj_labels.reshape(-1).cpu().numpy()

            mask = labels_flat != 0
            feats_flat = feats_flat[mask]
            labels_flat = labels_flat[mask]

            if len(labels_flat) == 0:
                continue

            feats_flat, labels_flat = subsample_per_class(feats_flat, labels_flat, points_per_class=points_per_class_per_batch)

            all_feats.append(feats_flat)
            all_labels.append(labels_flat)

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    print(f"\nExtracted {all_feats.shape[0]:,} points across {len(np.unique(all_labels))} classes.")
    print(f"Feature dimensionality: {all_feats.shape[1]}")

    label_names = DATA.get("labels", {})
    learning_map_inv = DATA.get("learning_map_inv", {})
    id_to_name = {}
    for learned_id in np.unique(all_labels):
        orig_id = learning_map_inv.get(int(learned_id), int(learned_id))
        id_to_name[learned_id] = label_names.get(orig_id, str(learned_id))

    display_separability(all_feats, all_labels, id_to_name)

    print("\nRunning separability on encoder features (feat_map)...")
    all_enc_feats = []

    train_loader_enc = parser.get_train_set()

    net.eval()
    with torch.no_grad():
        for i, (in_vol, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in tqdm(enumerate(train_loader_enc), total=len(train_loader_enc), desc="Extracting encoder features"):
            _, _, enc_feat = net(in_vol, return_enc=True)
            B, C, H, W = enc_feat.shape
            feats_flat = enc_feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_flat = proj_labels.reshape(-1).cpu().numpy()
            mask = labels_flat != 0
            feats_flat = feats_flat[mask]
            labels_flat = labels_flat[mask]
            if len(labels_flat) == 0:
                continue
            feats_flat, labels_flat = subsample_per_class(
                feats_flat, labels_flat, points_per_class=points_per_class_per_batch
            )
            all_enc_feats.append(feats_flat)

    all_enc_feats = np.concatenate(all_enc_feats, axis=0)
    print(f"Encoder feature dimensionality: {all_enc_feats.shape[1]}")
    display_separability(all_enc_feats, all_labels, id_to_name, f_suffix="_enc")

def display_separability(feats: np.ndarray, labels: np.ndarray, id_to_name: dict = None, f_suffix: str = ""):
    if id_to_name is None:
        unique = np.unique(labels)
        id_to_name = {k: str(k) for k in unique}

    unique_labels = np.unique(labels)
    class_names = [id_to_name.get(l, str(l)) for l in unique_labels]
    n_classes = len(unique_labels)

    print("\n" + "=" * 60)
    print("  FEATURE SEPARABILITY ANALYSIS")
    print("=" * 60)

    classifiers = {
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
        ]),
        "Linear SVM (OvR)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=2000, C=1.0)),
        ]),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="svd")),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    print("\n--- Cross-Validated Accuracy (5-fold) ---")
    for name, pipe in classifiers.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, feats, labels, cv=cv, scoring="accuracy", n_jobs=-1)
        mean, std = scores.mean(), scores.std()
        cv_results[name] = (mean, std)
        bar = "█" * int(mean * 40)
        print(f"  {name:<30s}  {mean:.4f} ± {std:.4f}  |{bar}")

    best_clf_name = max(cv_results, key=lambda k: cv_results[k][0])
    print(f"\nBest classifier: {best_clf_name} ({cv_results[best_clf_name][0]:.4f})")

    best_pipe = classifiers[best_clf_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipe.fit(feats, labels)
    preds = best_pipe.predict(feats)

    print("\n--- Per-class Report (train set, best classifier) ---")
    print(classification_report(labels, preds, labels=unique_labels, target_names=class_names, zero_division=0))

    overall_mean = feats.mean(axis=0)
    fisher_ratios = {}
    for lbl in unique_labels:
        mask = labels == lbl
        mu_c = feats[mask].mean(axis=0)
        n_c = mask.sum()
        sb = n_c * np.sum((mu_c - overall_mean) ** 2)
        sw = np.sum((feats[mask] - mu_c) ** 2)
        fisher_ratios[id_to_name.get(lbl, str(lbl))] = sb / (sw + 1e-8)

    fig = plt.figure(figsize=(20, 14), facecolor="#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color = "#e8eaf6"
    accent = "#7c83fd"
    plt.rcParams.update({"text.color": text_color, "axes.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color})

    ax_cv = fig.add_subplot(gs[0, 0])
    names = list(cv_results.keys())
    means = [cv_results[n][0] for n in names]
    stds = [cv_results[n][1] for n in names]
    colors = ["#7c83fd", "#f7797d", "#43e97b"]
    bars = ax_cv.barh(names, means, xerr=stds, color=colors[:len(names)], edgecolor="none", height=0.5, capsize=4)
    ax_cv.set_xlim(0, 1)
    ax_cv.set_xlabel("Accuracy")
    ax_cv.set_title("5-Fold CV Accuracy", color=text_color, fontweight="bold")
    ax_cv.set_facecolor("#1a1d2e")
    ax_cv.spines[:].set_color("#2a2d3e")
    for bar, val in zip(bars, means):
        ax_cv.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, color=text_color)

    ax_fisher = fig.add_subplot(gs[0, 1])
    sorted_fisher = dict(sorted(fisher_ratios.items(), key=lambda x: x[1], reverse=True))
    ax_fisher.barh(list(sorted_fisher.keys()), list(sorted_fisher.values()), color=accent, edgecolor="none")
    ax_fisher.set_xlabel("Fisher Ratio (↑ = more separable)")
    ax_fisher.set_title("Per-class Separability\n(Fisher Discriminant Ratio)", color=text_color, fontweight="bold")
    ax_fisher.set_facecolor("#1a1d2e")
    ax_fisher.spines[:].set_color("#2a2d3e")

    ax_cm = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(labels, preds, labels=unique_labels, normalize="true")
    im = ax_cm.imshow(cm, cmap="magma", vmin=0, vmax=1)
    ax_cm.set_xticks(range(n_classes))
    ax_cm.set_yticks(range(n_classes))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax_cm.set_yticklabels(class_names, fontsize=7)
    ax_cm.set_title(f"Normalised Confusion Matrix\n({best_clf_name})", color=text_color, fontweight="bold")
    ax_cm.set_facecolor("#1a1d2e")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_pca = fig.add_subplot(gs[1, :2])
    n_plot = min(20_000, len(feats))
    idx = np.random.choice(len(feats), n_plot, replace=False)
    pca = PCA(n_components=2, random_state=42)
    feats_2d = pca.fit_transform(feats[idx])
    lbl_plot = labels[idx]

    cmap = plt.cm.get_cmap("tab20", n_classes)
    for ci, lbl in enumerate(unique_labels):
        mask = lbl_plot == lbl
        ax_pca.scatter(feats_2d[mask, 0], feats_2d[mask, 1], s=2, alpha=0.35, color=cmap(ci), label=id_to_name.get(lbl, str(lbl)), rasterized=True)
    ax_pca.set_title(f"PCA 2-D Feature Space  (var explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%)", color=text_color, fontweight="bold")
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    ax_pca.set_facecolor("#1a1d2e")
    ax_pca.spines[:].set_color("#2a2d3e")
    ax_pca.legend(markerscale=4, fontsize=7, loc="upper right", framealpha=0.3, facecolor="#1a1d2e", edgecolor="#2a2d3e", labelcolor=text_color, ncol=2)

    ax_dist = fig.add_subplot(gs[1, 2])
    counts = {id_to_name.get(l, str(l)): (labels == l).sum() for l in unique_labels}
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    ax_dist.barh(list(sorted_counts.keys()), list(sorted_counts.values()),
                 color="#f7797d", edgecolor="none")
    ax_dist.set_xlabel("Point count")
    ax_dist.set_title("Class Distribution\n(background excluded)", color=text_color, fontweight="bold")
    ax_dist.set_facecolor("#1a1d2e")
    ax_dist.spines[:].set_color("#2a2d3e")

    fig.suptitle("LiDAR Feature Separability Dashboard", fontsize=16, fontweight="bold", color=text_color, y=1.01)

    plt.savefig(f"feature_separability{f_suffix}.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("\nPlot saved to feature_separability.png")

def main():
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r')) # higher res
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    ARCH["train"]["batch_size"] = 16

    train_dglss(ARCH, DATA)

    w_dict = torch.load(MODEL_DIR + "/SENet_valid_best", map_location=lambda storage, loc: storage)

    net = ResNet_34(NUM_CLASSES, ARCH["train"]["aux_loss"], depth=False)
    net.load_state_dict(w_dict['state_dict'], strict=True)

    test_features(ARCH, DATA, net)

if __name__=="__main__":
    main()