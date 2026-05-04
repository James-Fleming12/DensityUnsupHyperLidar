from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import optim
from torch import nn
from modules.losses.boundary_loss import BoundaryLoss
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.network.ResNet import Adaptor

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
KITTI_DATA_DIR = "/mnt/alpha/jmfleming/KITTI"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_EPOCHS = 10
MAX_HDC_EPOCHS = 10

HD_DIM = 10000

HDC_SUB_PATH = "logs/hdc_sub.pth"

SUBSAMPLE_RATIO = 0.1

ITER_COLORS = {
    "pre":  ["#4C9BE8", "#F5A623", "#7ED321"],
    "post": ["#E8574C", "#9B59B6", "#1ABC9C"],
}

NUM_ITERS = 10

def multistep_dumbbell(all_histories, total_iters=None, file_suffix=""):
    """
    all_histories: list of history dicts.
    All iterations must share the same steps_labels.
    """
    labels = all_histories[0]["steps_labels"]
    y_pos = np.arange(len(labels))
    n_iters_plotted = len(all_histories)

    fig = plt.figure(figsize=(18, max(8, len(labels) * 0.8 + 3)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    def draw_ax(ax, metric_key, title):
        for yi in range(len(labels)):
            ax.axhspan(yi - 0.45, yi + 0.45, color="#F4F4F4", zorder=0, alpha=0.6)

        for it_idx, history in enumerate(all_histories):
            pairs = np.array(history[metric_key])
            c_pre = ITER_COLORS["pre"][it_idx % len(ITER_COLORS["pre"])]
            c_post = ITER_COLORS["post"][it_idx % len(ITER_COLORS["post"])]

            iter_num = history.get("iter_num", it_idx + 1)
            iter_label = f"Iter {iter_num}"

            ax.hlines(y_pos, pairs[:, 0], pairs[:, 1], color=c_pre, alpha=0.35, linewidth=1.5, zorder=1)

            ax.scatter(pairs[:, 0], y_pos, color=c_pre,  s=120, zorder=3, edgecolors="white", linewidths=0.8, label=f"Pre-Update ({iter_label})",  marker="o")
            ax.scatter(pairs[:, 1], y_pos, color=c_post, s=120, zorder=3, edgecolors="white", linewidths=0.8, label=f"Post-Update ({iter_label})", marker="D")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlabel("Metric Value", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, ncol=n_iters_plotted)

    draw_ax(ax1, "acc_pairs",  "Accuracy Gain per Section")
    draw_ax(ax2, "miou_pairs", "mIoU Gain per Section")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax2.tick_params(labelleft=False)

    if total_iters and n_iters_plotted < total_iters:
        shown_iters = [h.get("iter_num", i+1) for i, h in enumerate(all_histories)]
        title_suffix = f" (Showing Iters {', '.join(map(str, shown_iters))} of {total_iters})"
    else:
        title_suffix = f" ({n_iters_plotted} iterations)"

    plt.suptitle(f"Impact of Subsample Online Updates on Unseen Data{title_suffix}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = f"subsample_online_dumbbell{file_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dumbbell plot saved to {out_path}")

def get_adaptor_optimizer(model, lr):
    """Return an SGD optimizer over only the adaptor parameters."""
    adaptor_params = [p for module in model.net.modules() if isinstance(module, Adaptor) for p in module.parameters()]
    if not adaptor_params:
        raise RuntimeError("No Adaptor modules found in model.net. Ensure ResNet_34 was instantiated with use_adaptor=True.")
    return optim.SGD(adaptor_params, lr=lr), adaptor_params

def freeze_backbone(model):
    """Freeze all params except adaptors."""
    for p in model.net.parameters():
        p.requires_grad_(False)
    for module in model.net.modules():
        if isinstance(module, Adaptor):
            for p in module.parameters():
                p.requires_grad_(True)

def unfreeze_backbone(model):
    """Restore full gradient flow after a TTA section."""
    for p in model.net.parameters():
        p.requires_grad_(True)

@torch.no_grad()
def compute_train_feature_stats(model, batches, device):
    model.net.eval()
    all_feats = []
    for proj_in, _, _proj_labels, *_ in batches:
        proj_in = proj_in.to(device)
        feat = model.net(proj_in, only_feat=True)
        all_feats.append(feat.mean(dim=[0, 2, 3]))

    stacked = torch.stack(all_feats, dim=0)
    mu_tr    = stacked.mean(dim=0)
    sigma_tr = stacked.var(dim=0).clamp(min=1e-6)
    return mu_tr, sigma_tr

def tta_image_loss(feat, mu_tr, sigma_tr, mu_te):
    loss = (0.5 * (mu_tr - mu_te) ** 2 / sigma_tr).sum()
    return loss

def subsample_online_update(model, dataloader, ARCH, loss_w, section_size=100):
    """
    Supervised sections -> freeze backbone, fine-tune adaptors only (+ subcluster update).
    Unsupervised sections -> original inference_update calls only, no network gradient updates whatsoever.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_supervised = max(1, int(section_size * SUBSAMPLE_RATIO))
    n_unsupervised = section_size - n_supervised

    print(f"Section size: {section_size} | Supervised per section: {n_supervised} | Unsupervised: {n_unsupervised}")

    all_histories = []

    for iteration in range(NUM_ITERS):
        print(f"\n{'='*60}")
        print(f"  DATASET ITERATION {iteration + 1} / {NUM_ITERS}")
        print(f"{'='*60}")

        history = {
            "iter_num": iteration + 1,
            "steps_labels": [],
            "acc_pairs": [],
            "miou_pairs": [],
        }

        section_idx = 0
        batch_iter = iter(dataloader)
        exhausted = False

        while not exhausted:
            section_start = section_idx * section_size

            supervised_batches = []
            for _ in range(n_supervised):
                try:
                    supervised_batches.append(next(batch_iter))
                except StopIteration:
                    exhausted = True
                    break

            if not supervised_batches:
                break

            section_end = section_start + section_size
            label = f"Batches {section_start}-{section_end}"
            history["steps_labels"].append(label)
            print(f"\n--- Section: {label} ---")

            acc_pre, miou_pre = _eval_on_batches(model, supervised_batches, device)
            print(f"  Pre-update  | Acc: {acc_pre:.4f}  mIoU: {miou_pre:.4f}")

            print(f"  Fine-tuning adaptors on {len(supervised_batches)} supervised batches...")
            freeze_backbone(model)
            model.net.train()

            optimizer, _ = get_adaptor_optimizer(model, lr=ARCH["train"]["decay"]["lr"])

            criterion = nn.NLLLoss(weight=loss_w, ignore_index=0).to(device)
            ls = Lovasz_softmax(ignore=0).to(device)
            bd = BoundaryLoss().to(device)
            scaler = torch.amp.GradScaler("cuda")

            for proj_in, _, proj_labels, *_ in supervised_batches:
                proj_in = proj_in.to(device)
                proj_labels = proj_labels.to(device).long().squeeze(1)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    output = model.net(proj_in)
                    pred = output[0] if isinstance(output, tuple) else output

                    bdloss = bd(pred, proj_labels)
                    nll_loss = criterion(torch.log(pred.clamp(min=1e-8)), proj_labels)
                    lovasz_loss = 1.5 * ls(pred, proj_labels)
                    loss = (nll_loss + lovasz_loss + bdloss).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                model.update_subclusters(proj_in, proj_labels)

            unfreeze_backbone(model)
            model.net.eval()

            print(f"  Inference updating on {n_unsupervised} unsupervised batches...")
            model.train()
            for _ in tqdm(range(n_unsupervised)):
                try:
                    proj_in, *_ = next(batch_iter)
                except StopIteration:
                    exhausted = True
                    break

                if proj_in.shape[1] > 0:
                    model.inference_update(proj_in.to(device), learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.60, 0.80])

            acc_post, miou_post = _eval_on_batches(model, supervised_batches, device)
            print(f"  Post-update | Acc: {acc_post:.4f}  mIoU: {miou_post:.4f}")

            history["acc_pairs"].append((acc_pre, acc_post))
            history["miou_pairs"].append((miou_pre, miou_post))

            section_idx += 1
            del supervised_batches

        all_histories.append(history)

    if len(all_histories) >= 3:
        plot_histories = [all_histories[0], all_histories[1], all_histories[-1]]
    else:
        plot_histories = all_histories

    multistep_dumbbell(plot_histories, total_iters=NUM_ITERS)
    
    return all_histories

def _eval_on_batches(model, batches, device):
    model.eval()
    class_intersection = torch.zeros(model.num_classes, device=device)
    class_union = torch.zeros(model.num_classes, device=device)
    global_correct = 0
    global_total = 0

    with torch.no_grad():
        for proj_in, _, proj_labels, *_ in batches:
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device)

            logits, _, indices, _ = model(proj_in, PERCENTAGE=None, is_wrong=None)
            predictions = torch.argmax(logits, dim=1)

            proj_labels_flat  = proj_labels.view(-1)
            selected_labels = proj_labels_flat[indices]

            global_correct += (predictions == selected_labels).sum().item()
            global_total += selected_labels.size(0)

            for class_id in range(model.num_classes):
                class_mask = selected_labels == class_id
                pred_mask = predictions == class_id
                class_intersection[class_id] += (class_mask & pred_mask).sum().item()
                class_union[class_id] += (class_mask | pred_mask).sum().item()

    accuracy = global_correct / global_total if global_total > 0 else 0.0
    valid_ious = [(class_intersection[c] / class_union[c]).item() for c in range(model.num_classes) if class_union[c] > 0]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    return accuracy, miou

def main():
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        KITTI_DATA = yaml.safe_load(open("config/labels/semantic-kitti-all.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    ARCH["train"]["batch_size"] = 1

    kitti_parser = Parser(root=KITTI_DATA_DIR,
                    train_sequences=KITTI_DATA["split"]["train"],
                    valid_sequences=KITTI_DATA["split"]["valid"],
                    test_sequences=KITTI_DATA["split"]["test"],
                    labels=KITTI_DATA["labels"],
                    color_map=KITTI_DATA["color_map"],
                    learning_map=KITTI_DATA["learning_map"],
                    learning_map_inv=KITTI_DATA["learning_map_inv"],
                    sensor=ARCH["dataset"]["sensor"],
                    max_points=ARCH["dataset"]["max_points"],
                    batch_size=ARCH["train"]["batch_size"],
                    workers=ARCH["train"]["workers"],
                    gt=True,
                    shuffle_train=True)

    kittiloader = kitti_parser.get_valid_set()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epsilon_w = ARCH["train"]["epsilon_w"]
    content = torch.zeros(kitti_parser.get_n_classes(), dtype=torch.float)
    for cl, freq in KITTI_DATA["content"].items():
        x_cl = kitti_parser.to_xentropy(cl)
        content[x_cl] += freq
    loss_w = 1 / (content + epsilon_w)
    
    for x_cl, w in enumerate(loss_w):
        if KITTI_DATA["learning_ignore"][x_cl]:
            loss_w[x_cl] = 0
            
    loss_w = loss_w.to(device)

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    _ = subsample_online_update(model, kittiloader, ARCH, loss_w)

if __name__=="__main__":
    main()