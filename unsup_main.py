import numpy as np
import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel, Model, NewModel
from modules.trainer import Trainer
from modules.Basic_HD import DenseHDTrainer, BasicHD, NewDenseTrain
from modules.ioueval import iouEval

from dataset.export_semantickitti import KittiConverter

MODEL_DIR = "logs"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 10
FEATURE_EXTRACTOR_EPOCHS = 250

HD_DIM = 5000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def convert_dataset():
    converter = KittiConverter(
        nusc_dir=NU_DATA_DIR,
        nusc_skitti_dir=DATA_DIR,
        lidar_name='LIDAR_TOP',
        nusc_version='v1.0-mini'
    )

    converter.nuscenes_gt_to_semantickitti()

    print("Conversion Complete: Output Saved to ")

def train_extractor(ARCH, DATA):
    trainer = Trainer(ARCH, DATA, DATA_DIR, LOG_DIR) # saves in "/logs/SENet_..."
    trainer.train(epochs=FEATURE_EXTRACTOR_EPOCHS)

def train_hdc(ARCH, DATA) -> DensityModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=2,
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()
    val_loader = parser.get_valid_set() # val_loader is empty???

    ignore = []
    for cl, ign in DATA['learning_ignore'].items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)

    evaluator = iouEval(NUM_CLASSES, device, ignore)

    trainer = NewDenseTrain(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(1):
        trainer.retrain(dataloader, trainer.model, i+1, None)

    model: NewModel = trainer.model
    torch.save(model, HDC_SAVE_PATH)

    test_hdc_model(model, dataloader)

    return model

def test_hdc_model(model, dataloader) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_accuracies = []
    class_correct = torch.zeros(model.num_classes, device=device)
    class_total = torch.zeros(model.num_classes, device=device)
    global_correct = 0
    global_total = 0

    model.eval()
    
    with torch.no_grad():
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(dataloader):
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device)

            logits, _, indices, _ = model(proj_in, PERCENTAGE=None, is_wrong=None)
            predictions = torch.argmax(logits, dim=1)

            proj_labels_flat = proj_labels.view(-1)
            selected_labels = proj_labels_flat[indices]

            batch_correct = (predictions == selected_labels).sum().item()
            batch_total = selected_labels.size(0)
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
            
            all_accuracies.append(batch_accuracy)
            global_correct += batch_correct
            global_total += batch_total

            for class_id in range(model.num_classes):
                class_mask = (selected_labels == class_id)
                if class_mask.any():
                    class_correct[class_id] += (predictions[class_mask] == class_id).sum().item()
                    class_total[class_id] += class_mask.sum().item()

    global_accuracy = global_correct / global_total if global_total > 0 else 0
    mean_batch_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    
    per_class_accuracy = {}
    for class_id in range(model.num_classes):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = (class_correct[class_id] / class_total[class_id]).item()
        else:
            per_class_accuracy[class_id] = 0.0

    print(f"\n{'='*60}")
    print("Training Set Accuracy Results")
    print(f"{'='*60}")
    print(f"Global Accuracy: {global_accuracy:.4f} ({global_correct}/{global_total})")
    print(f"Mean Batch Accuracy: {mean_batch_accuracy:.4f}")
    print()
    print("Per-Class Accuracies:")

    for class_id in sorted(range(model.num_classes)):
        if class_total[class_id] > 0:
            acc = per_class_accuracy[class_id]
            correct = int(class_correct[class_id].item())
            total = int(class_total[class_id].item())
            print(f"  Class {class_id}: {acc:.4f} ({correct}/{total})")
        else:
            print(f"  Class {class_id}: No samples")

def test_orig(ARCH, DATA) -> Model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                    shuffle_train=True)
    
    dataloader = parser.get_train_set()
    val_loader = parser.get_valid_set()

    ignore = []
    for cl, ign in DATA['learning_ignore'].items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)

    evaluator = iouEval(NUM_CLASSES, device, ignore)

    trainer = BasicHD(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(1):
        trainer.retrain(dataloader, trainer.model, i+1, None)

    model: Model = trainer.model
    torch.save(trainer.model, HDC_SAVE_PATH)

    model.eval()
    all_accuracies = []
    class_correct = torch.zeros(model.num_classes, device=device)
    class_total = torch.zeros(model.num_classes, device=device)
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(dataloader):
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device)

            logits, _, indices, _ = model(proj_in, PERCENTAGE=None, is_wrong=None)
            predictions = torch.argmax(logits, dim=1)

            proj_labels_flat = proj_labels.view(-1)
            selected_labels = proj_labels_flat[indices]

            batch_correct = (predictions == selected_labels).sum().item()
            batch_total = selected_labels.size(0)
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
            all_accuracies.append(batch_accuracy)
            
            total_correct += batch_correct
            total_samples += batch_total

            for class_id in range(model.num_classes):
                class_mask = (selected_labels == class_id)
                if class_mask.any():
                    class_correct[class_id] += (predictions[class_mask] == class_id).sum().item()
                    class_total[class_id] += class_mask.sum().item()

    mean_batch_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0

    global_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    per_class_accuracy = {}
    
    for class_id in range(model.num_classes):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = (class_correct[class_id] / class_total[class_id]).item()
        else:
            per_class_accuracy[class_id] = 0.0
    
    print(f"\n{'='*60}")
    print("Training Set Accuracy Results")
    print(f"{'='*60}")
    print(f"Global Accuracy: {global_accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"Mean Batch Accuracy: {mean_batch_accuracy:.4f}")
    
    print("\nPer-Class Accuracies:")
    for class_id in sorted(per_class_accuracy.keys()):
        if class_total[class_id] > 0:
            print(f"  Class {class_id}: {per_class_accuracy[class_id]:.4f} "
                  f"({int(class_correct[class_id])}/{int(class_total[class_id])})")
        else:
            print(f"  Class {class_id}: No samples")

    # trainer.validate(val_loader, model, evaluator)

    return model

def init_sub(ARCH, DATA):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
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
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()

    model: NewModel = NewModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model = torch.load(HDC_SAVE_PATH, weights_only=False)

    model.init_subclusters(dataloader)
    torch.save(model.state_dict(), HDC_SUB_PATH)

    print(f"Subcluster Initialized Model saved to {HDC_SUB_PATH}")

def test_inference(ARCH, DATA):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
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
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()

    model: NewModel = NewModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    images, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = next(iter(dataloader))

    image = images[0].to(device).unsqueeze(0)

    model.inference_update(image)

def main():
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-1024p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_mini.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    # DATA['split']['train'] = [61, 103, 553, 655]
    DATA['split']['train'] = [61, 103, 553, 655, 757, 796, 916, 1077, 1094, 1100]
    ARCH["train"]["batch_size"] = 2

    # convert_dataset()
    # train_extractor(ARCH, DATA)
    hdc = train_hdc(ARCH, DATA)
    init_sub(ARCH, DATA)
    test_inference(ARCH, DATA)

    # test_orig(ARCH, DATA)

if __name__=="__main__":
    main()