import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from modules.trainer import Trainer
from modules.Basic_HD import DenseHDTrainer
from modules.ioueval import iouEval

import numpy as np

from unsup_main import train_extractor, train_hdc

MODEL_DIR = "logs"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_EPOCHS = 10
MAX_HDC_EPOCHS = 10

HD_DIM = 5000

HDC_SUB_PATH = "logs/hdc_sub.pth"

def test_collapse(ARCH, DATA, inference_epochs=5):
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
    
    trainloader = parser.get_train_set()

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES, hd_dim=HD_DIM, device=device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))

    model.to(device)

    all_accuracies = []
    all_class_accuracies = {}

    model.eval()
    with torch.no_grad():
        for _, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device)

            curr_acc, _, curr_class_accs = model.get_accuracy(proj_in, proj_labels)
            all_accuracies.append(curr_acc)

            for class_id, class_acc in curr_class_accs.items():
                if class_id not in all_class_accuracies:
                    all_class_accuracies[class_id] = []
                all_class_accuracies[class_id].append(class_acc)

        init_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0

        init_class_accuracy = {}
        for class_id, acc_list in all_class_accuracies.items():
            init_class_accuracy[class_id] = np.mean(acc_list)

        print(f"Beginning Accuracy of {init_accuracy}")
        for i in init_class_accuracy:
            print(f"Accuracy for class {i} is {init_class_accuracy[i]}")

    model.train()
    for epoch in range(inference_epochs):
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            proj_in = proj_in.to(device)
            for i in proj_in:
                model.inference_update(i.unsqueeze(0))

    all_accuracies = []
    all_class_accuracies = {}

    # evaluate
    model.eval()
    with torch.no_grad():
        for _, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device)

            curr_acc, _, curr_class_accs = model.get_accuracy(proj_in, proj_labels)
            all_accuracies.append(curr_acc)

            for class_id, class_acc in curr_class_accs.items():
                if class_id not in all_class_accuracies:
                    all_class_accuracies[class_id] = []
                all_class_accuracies[class_id].append(class_acc)

        accuracy = np.mean(all_accuracies) if all_accuracies else 0.0

        class_accuracy = {}
        for class_id, acc_list in all_class_accuracies.items():
            class_accuracy[class_id] = np.mean(acc_list)

        print(f"Final Accuracy of {accuracy} from {init_accuracy}")
        for i in class_accuracy:
            print(f"Accuracy for class {i} is {class_accuracy[i]} from {init_class_accuracy[i]}")

def main():
    # A code snippet to test model collapse in the model after updating over the training set/test set
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

    test_collapse(ARCH, DATA)

if __name__=="__main__":
    main()