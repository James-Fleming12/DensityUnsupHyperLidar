import os
import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from faster_mean_shift.mean_shift_cosine_gpu import get_binary_density_centroids
from modules.trainer import Trainer, TrainingPipeline
from modules.Basic_HD import DenseHDTrainer
from modules.ioueval import iouEval

from dataset.export_semantickitti import KittiConverter

MODEL_DIR = "logs"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_EPOCHS = 10
MAX_HDC_EPOCHS = 10

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
    trainer.train()

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
                        batch_size=ARCH["train"]["batch_size"],
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

    model = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES, hd_dim=HD_DIM, device=device)
    trainer = DenseHDTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, hd_dim=HD_DIM)

    trainer.train(dataloader, model)

    for i in range(MAX_HDC_EPOCHS):
        trainer.retrain(dataloader, model, i+1)

    torch.save(model, HDC_SAVE_PATH)

    # trainer.validate(val_loader, model, evaluator)

    return model

def test_init(ARCH, DATA):
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

    model = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES, hd_dim=HD_DIM, device=device)
    model: DensityModel = torch.load(HDC_SAVE_PATH, weights_only=False)

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

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES, hd_dim=HD_DIM, device=device)
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    images, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = next(iter(dataloader))

    image = images[0].to(device).unsqueeze(0)

    model.update(image)

def main():
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

    DATA['split']['train'] = [61, 103, 553, 655]

    # convert_dataset()
    # train_extractor(ARCH, DATA)
    # hdc = train_hdc(ARCH, DATA)
    # test_init(ARCH, DATA)
    test_inference(ARCH, DATA)

if __name__=="__main__":
    main()