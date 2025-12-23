import os
import yaml

from modules.HDC_utils import DensityModel
from faster_mean_shift.mean_shift_cosine_gpu import get_binary_density_centroids
from modules.trainer import Trainer, TrainingPipeline
from modules.Basic_HD import DenseHDTrainer

from dataset.export_semantickitti import KittiConverter

MODEL_DIR = "models/model.pth"
NU_DATA_DIR = "v1.0-mini"
DATA_DIR = "nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 23 # dont know yet

MAX_EPOCHS = 10

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
    trainer = Trainer(ARCH, DATA, DATA_DIR, LOG_DIR) # saves in "/models/model.pth"
    trainer.train()

def train_hdc(ARCH, DATA):
    dataloader = None # figure out how data is going to work

    model = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES)
    trainer = DenseHDTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR)

    trainer.train(dataloader, model)

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
    train_extractor(ARCH, DATA)
    # train_hdc(ARCH, DATA)

if __name__=="__main__":
    main()