import os
import yaml

from modules.HDC_utils import DensityModel
from faster_mean_shift.mean_shift_cosine_gpu import get_binary_density_centroids
from modules.trainer import TrainingPipeline
from modules.Basic_HD import DenseHDTrainer

MODEL_DIR = "models/model.pth"
DATA_DIR = "data"
LOG_DIR = "logs"
NUM_CLASSES = 10 # dont know yet

MAX_EPOCHS = 10

def train_extractor(ARCH, DATA):
    trainer = TrainingPipeline(ARCH, DATA, DATA_DIR, LOG_DIR) # saves in "/models/model.pth"
    trainer.train()

def train_hdc(ARCH, DATA):
    dataloader = None # figure out how data is going to work

    model = DensityModel(ARCH, MODEL_DIR, NUM_CLASSES)
    trainer = DenseHDTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR)

    trainer.train(dataloader, model)
    

def main():
    try: # open arch config file
        ARCH = yaml.safe_load(open("config/arch/senet-512.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:    # open data config file
        DATA = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()
    
    print(ARCH)
    print(DATA)

    # train_extractor(ARCH, DATA)
    # train_hdc(ARCH, DATA)

if __name__=="__main__":
    main()