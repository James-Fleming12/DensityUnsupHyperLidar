import yaml
from unsup_main import train_hdc, init_sub
from modules.trainer import DGLSSTrainer

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 # the arch config has a learning_map that maps the 32 classes to 17 (???)

MAX_HDC_EPOCHS = 20
FEATURE_EXTRACTOR_EPOCHS = 400

HD_DIM = 10000

def train_l_feature_extractor(ARCH, DATA, dist_type="standard", epochs=FEATURE_EXTRACTOR_EPOCHS):
    trainer = DGLSSTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, dist_type=dist_type, depth=True) # saves in "/logs/SENet_..."
    trainer.train(epochs=epochs)

def main():
    try:
        # ARCH = yaml.safe_load(open("config/arch/senet-1024p.yml", 'r'))
        ARCH = yaml.safe_load(open("config/arch/senet-2048p-gen.yml", 'r')) # higher res
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_new.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    # convert_dataset()

    ARCH["train"]["batch_size"] = 16

    train_l_feature_extractor(ARCH, DATA)

    ARCH["train"]["batch_size"] = 2

    hdc = train_hdc(ARCH, DATA)
    init_sub(ARCH, DATA)

if __name__ == "__main__":
    main()