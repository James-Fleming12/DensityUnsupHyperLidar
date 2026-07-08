import torch
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.aug_model import AugModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    pretrained_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = AugModel(ARCH, os.path.dirname(pretrained_path), 'rp', 0, 0, 17, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    model.eval()

    print("Testing 32x2048 forward pass...")
    try:
        dummy_input = torch.randn(1, 5, 32, 2048).to(device)
        with torch.no_grad():
            out = model(dummy_input)
        print("Success! The network is fully convolutional and supports 32x2048.")
    except Exception as e:
        print(f"Failed! {e}")

if __name__ == "__main__":
    main()
