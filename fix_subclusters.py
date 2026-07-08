import torch
import os
import yaml
from modules.aug_model import AugModel
from dataset.kitti.parser import Parser
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    DATA_KITTI = yaml.safe_load(open("config/labels/semantic-kitti.yaml", 'r'))
    
    pretrained_path = "logs/kitti_pretrain/hdc_sub.pth"
    model = AugModel(ARCH, os.path.dirname(pretrained_path), 'rp', 0, 0, 17, device, subcluster_type='continuous')
    
    # Load the current weights (which have broken subclusters)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    
    # Reset subclusters to zero before re-initializing
    model.subclusters.data.zero_()
    
    print("Initializing SemanticKITTI Dataset to fix subclusters...")
    parser_obj = Parser(root="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C/fog/heavy",
                        train_sequences=[8], valid_sequences=[8], test_sequences=None,
                        labels=DATA_KITTI["labels"], color_map=DATA_KITTI.get("color_map", {}),
                        learning_map=DATA_KITTI["learning_map"], learning_map_inv=DATA_KITTI["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"], max_points=ARCH["dataset"]["max_points"],
                        batch_size=1, workers=1, gt=True, shuffle_train=False)
                        
    dataloader = DataLoader(parser_obj.validloader.dataset, batch_size=1, shuffle=False)
    
    # Run initialization
    model.init_subclusters(dataloader, max_samples_per_class=2000, sampling_strategy='diverse')
    
    # Save the fixed checkpoint
    torch.save(model.state_dict(), pretrained_path)
    print(f"Fixed subclusters saved to {pretrained_path}")

if __name__ == "__main__":
    main()
