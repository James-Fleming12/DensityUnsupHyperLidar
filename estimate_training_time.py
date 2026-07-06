import os
import argparse

def count_frames(base_dir, sequences):
    total_frames = 0
    seq_counts = {}
    
    for seq in sequences:
        seq_str = f"{seq:02d}"
        velodyne_dir = os.path.join(base_dir, "dataset", "sequences", seq_str, "velodyne")
        
        if not os.path.exists(velodyne_dir):
            # Some datasets might not have a 'dataset' subfolder
            velodyne_dir = os.path.join(base_dir, "sequences", seq_str, "velodyne")
            
        if os.path.exists(velodyne_dir):
            try:
                # Count only .bin files
                frames = len([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
                total_frames += frames
                seq_counts[seq_str] = frames
            except Exception as e:
                print(f"Error reading sequence {seq_str}: {e}")
        else:
            print(f"Warning: Could not find velodyne directory for sequence {seq_str} in {base_dir}")
            
    return total_frames, seq_counts

def main():
    parser = argparse.ArgumentParser(description="Estimate Synth4D vs KITTI Training Time via Frame Count")
    parser.add_argument('--kitti_dir', type=str, default='/mnt/alpha/jmfleming/KITTI', help='Path to real KITTI dataset')
    parser.add_argument('--synth4d_dir', type=str, default='/mnt/alpha/jmfleming/Synth4D', help='Path to Synth4D dataset')
    args = parser.parse_args()

    # The training split used in unsup_kitti-c.py (00-07, 09-10)
    train_sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

    print("="*60)
    print(f"Scanning SemanticKITTI (Targeting {len(train_sequences)} Train Sequences)...")
    kitti_total, kitti_seqs = count_frames(args.kitti_dir, train_sequences)
    print(f"SemanticKITTI Total Train Frames: {kitti_total:,}")
    
    print("\n" + "="*60)
    print(f"Scanning Synth4D (Targeting {len(train_sequences)} Train Sequences)...")
    synth_total, synth_seqs = count_frames(args.synth4d_dir, train_sequences)
    print(f"Synth4D Total Train Frames: {synth_total:,}")

    print("\n" + "="*60)
    print("ESTIMATE:")
    if kitti_total > 0 and synth_total > 0:
        ratio = synth_total / kitti_total
        print(f"Synth4D Frame Ratio vs KITTI: {ratio:.2f}x")
        print("\nTime Scaling:")
        print(f"If 1 epoch on KITTI took X minutes, 1 epoch on Synth4D will take roughly {ratio:.2f} * X minutes.")
        print(f"If your full KITTI pretraining took Y hours, Synth4D will take roughly {ratio:.2f} * Y hours.")
        print("\nNote: Since your previous KITTI pretraining also used unsup_kitti-c.py with the same batch sizes (24/6),")
        print("the time will scale purely based on this frame ratio.")
    else:
        print("Could not compute estimate. Ensure both dataset paths are correct and accessible.")
    print("="*60)

if __name__ == "__main__":
    main()
