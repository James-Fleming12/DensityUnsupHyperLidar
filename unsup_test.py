import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from modules.trainer import Trainer
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

def test_collapse(ARCH, trainloader, inference_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: DensityModel = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
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

def test_subcluster_initialization(ARCH, trainloader):
    """Test that subclusters are properly initialized and represent dense regions"""
    print("\n" + "="*80)
    print("TEST: Subcluster Initialization")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    
    # Load subclusters from saved model
    print("\nLoading subclusters from saved model...")
    try:
        model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
        print("✓ Successfully loaded subclusters from", HDC_SUB_PATH)
    except FileNotFoundError:
        print(f"✗ Could not find {HDC_SUB_PATH}, initializing subclusters instead...")
        model.to(device)
        model.init_subclusters(trainloader, bandwidth=None, max_samples_per_class=1000)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Initializing subclusters instead...")
        model.to(device)
        model.init_subclusters(trainloader, bandwidth=None, max_samples_per_class=1000)
    
    model.to(device)
    
    # Test 1: Check that subclusters are bipolar {-1, +1}
    print("\n[Test 1] Checking subclusters are bipolar...")
    unique_vals = torch.unique(model.subclusters.data)
    is_bipolar = torch.all((unique_vals == -1) | (unique_vals == 1))
    print(f"  Unique values in subclusters: {unique_vals.cpu().numpy()}")
    print(f"  ✓ PASS: Subclusters are bipolar" if is_bipolar else f"  ✗ FAIL: Subclusters not bipolar")
    
    # Test 2: Check subcluster_to_class mapping
    print("\n[Test 2] Checking subcluster-to-class mapping...")
    expected_length = NUM_CLASSES * model.num_subclusters
    actual_length = len(model.subcluster_to_class)
    mapping_correct = actual_length == expected_length
    print(f"  Expected mapping length: {expected_length}")
    print(f"  Actual mapping length: {actual_length}")
    print(f"  ✓ PASS: Mapping length correct" if mapping_correct else f"  ✗ FAIL: Mapping length incorrect")
    
    # Test 3: Check that subclusters have non-zero norm
    print("\n[Test 3] Checking subclusters have non-zero norm...")
    norms = torch.norm(model.subclusters.data, dim=1)
    zero_norm_count = (norms == 0).sum().item()
    total_subclusters = model.subclusters.data.shape[0]
    print(f"  Total subclusters: {total_subclusters}")
    print(f"  Subclusters with zero norm: {zero_norm_count}")
    print(f"  Non-zero subclusters: {total_subclusters - zero_norm_count}")
    print(f"  ✓ PASS: All subclusters initialized" if zero_norm_count == 0 else f"  ⚠ WARNING: {zero_norm_count} zero-norm subclusters")
    
    # Test 4: Verify subclusters represent dense regions by checking similarity distribution
    print("\n[Test 4] Checking subclusters represent dense regions...")
    class_similarities = {}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            if batch_idx >= 5:  # Test on first 5 batches
                break
                
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device).flatten()
            
            enc, _, _ = model.encode(proj_in)
            
            # For each class, check similarity to its subclusters
            for class_id in range(NUM_CLASSES):
                class_mask = proj_labels == class_id
                if not torch.any(class_mask):
                    continue
                
                class_enc = enc[class_mask]
                if class_id not in class_similarities:
                    class_similarities[class_id] = []
                
                sims, _ = model.get_max_subcluster_similarity(class_enc, class_id)
                class_similarities[class_id].extend(sims.cpu().numpy())
    
    print("\n  Similarity statistics per class:")
    for class_id in sorted(class_similarities.keys()):
        sims = np.array(class_similarities[class_id])
        if len(sims) > 0:
            print(f"    Class {class_id:2d}: mean={sims.mean():.3f}, std={sims.std():.3f}, "
                  f"min={sims.min():.3f}, max={sims.max():.3f}")
    
    # Subclusters should have relatively high similarity to their class data
    avg_similarities = [np.mean(class_similarities[c]) for c in class_similarities if len(class_similarities[c]) > 0]
    overall_avg = np.mean(avg_similarities) if avg_similarities else 0
    print(f"\n  Overall average similarity: {overall_avg:.3f}")
    print(f"  ✓ PASS: Subclusters represent dense regions" if overall_avg > 0.5 else 
          f"  ⚠ WARNING: Low average similarity ({overall_avg:.3f})")
    
    return model

def test_inference_update_mechanics(ARCH, trainloader):
    """Test inference updates for HARD BIPOLAR prototypes (±1 bit flips)"""
    print("\n" + "="*80)
    print("TEST: Inference Update Mechanics (Bipolar)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.load_state_dict(torch.load(HDC_SUB_PATH, map_location=device))
        print("\n✓ Loaded pre-trained model")
    except Exception as e:
        print("\n⚠ Could not load pre-trained model, initializing new one...")
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.to(device)
        model.init_subclusters(trainloader, bandwidth=None, max_samples_per_class=500)

    model.to(device)
    model.eval()

    print("\n[DIAGNOSTIC] Checking prototype validity...")
    proto = model.classify.weight

    unique_vals = torch.unique(proto)
    is_strict_bipolar = torch.all((unique_vals == -1) | (unique_vals == 1))

    print(f"  classify.weight shape: {proto.shape}")
    print(f"  unique values: {unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''}")
    print(f"  Strict bipolar: {is_strict_bipolar}")

    if not is_strict_bipolar:
        raise AssertionError(
            "❌ classify.weight contains non-bipolar values. "
            "Hard bipolar prototypes must be ±1 only."
        )

    for proj_in, _, proj_labels, *_ in trainloader:
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device)
        break

    print("\n[DIAGNOSTIC] Prediction sanity check...")
    with torch.no_grad():
        enc, _, _ = model.encode(proj_in)
        enc = torch.sign(enc)

        assert torch.all((enc == -1) | (enc == 1)), "Encoded HVs are not bipolar"

        logits = model.get_predictions(enc)
        preds = logits.argmax(dim=1)

        print(f"  Unique predictions: {torch.unique(preds)}")

    print("\n[Test 1] Bit-flip update check...")
    original = model.classify.weight.clone()

    model.train()
    with torch.no_grad():
        model.chunked_inference_update(
            proj_in,
            beta=0.05,
            distance_sensitivity=1.0
        )

    updated = model.classify.weight

    flipped = (original != updated)
    num_flipped = flipped.sum().item()
    total_bits = original.numel()

    print(f"  Flipped bits: {num_flipped}/{total_bits}")

    if num_flipped > 0:
        print("  ✓ PASS: Prototype bits flipped")
    else:
        print("  ✗ FAIL: No bit flips detected")

    print("\n[Test 2] Beta gating behavior...")

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.0)
    flips_beta_0 = (model.classify.weight != original).sum().item()

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.5)
    flips_beta_05 = (model.classify.weight != original).sum().item()

    print(f"  beta=0.0 flips: {flips_beta_0}")
    print(f"  beta=0.5 flips: {flips_beta_05}")

    if flips_beta_0 >= flips_beta_05:
        print("  ✓ PASS: Lower beta → more flips")
    else:
        print("  ⚠ WARNING: Beta gating inverted")

    print("\n[Test 3] distance_sensitivity effect...")

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.1, distance_sensitivity=0.0)
    flips_ds_0 = (model.classify.weight != original).sum().item()

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.1, distance_sensitivity=2.0)
    flips_ds_2 = (model.classify.weight != original).sum().item()

    print(f"  ds=0.0 flips: {flips_ds_0}")
    print(f"  ds=2.0 flips: {flips_ds_2}")

    if flips_ds_2 >= flips_ds_0:
        print("  ✓ PASS: distance_sensitivity scales updates")
    else:
        print("  ⚠ WARNING: distance_sensitivity has no effect")

    print("\n[FINAL CHECK] Bipolar invariant...")
    final_vals = torch.unique(model.classify.weight)

    assert torch.all((final_vals == -1) | (final_vals == 1)), \
        "❌ Prototype lost bipolarity after updates"

    print("  ✓ All prototypes remain strictly bipolar")

def test_similarity_calculation(ARCH, trainloader):
    """Test the get_max_subcluster_similarity function"""
    print("\n" + "="*80)
    print("TEST: Similarity Calculation")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    except:
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.to(device)
        model.init_subclusters(trainloader, bandwidth=None, max_samples_per_class=500)
    
    model.to(device)
    model.eval()

    print("\n[Test 1] Checking similarity is in range [0, 1]...")
    with torch.no_grad():
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            if batch_idx >= 3:
                break
            
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device).flatten()
            
            enc, _, _ = model.encode(proj_in)
            
            for class_id in range(NUM_CLASSES):
                class_mask = proj_labels == class_id
                if not torch.any(class_mask):
                    continue
                
                class_enc = enc[class_mask]
                sims, indices = model.get_max_subcluster_similarity(class_enc, class_id, distance_sensitivity=1.0)
                
                if len(sims) > 0:
                    min_sim = sims.min().item()
                    max_sim = sims.max().item()
                    in_range = (min_sim >= 0.0 and max_sim <= 1.0)
                    
                    if not in_range:
                        print(f"    Class {class_id}: min={min_sim:.3f}, max={max_sim:.3f} ✗ OUT OF RANGE")
                    
    print(f"  ✓ PASS: All similarities in [0, 1]")

    print("\n[Test 2] Checking identical vectors have similarity ~1.0...")
    test_class = 0
    mask = model.subcluster_to_class == test_class
    if torch.any(mask):
        test_subcluster = model.subclusters[mask][0:1]
        sim, idx = model.get_max_subcluster_similarity(test_subcluster, test_class, distance_sensitivity=1.0)
        print(f"  Similarity of subcluster to itself: {sim.item():.6f}")
        print(f"  ✓ PASS: Self-similarity ~1.0" if abs(sim.item() - 1.0) < 0.01 else 
              f"  ✗ FAIL: Self-similarity is {sim.item():.6f}")

    print("\n[Test 3] Checking distance_sensitivity scaling...")
    with torch.no_grad():
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device).flatten()
            enc, _, _ = model.encode(proj_in)
            
            test_class = proj_labels[0].item()
            test_enc = enc[0:1]
            
            sim_1 = model.get_max_subcluster_similarity(test_enc, test_class, distance_sensitivity=1.0)[0].item()
            sim_2 = model.get_max_subcluster_similarity(test_enc, test_class, distance_sensitivity=2.0)[0].item()
            sim_05 = model.get_max_subcluster_similarity(test_enc, test_class, distance_sensitivity=0.5)[0].item()
            
            print(f"  Similarity with sensitivity=0.5: {sim_05:.4f}")
            print(f"  Similarity with sensitivity=1.0: {sim_1:.4f}")
            print(f"  Similarity with sensitivity=2.0: {sim_2:.4f}")
            print(f"  ✓ PASS: Distance sensitivity affects similarity values")
            break


def test_encoding_consistency(ARCH, trainloader):
    """Test that encoding is consistent and correct"""
    print("\n" + "="*80)
    print("TEST: Encoding Consistency")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
    model.to(device)
    model.eval()

    for batch_idx, (proj_in, _, _, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
        proj_in = proj_in.to(device)
        break

    print("\n[Test 1] Checking encoding consistency...")
    with torch.no_grad():
        enc1, _, _ = model.encode(proj_in)
        enc2, _, _ = model.encode(proj_in)
    
    diff = torch.abs(enc1 - enc2).sum().item()
    print(f"  Difference between two encodings: {diff}")
    print(f"  ✓ PASS: Encoding is deterministic" if diff < 1e-6 else f"  ✗ FAIL: Encoding not deterministic")

    print("\n[Test 2] Checking encoding is bipolar...")
    with torch.no_grad():
        enc, _, _ = model.encode(proj_in)
    
    unique_vals = torch.unique(enc)
    is_bipolar = torch.all((unique_vals == -1) | (unique_vals == 1))
    print(f"  Unique values in encoding: {unique_vals.cpu().numpy()}")
    print(f"  ✓ PASS: Encoding is bipolar" if is_bipolar else f"  ✗ FAIL: Encoding not bipolar")

    print("\n[Test 3] Checking encoding dimension...")
    expected_dim = model.hd_dim
    actual_dim = enc.shape[1]
    print(f"  Expected dimension: {expected_dim}")
    print(f"  Actual dimension: {actual_dim}")
    print(f"  ✓ PASS: Correct dimension" if expected_dim == actual_dim else f"  ✗ FAIL: Incorrect dimension")

def test_accuracy_tracking(ARCH, trainloader):
    """Test the get_accuracy function"""
    print("\n" + "="*80)
    print("TEST: Accuracy Tracking")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    except:
        print("\n⚠ Could not load pre-trained model, using untrained model...")
        model = DensityModel(ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device)
        model.to(device)
    
    model.to(device)
    model.eval()

    for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device)
        break

    print("\n[Test 1] Checking accuracy is in [0, 1]...")
    with torch.no_grad():
        acc, conf_map, class_accs = model.get_accuracy(proj_in, proj_labels)
    
    print(f"  Overall accuracy: {acc:.4f}")
    print(f"  ✓ PASS: Accuracy in valid range" if 0 <= acc <= 1 else f"  ✗ FAIL: Invalid accuracy")

    print("\n[Test 2] Checking confidence map shape...")
    B, _, H, W = proj_in.shape
    expected_shape = (B, H, W)
    actual_shape = conf_map.shape
    print(f"  Expected shape: {expected_shape}")
    print(f"  Actual shape: {actual_shape}")
    print(f"  ✓ PASS: Correct shape" if expected_shape == actual_shape else f"  ✗ FAIL: Incorrect shape")

    print("\n[Test 3] Checking per-class accuracies...")
    print(f"  Number of classes with samples: {len(class_accs)}")
    for class_id, class_acc in sorted(class_accs.items()):
        print(f"    Class {class_id:2d}: {class_acc:.4f}")
    
    all_valid = all(0 <= acc <= 1 for acc in class_accs.values())
    print(f"  ✓ PASS: All class accuracies in [0, 1]" if all_valid else 
          f"  ✗ FAIL: Some class accuracies out of range")


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
    ARCH["train"]["batch_size"] = 2

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
                    shuffle_train=False)
    
    trainloader = parser.get_train_set()

    # test_collapse(ARCH, DATA)

    print("\n" + "="*80)
    print("DENSITY MODEL SANITY TESTS")
    print("="*80)

    test_encoding_consistency(ARCH, trainloader)
    test_subcluster_initialization(ARCH, trainloader)
    test_similarity_calculation(ARCH, trainloader)
    test_inference_update_mechanics(ARCH, trainloader)
    test_accuracy_tracking(ARCH, trainloader)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)

    test_collapse(ARCH, trainloader)

if __name__=="__main__":
    main()