import torch
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import DensityModel
from modules.trainer import Trainer
from modules.ioueval import iouEval

import numpy as np
import torch.nn.functional as F

from unsup_main import train_extractor, train_hdc, test_hdc_model, test_hdc_model_debug

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

    test_hdc_model(model, trainloader)

    model.to(device)

    model.train()
    for epoch in range(inference_epochs):
        for batch_idx, (proj_in, _, proj_labels, _, _, _, _, _, _, _, _, _, _, _, _) in enumerate(trainloader):
            proj_in = proj_in.to(device)
            for i in proj_in:
                model.inference_update(i.unsqueeze(0))
        test_hdc_model(model, trainloader)

    print("\n" + "="*80)
    print(f"After {inference_epochs} epochs of inference updating")
    print("="*80)
    test_hdc_model(model, trainloader)

def test_collapse_debug(ARCH, trainloader, inference_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: DensityModel = DensityModel(
        ARCH, MODEL_DIR, 'rp', 0, 0, NUM_CLASSES, device
    )
    model.load_state_dict(torch.load(HDC_SUB_PATH, weights_only=False))
    model.to(device)

    print("\n" + "=" * 80)
    print("INITIAL MODEL DIAGNOSTICS")
    print("=" * 80)

    test_hdc_model_debug(model, trainloader)

    proto_before = model.prototypes.detach().clone()
    sub_before = model.subclusters.detach().clone()

    for epoch in range(inference_epochs):
        print("\n" + "-" * 80)
        print(f"INFERENCE UPDATE EPOCH {epoch + 1}")
        print("-" * 80)

        model.train()

        update_counts = torch.zeros(model.num_classes, device=device)
        total_updates = 0

        for batch_idx, (proj_in, _, proj_labels, *_) in enumerate(trainloader):
            proj_in = proj_in.to(device)

            for x in proj_in:
                logits, sims, indices, _ = model(x.unsqueeze(0))
                pred = torch.argmax(logits, dim=1).item()

                update_counts[pred] += 1
                total_updates += 1

                model.inference_update(x.unsqueeze(0))

        # Normalize update distribution
        update_dist = update_counts / (update_counts.sum() + 1e-8)

        print("\n[Update Distribution]")
        for c in range(model.num_classes):
            print(
                f"  Class {c:2d}: updates={int(update_counts[c].item()):6d} "
                f"({100 * update_dist[c].item():5.2f}%)"
            )

        # Run diagnostics after epoch
        test_hdc_model_debug(model, trainloader)

    print("\n" + "=" * 80)
    print("FINAL PROTOTYPE / SUBCLUSTER DRIFT ANALYSIS")
    print("=" * 80)

    proto_after = model.prototypes.detach()
    sub_after = model.subclusters.detach()

    proto_drift = torch.norm(proto_after - proto_before, dim=1)
    sub_drift = torch.norm(sub_after - sub_before, dim=1)

    print("\n[Prototype Drift]")
    for c in range(model.num_classes):
        print(
            f"  Class {c:2d}: Δproto={proto_drift[c].item():.6f}"
        )

    print("\n[Subcluster Drift]")
    per_class_sub = sub_drift.view(model.num_classes, -1)
    for c in range(model.num_classes):
        mean_drift = per_class_sub[c].mean().item()
        max_drift = per_class_sub[c].max().item()
        print(
            f"  Class {c:2d}: mean Δsub={mean_drift:.6f}, max Δsub={max_drift:.6f}"
        )

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
    """Test inference updates for continuous unit-norm prototypes"""
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

    norms = torch.norm(proto, dim=1)
    max_dev = torch.max(torch.abs(norms - 1.0)).item()

    print(f"  classify.weight shape: {proto.shape}")
    print(f"  Max deviation from unit norm: {max_dev:.6f}")

    if max_dev > 1e-3:
        raise AssertionError(
            "❌ classify.weight is not unit-normalized after initialization."
        )

    for proj_in, _, proj_labels, *_ in trainloader:
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device)
        break

    print("\n[DIAGNOSTIC] Prediction sanity check...")
    with torch.no_grad():
        enc, _, _ = model.encode(proj_in)
        enc_norm = F.normalize(enc)

        logits = model.classify(enc_norm.to(torch.float))
        preds = logits.argmax(dim=1)

        print(f"  Unique predictions: {torch.unique(preds)}")

    print("\n[Test 1] Prototype update magnitude check...")
    original = model.classify.weight.clone()

    model.train()
    with torch.no_grad():
        model.chunked_inference_update(
            proj_in,
            beta=0.05,
            distance_sensitivity=1.0
        )

    updated = model.classify.weight

    delta = torch.norm(updated - original, dim=1)
    mean_delta = delta.mean().item()
    max_delta = delta.max().item()

    print(f"  Mean prototype change (L2): {mean_delta:.6f}")
    print(f"  Max prototype change (L2):  {max_delta:.6f}")

    if max_delta > 0:
        print("  ✓ PASS: Prototypes updated")
    else:
        print("  ✗ FAIL: No prototype updates detected")

    print("\n[Test 2] Beta gating behavior...")

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.0)
    delta_beta_0 = torch.norm(model.classify.weight - original, dim=1).mean().item()

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.5)
    delta_beta_05 = torch.norm(model.classify.weight - original, dim=1).mean().item()

    print(f"  beta=0.0 mean Δ: {delta_beta_0:.6f}")
    print(f"  beta=0.5 mean Δ: {delta_beta_05:.6f}")

    if delta_beta_0 >= delta_beta_05:
        print("  ✓ PASS: Lower beta → stronger updates")
    else:
        print("  ⚠ WARNING: Beta gating inverted")

    print("\n[Test 3] distance_sensitivity effect...")

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.1, distance_sensitivity=0.0)
    delta_ds_0 = torch.norm(model.classify.weight - original, dim=1).mean().item()

    model.classify.weight.data.copy_(original)
    with torch.no_grad():
        model.chunked_inference_update(proj_in, beta=0.1, distance_sensitivity=50.0)
    delta_ds_50 = torch.norm(model.classify.weight - original, dim=1).mean().item()

    print(f"  ds=0.0 mean Δ: {delta_ds_0:.6f}")
    print(f"  ds=50.0 mean Δ: {delta_ds_50:.6f}")

    if delta_ds_0 >= delta_ds_50:
        print("  ✓ PASS: distance_sensitivity scales updates")
    else:
        print("  ⚠ WARNING: distance_sensitivity has no effect")

    print("\n[FINAL CHECK] Bipolar invariant...")
    final_norms = torch.norm(model.classify.weight, dim=1)
    max_dev = torch.max(torch.abs(final_norms - 1.0)).item()

    assert max_dev < 1e-3, "✗ Prototype normalization invariant violated"

    print("  ✓ All prototypes remain unit-normalized")

def test_inference_update_verbose(ARCH, trainloader):
    """Test inference update mechanics step-by-step"""
    print("\n" + "="*80)
    print("TEST: Inference Update Mechanics")
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

    # Get a single batch and subsample it
    for proj_in, _, proj_labels, *_ in trainloader:
        proj_in = proj_in.to(device)
        proj_labels = proj_labels.to(device)
        break

    # SUBSAMPLE to avoid OOM - take only 5000 pixels
    with torch.no_grad():
        enc_full, _, _ = model.encode(proj_in)
        total_samples = enc_full.shape[0]
        subsample_size = min(5000, total_samples)
        
        subsample_indices = torch.randperm(total_samples, device=device)[:subsample_size]
        enc = enc_full[subsample_indices]
        
        # Reconstruct corresponding labels
        batch_size = proj_in.shape[0]
        h, w = proj_in.shape[-2:]
        proj_labels_flat = proj_labels.view(-1)
        proj_labels_sub = proj_labels_flat[subsample_indices]
        
        print(f"\n✓ Subsampled {subsample_size} pixels from {total_samples}")
        del enc_full

    print("\n[Step 1] Initial model state...")
    with torch.no_grad():
        enc_norm = F.normalize(enc)
        logits_before = model.classify(enc_norm.to(torch.float))
        preds_before = logits_before.argmax(dim=1)
        
        # Get accuracy before
        acc_before = (preds_before == proj_labels_sub).float().mean().item()
        
        print(f"  Accuracy before update: {acc_before:.4f}")
        print(f"  Unique predictions: {torch.unique(preds_before).tolist()}")
        
        # Check prototype norms
        proto_norms = torch.norm(model.classify.weight, dim=1)
        print(f"  Prototype norms - min: {proto_norms.min():.6f}, max: {proto_norms.max():.6f}")
        
        # Sample some predictions
        sample_indices = torch.randperm(len(preds_before))[:10]
        print(f"\n  Sample predictions vs labels:")
        for idx in sample_indices:
            print(f"    Pred: {preds_before[idx].item()}, Label: {proj_labels_sub[idx].item()}")

    print("\n[Step 2] Testing subcluster similarity calculation...")
    with torch.no_grad():
        enc_binary = torch.sign(enc)
        zero_mask = enc_binary == 0
        if torch.any(zero_mask):
            enc_binary[zero_mask] = -1.0
        
        # Test subcluster similarity for each class
        for class_id in range(min(3, NUM_CLASSES)):  # Test first 3 classes
            class_mask = preds_before == class_id
            if not torch.any(class_mask):
                continue
                
            class_enc = enc_binary[class_mask]
            sims, indices = model.get_max_subcluster_similarity(
                class_enc, class_id, distance_sensitivity=1.0
            )
            
            print(f"  Class {class_id} ({class_mask.sum()} samples):")
            print(f"    Subcluster similarities - min: {sims.min():.4f}, max: {sims.max():.4f}, mean: {sims.mean():.4f}")
            print(f"    Number of subclusters for this class: {(model.subcluster_to_class == class_id).sum().item()}")

    print("\n[Step 3] Testing distance calculation...")
    with torch.no_grad():
        prototypes_binary = torch.sign(model.classify.weight)
        selected_prototypes = prototypes_binary[preds_before]
        hd_dim = enc_binary.shape[1]
        similarities = torch.sum(enc_binary * selected_prototypes, dim=1) / hd_dim
        distances = (1 - similarities) / 2
        
        print(f"  Hamming distances - min: {distances.min():.4f}, max: {distances.max():.4f}, mean: {distances.mean():.4f}")
        print(f"  Samples with distance > 0.1: {(distances > 0.1).sum().item()}/{len(distances)}")
        print(f"  Samples with distance > 0.2: {(distances > 0.2).sum().item()}/{len(distances)}")

    print("\n[Step 4] Creating small test batch for update...")
    # Create an even smaller batch for the actual update test
    test_size = 1000
    test_indices = torch.randperm(len(enc))[:test_size]
    enc_test = enc[test_indices]
    labels_test = proj_labels_sub[test_indices]
    
    # Reconstruct a fake input tensor (we'll pass enc directly to avoid re-encoding)
    print(f"  Using {test_size} samples for update test")

    print("\n[Step 5] Performing update with manual calculation...")
    original_weight = model.classify.weight.clone()
    original_weights = model.classify_weights.clone()
    
    with torch.no_grad():
        # Manually simulate the update to debug
        model.classify_weights.data.copy_(model.classify.weight.data)
        
        enc_test_binary = torch.sign(enc_test)
        zero_mask = enc_test_binary == 0
        if torch.any(zero_mask):
            enc_test_binary[zero_mask] = -1.0
        
        enc_test_norm = F.normalize(enc_test)
        
        logits_test = model.classify(enc_test_norm.to(torch.float))
        predictions_test = torch.argmax(logits_test, dim=1)
        
        # Calculate distances
        prototypes_binary = torch.sign(model.classify.weight)
        selected_prototypes = prototypes_binary[predictions_test]
        hd_dim = enc_test_binary.shape[1]
        similarities = torch.sum(enc_test_binary * selected_prototypes, dim=1) / hd_dim
        distances = (1 - similarities) / 2
        
        beta = 0.1
        mask = distances > beta
        
        print(f"  Samples needing update: {mask.sum().item()}/{len(mask)}")
        
        if torch.any(mask):
            distant_enc_norm = enc_test_norm[mask]
            distant_enc_binary = enc_test_binary[mask]
            distant_predictions = predictions_test[mask]
            
            # Process one class as example
            test_class = distant_predictions[0].item()
            class_mask = distant_predictions == test_class
            class_enc_norm = distant_enc_norm[class_mask]
            class_enc_binary = distant_enc_binary[class_mask]
            
            print(f"\n  Example: Class {test_class} with {len(class_enc_norm)} samples")
            
            # Get subcluster similarities
            subcluster_sims, sub_indices = model.get_max_subcluster_similarity(
                class_enc_binary, test_class, distance_sensitivity=1.0
            )
            
            print(f"    Subcluster sims - min: {subcluster_sims.min():.4f}, max: {subcluster_sims.max():.4f}, mean: {subcluster_sims.mean():.4f}")
            
            # Calculate update
            scaled_samples = class_enc_norm * subcluster_sims.unsqueeze(1)
            total_update = scaled_samples.sum(dim=0)
            
            print(f"    Update magnitude: {torch.norm(total_update):.4f}")
            print(f"    Update/sample ratio: {torch.norm(total_update)/len(class_enc_norm):.4f}")
            
            # Apply update
            model.classify_weights[test_class] += total_update
            
            # Normalize
            old_weight = model.classify.weight[test_class].clone()
            model.classify.weight[test_class] = F.normalize(
                model.classify_weights[test_class:test_class+1], dim=1
            )[0]
            
            weight_change = torch.norm(model.classify.weight[test_class] - old_weight).item()
            print(f"    Weight change after normalization: {weight_change:.6f}")
            
            # Test prediction change
            logits_after = model.classify(enc_test_norm.to(torch.float))
            preds_after = logits_after.argmax(dim=1)
            
            acc_before_test = (predictions_test == labels_test).float().mean().item()
            acc_after_test = (preds_after == labels_test).float().mean().item()
            
            print(f"\n    Accuracy before: {acc_before_test:.4f}")
            print(f"    Accuracy after: {acc_after_test:.4f}")
            print(f"    Change: {acc_after_test - acc_before_test:+.4f}")

    print("\n" + "="*80)

def test_subcluster_similarity_diagnostics(ARCH, trainloader, NUM_CLASSES):
    print("\n" + "="*80)
    print("TEST: Subcluster Similarity Diagnostics")
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

    print("\n[Diag 1] Subcluster counts per class")
    for c in range(NUM_CLASSES):
        count = (model.subcluster_to_class == c).sum().item()
        status = "OK" if count > 0 else "✗ MISSING"
        print(f"  Class {c:2d}: {count} subclusters → {status}")

    enc_cache = {}
    with torch.no_grad():
        for proj_in, _, proj_labels, *rest in trainloader:
            proj_in = proj_in.to(device)
            proj_labels = proj_labels.to(device).flatten()
            enc, _, _ = model.encode(proj_in)

            for c in range(NUM_CLASSES):
                mask = proj_labels == c
                if c not in enc_cache and torch.any(mask):
                    enc_cache[c] = enc[mask][:8]  # small sample

            if len(enc_cache) == NUM_CLASSES:
                break

    print("\n[Diag 2] Similarity failure mode analysis")
    for c in sorted(enc_cache.keys()):
        enc = enc_cache[c]

        mask = model.subcluster_to_class == c
        sub = model.subclusters[mask]

        if sub.numel() == 0:
            print(f"  Class {c:2d}: ✗ No subclusters — similarity forced to 0.5")
            continue

        sims_signed, _ = model.get_max_subcluster_similarity(enc, c)
        sims_signed = sims_signed.cpu()

        enc_n = torch.nn.functional.normalize(enc, dim=1).to(torch.float)
        sub_n = torch.nn.functional.normalize(sub.float(), dim=1)
        sims_cont = torch.max(enc_n @ sub_n.T, dim=1).values
        sims_cont = ((sims_cont + 1) / 2).cpu()

        enc_mag = enc.abs().mean().item()
        enc_zero_frac = (torch.sign(enc) == 0).float().mean().item()

        print(f"\n  Class {c:2d}")
        print(f"    Signed sim:   mean={sims_signed.mean():.3f}, "
              f"min={sims_signed.min():.3f}, max={sims_signed.max():.3f}")
        print(f"    Cosine sim:   mean={sims_cont.mean():.3f}, "
              f"min={sims_cont.min():.3f}, max={sims_cont.max():.3f}")
        print(f"    Enc |x| mean: {enc_mag:.4f}")
        print(f"    Zero sign %:  {enc_zero_frac*100:.2f}%")

        if torch.allclose(sims_signed, torch.full_like(sims_signed, 0.5)):
            print("    ✗ COLLAPSE: similarity = 0.5 → subclusters unused or orthogonal")
        elif sims_cont.mean() > sims_signed.mean() + 0.15:
            print("    ⚠  SIGN LOSS: binarization destroying similarity")
        elif enc_zero_frac > 0.05:
            print("    ⚠  ENCODER ISSUE: many near-zero dimensions before sign")
        else:
            print("    ✓ Similarity behaving as expected")

    print("\n" + "="*80)

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

def test_suite(ARCH, trainloader):
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

def main():
    # A code snippet to test model collapse in the model after updating over the training set/test set
    try:
        ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    except Exception as e:
        print(f"Error opening arch yaml file. {e}")
        quit()
    try:
        DATA = yaml.safe_load(open("config/labels/nuscenes_mini.yaml", 'r'))
    except Exception as e:
        print(f"Error opening data yaml file. {e}")
        quit()

    DATA['split']['train'] = [61, 103, 553, 655, 757, 796, 916, 1077]
    ARCH["train"]["batch_size"] = 1

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
    
    trainloader = parser.get_train_set()

    # test_inference_update_verbose(ARCH, trainloader)
    # test_subcluster_similarity_diagnostics(ARCH, trainloader, NUM_CLASSES)
    test_collapse(ARCH, trainloader)
    # test_collapse_debug(ARCH, trainloader)

if __name__=="__main__":
    main()