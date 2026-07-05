import json
import os

SAVE_DIR = "logs/diagnostics"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def summarize_group(g):
    if g["count"] == 0:
        return None
    mean = g["sum"] / g["count"]
    var = max(g["sumsq"] / g["count"] - mean ** 2, 0.0)
    std = var ** 0.5
    return mean, std, g["count"]

def analyze():
    snow_path = os.path.join(SAVE_DIR, "baseline_diagnostics_snow_frozen.json")
    fog_path = os.path.join(SAVE_DIR, "baseline_diagnostics_fog_frozen.json")

    if not os.path.exists(snow_path) or not os.path.exists(fog_path):
        print("Diagnostic files not found.")
        return

    snow_data = load_json(snow_path)
    fog_data = load_json(fog_path)

    print("=== TEST 7: CALIBRATION (FOG vs SNOW) ===")
    for condition, data in [("SNOW", snow_data), ("FOG", fog_data)]:
        total_bins = [0] * 10
        correct_bins = [0] * 10
        for rz in range(3):
            for b in range(10):
                total_bins[b] += data["T7_calibration"][str(rz)]["total_bins"][b]
                correct_bins[b] += data["T7_calibration"][str(rz)]["correct_bins"][b]

        print(f"\n{condition} Calibration (Confidence -> Accuracy):")
        for b in range(10):
            if total_bins[b] > 0:
                acc = correct_bins[b] / total_bins[b]
                print(f"  Conf {b/10.0:.1f}-{(b+1)/10.0:.1f}: Acc = {acc:.2%} (N={total_bins[b]})")
            else:
                print(f"  Conf {b/10.0:.1f}-{(b+1)/10.0:.1f}: Acc = N/A (N=0)")

    print("\n=== TEST 2: CONFIDENT-BUT-WRONG ARTIFACTS ===")
    for condition, data in [("SNOW", snow_data), ("FOG", fog_data)]:
        t2 = data["T2_confusion_hists"]
        high_conf_wrong = 0
        total_wrong = 0
        for pair, hist in t2.items():
            total_wrong += sum(hist)
            high_conf_wrong += sum(hist[15:])
        pct_high_conf = high_conf_wrong / max(1, total_wrong)
        print(f"{condition}: {high_conf_wrong} out of {total_wrong} false positives "
              f"({pct_high_conf:.2%}) were highly confident (>0.75).")

    print("\n=== TEST 2b: CROSS-VIEW VARIANCE (VGP PRE-CHECK) ===")
    for condition, data in [("SNOW", snow_data), ("FOG", fog_data)]:
        if "T8_variance_stats" not in data or "correct" not in data["T8_variance_stats"]:
            print(f"{condition}: No T8_variance_stats found -- rerun baseline_test.py "
                  f"with the T8 patch applied before this will show anything.")
            continue

        t8 = data["T8_variance_stats"]
        correct_stats = summarize_group(t8["correct"])
        wrong_stats = summarize_group(t8["wrong"])

        if correct_stats is None or wrong_stats is None:
            print(f"{condition}: insufficient high-confidence samples in one group to compare.")
            continue

        mean_c, std_c, n_c = correct_stats
        mean_w, std_w, n_w = wrong_stats
        separation = mean_w / mean_c if mean_c > 0 else float("inf")

        print(f"\n{condition}:")
        print(f"  High-conf CORRECT: mean_variance={mean_c:.5f} std={std_c:.5f} n={n_c}")
        print(f"  High-conf WRONG:   mean_variance={mean_w:.5f} std={std_w:.5f} n={n_w}")
        print(f"  Separation ratio (wrong/correct): {separation:.2f}x")
        if separation >= 2.0:
            print("  -> Real signal: VGP has something to gate on for this condition.")
        elif separation >= 1.3:
            print("  -> Weak/marginal signal: VGP may need a different perturbation "
                  "set (e.g. intensity/range jitter) to sharpen this.")
        else:
            print("  -> No meaningful separation: cross-view variance isn't "
                  "discriminating artifacts here with yaw+dropout perturbations.")

if __name__ == "__main__":
    analyze()
