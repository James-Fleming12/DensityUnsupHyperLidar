import json
import os

SAVE_DIR = "logs/diagnostics"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

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
        total_bins = [0]*10
        correct_bins = [0]*10
        for rz in range(3):
            for b in range(10):
                total_bins[b] += data["T7_calibration"][str(rz)]["total_bins"][b]
                correct_bins[b] += data["T7_calibration"][str(rz)]["correct_bins"][b]
        
        print(f"\n{condition} Calibration (Confidence -> Accuracy):")
        for b in range(10):
            if total_bins[b] > 0:
                acc = correct_bins[b] / total_bins[b]
                print(f"  Conf {(b)/10.0:.1f}-{(b+1)/10.0:.1f}: Acc = {acc:.2%} (N={total_bins[b]})")
            else:
                print(f"  Conf {(b)/10.0:.1f}-{(b+1)/10.0:.1f}: Acc = N/A (N=0)")
                
    print("\n=== TEST 2: CONFIDENT-BUT-WRONG ARTIFACTS ===")
    for condition, data in [("SNOW", snow_data), ("FOG", fog_data)]:
        t2 = data["T2_confusion_hists"]
        # High confidence incorrect predictions are in bins 15-19 (0.75 - 1.0 confidence)
        high_conf_wrong = 0
        total_wrong = 0
        for pair, hist in t2.items():
            total_wrong += sum(hist)
            high_conf_wrong += sum(hist[15:]) # Conf >= 0.75
            
        pct_high_conf = high_conf_wrong / max(1, total_wrong)
        print(f"{condition}: {high_conf_wrong} out of {total_wrong} false positives ({pct_high_conf:.2%}) were highly confident (>0.75).")

if __name__ == "__main__":
    analyze()
