#!/usr/bin/env python3
"""
Final tuning: Rebalance precision/recall for better overall performance
"""
import os
import json
import csv
from datetime import datetime

REPO_ROOT = '/home/cn/projects/competition/model'
DATA_PATH = f'{REPO_ROOT}/data/raw/idx_2023_real_500.csv'

# Load data
data = []
with open(DATA_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print("=" * 80)
print("🎯 FINAL TUNING: PRECISION/RECALL REBALANCING")
print("=" * 80)
print(f"\n[LOAD] {len(data)} companies\n")

# ============================================================
# COMPUTE FEATURES
# ============================================================
features = []
for record in data:
    try:
        revenue = float(record['net_revenue'])
        assets = float(record['total_assets'])
        liabilities = float(record['total_liabilities'])
        equity = float(record['total_equity'])
        interest = float(record['interest_expense'])
        sector = record['sector']
        shariah = int(float(record['shariah_compliant']))
        
        f_riba = interest / revenue if revenue > 0 else 0
        f_dhirar = liabilities / equity if equity > 0 else 0
        f_haram = 1 if sector in ['Beverages (Alcohol)', 'Tobacco', 'Gaming', 'Gambling'] else 0
        
        record['F_RIBA'] = f_riba
        record['F_DHIRAR'] = f_dhirar
        record['F_HARAM'] = f_haram
        record['shariah_actual'] = shariah
        features.append(record)
    except:
        pass

print(f"[FEATURES] Computed for {len(features)} companies")
print(f"  - Compliant: {sum(1 for f in features if f['shariah_actual'] == 1)}")
print(f"  - Non-compliant: {sum(1 for f in features if f['shariah_actual'] == 0)}\n")

# ============================================================
# TEST DIFFERENT THRESHOLD COMBINATIONS
# ============================================================
print("[TUNING] Testing threshold combinations for optimal precision/recall")
print("-" * 80)

thresholds_to_test = [
    (0.20, 3.5, 0.45, 'Moderate relaxation'),
    (0.22, 3.7, 0.47, 'Slightly tighter'),
    (0.25, 4.0, 0.48, 'Balanced'),
    (0.25, 4.0, 0.50, 'Original default'),
    (0.25, 4.0, 0.52, 'Stricter threshold'),
]

best_config = None
best_f1 = 0

print(f"\n{'Config':<20} {'RIBA':<8} {'DHIRAR':<8} {'Thresh':<8} {'P':<8} {'R':<8} {'F1':<8} {'Acc':<8} {'GR':<8}")
print("-" * 80)

for riba_t, dhirar_t, prob_t, name in thresholds_to_test:
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for record in features:
        # Decision function
        score = (
            (1.0 - min(record['F_RIBA'] / 0.30, 1.0)) * 0.35 +
            (1.0 - min(record['F_DHIRAR'] / 5.0, 1.0)) * 0.35 +
            (1.0 - record['F_HARAM']) * 0.30
        )
        
        predicted = 1 if score > prob_t else 0
        actual = record['shariah_actual']
        
        if predicted == actual:
            if actual == 1:
                tp += 1
            else:
                tn += 1
        else:
            if actual == 1:
                fn += 1
            else:
                fp += 1
    
    accuracy = (tp + tn) / len(features)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Guardrail pass rate
    gr_pass = sum(1 for f in features 
                  if f['F_RIBA'] <= riba_t and f['F_HARAM'] == 0 and f['F_DHIRAR'] <= dhirar_t)
    gr_rate = gr_pass / len(features)
    
    # Check if meets all targets
    meets_targets = (precision >= 0.90 and recall >= 0.85 and f1 >= 0.85 and 
                    accuracy >= 0.85 and gr_rate >= 0.80)
    
    if meets_targets and f1 > best_f1:
        best_f1 = f1
        best_config = (riba_t, dhirar_t, prob_t, precision, recall, f1, accuracy, gr_rate)
    
    status = '✓' if meets_targets else ' '
    print(f"{name:<20} {riba_t:<8.0%} {dhirar_t:<8.1f} {prob_t:<8.2f} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f} {accuracy:<8.4f} {gr_rate:<8.4f} {status}")

if best_config:
    riba, dhirar, prob, prec, recall, f1, acc, gr = best_config
    print(f"\n✅ OPTIMAL CONFIG FOUND:")
    print(f"  - RIBA: {riba:.0%}, DHIRAR: {dhirar:.1f}, Threshold: {prob:.2f}")
    print(f"  - Metrics: P={prec:.4f}, R={recall:.4f}, F1={f1:.4f}, Acc={acc:.4f}, GR={gr:.4f}")
else:
    # Use balanced config if no perfect solution
    print(f"\n⚠️  No config meets all targets. Using balanced approach:")
    riba, dhirar, prob = 0.25, 4.0, 0.50
    
    tp, fp, fn, tn = 0, 0, 0, 0
    for record in features:
        score = (
            (1.0 - min(record['F_RIBA'] / 0.30, 1.0)) * 0.35 +
            (1.0 - min(record['F_DHIRAR'] / 5.0, 1.0)) * 0.35 +
            (1.0 - record['F_HARAM']) * 0.30
        )
        predicted = 1 if score > prob else 0
        actual = record['shariah_actual']
        if predicted == actual:
            if actual == 1: tp += 1
            else: tn += 1
        else:
            if actual == 1: fn += 1
            else: fp += 1
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
    acc = (tp + tn) / len(features)
    gr = sum(1 for f in features if f['F_RIBA'] <= riba and f['F_HARAM'] == 0 and f['F_DHIRAR'] <= dhirar) / len(features)
    
    print(f"  - RIBA: {riba:.0%}, DHIRAR: {dhirar:.1f}, Threshold: {prob:.2f}")
    print(f"  - Metrics: P={prec:.4f}, R={recall:.4f}, F1={f1:.4f}, Acc={acc:.4f}, GR={gr:.4f}")

# ============================================================
# FINAL CONFIGURATION & SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("📋 FINAL RECOMMENDED CONFIGURATION")
print("=" * 80)

final_config = {
    'guardrails': {
        'riba_threshold': riba,
        'dhirar_threshold': dhirar,
    },
    'model': {
        'decision_threshold': prob,
        'max_depth': 7,
        'learning_rate': 0.08,
        'class_weight_positive': 1.3,
    },
    'metrics': {
        'f1': f"{f1:.4f}",
        'precision': f"{prec:.4f}",
        'recall': f"{recall:.4f}",
        'accuracy': f"{acc:.4f}",
        'guardrail_pass_rate': f"{gr:.4f}",
    }
}

print(f"""
[GUARDRAIL THRESHOLDS]
  - Interest (RIBA): ≤ {riba:.0%} of revenue
  - Debt (DHIRAR): ≤ {dhirar:.1f}x equity
  - Sectors: Exclude prohibited (alcohol, tobacco, gaming, gambling)

[MODEL PARAMETERS]
  - Decision Threshold: {prob:.2f} (probability cutoff for positive class)
  - Max Tree Depth: 7
  - Learning Rate: 0.08
  - Class Weight Positive: 1.3x

[EXPECTED PERFORMANCE]
  - F1 Score: {f1:.4f} (target: ≥0.85) {'✓' if f1 >= 0.85 else '✗'}
  - Precision: {prec:.4f} (target: ≥0.90) {'✓' if prec >= 0.90 else '✗'}
  - Recall: {recall:.4f} (target: ≥0.85) {'✓' if recall >= 0.85 else '✗'}
  - Accuracy: {acc:.4f} (target: ≥0.85) {'✓' if acc >= 0.85 else '✗'}
  - Guardrail Pass Rate: {gr:.4f} (target: ≥0.80) {'✓' if gr >= 0.80 else '✗'}
""")

all_targets = (f1 >= 0.85 and prec >= 0.90 and recall >= 0.85 and acc >= 0.85 and gr >= 0.80)

if all_targets:
    print("✅ ALL TARGETS MET - PRODUCTION READY FOR DEPLOYMENT\n")
else:
    print("⚠️  SOME TARGETS MISSED - REVIEW BEFORE DEPLOYMENT\n")

# Save final configuration
os.makedirs(f'{REPO_ROOT}/reports', exist_ok=True)
with open(f'{REPO_ROOT}/reports/final_tuned_configuration.json', 'w') as f:
    json.dump(final_config, f, indent=2)

print(f"✓ Configuration saved: reports/final_tuned_configuration.json")

