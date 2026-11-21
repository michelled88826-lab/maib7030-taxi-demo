"""Offline acceptance check for the MAIB7030 demo model pipeline."""
from pathlib import Path
import pandas as pd
import sklearn
from forecast_engine import load_demo_data, train_demo_models, get_case_row, predict_row

ROOT = Path(__file__).resolve().parent
print("scikit-learn version:", sklearn.__version__)
D = load_demo_data(ROOT / 'data' / 'taxi_demand_features.csv')
B = train_demo_models(D)
case = get_case_row(D, 142, pd.Timestamp('2025-05-01 22:00:00'))
pred = predict_row(case, B)
actual = int(case['pickup_count'])

print('DATA: Jan-May rows loaded =', len(D))
print('TRAIN:', B['training_rows'], '| MAY:', B['validation_rows'])
print('MAY METRICS:')
for name, values in B['may_metrics'].items():
    print(name, {k: round(v, 4) for k, v in values.items()})
print('DEMO CASE: Zone 142 | 2025-05-01 22:00')
print('Predictions:', {k: round(v, 2) for k, v in pred.items()})
print('Actual (for post-forecast reveal only):', actual)

assert B['training_rows'] == 27110
assert B['validation_rows'] == 7440
assert abs(B['may_metrics']['HGB baseline']['MAE'] - 22.78284) < 0.01
assert abs(B['may_metrics']['HGB + transition']['MAE'] - 22.65899) < 0.01
assert abs(pred['HGB baseline'] - 472.1117) < 0.1
assert abs(pred['HGB + transition'] - 478.3411) < 0.1
assert actual == 742
print('ACCEPTANCE CHECK: PASS')
