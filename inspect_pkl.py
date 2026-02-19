
import joblib
from pathlib import Path
import json

model_path = Path("d:/projects/datamodeMLOPS/mlops-auto-retrain/production_models/v1.0/lasso_pipeline.pkl")
model = joblib.load(model_path)

preprocessor = model.named_steps['preprocessor']

report = {
    "pipeline_steps": [s[0] for s in model.steps],
    "transformers": []
}

for name, transformer, cols in preprocessor.transformers:
    if name != 'remainder':
        report["transformers"].append({
            "name": name,
            "columns": list(cols)
        })

print(json.dumps(report, indent=2))
