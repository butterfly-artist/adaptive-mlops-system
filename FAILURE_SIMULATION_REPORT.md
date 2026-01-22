# 🛡️ Failure Simulation Case Study: Safe MLOps

This report documents how the Car Sales Prediction MLOps Platform handles extreme failures and edge cases. A production system is defined not by how it works, but by how it **fails**.

## 1. Scenario: Empty Production Data
*   **Condition**: The `logs/predictions.csv` file is missing or empty.
*   **System Action**: The pipeline detected the missing file at Step 1.
*   **Result**: **SAFE SKIP**.
*   **Why it didn't break**: The orchestrator includes a guard clause that verifies data availability before initiating any analysis. It logged a warning and exited gracefully without wasting compute resources.

## 2. Scenario: Data Validation Failure (Bad Retraining Data)
*   **Condition**: Production logs contained "garbage" data (e.g., `Price_in_thousands` = 99,999).
*   **System Action**: The retraining script initiated but failed during the **Mandatory Validation Step**.
*   **Result**: **RETRAINING ABORTED**.
*   **Why it didn't break**: The training script is "validation-aware." It refused to fit a model on data that violates business constraints. The pipeline caught the failure, logged the error, and **kept the current production model intact**.

## 3. Scenario: Performance Degradation (Worse Candidate Model)
*   **Condition**: A new model was trained on noisy data, resulting in poor performance (High RMSE).
*   **System Action**: The model trained successfully, but the **Deployment Gate** compared it against the current production model.
*   **Result**: **PROMOTION REJECTED**.
*   **Why it didn't break**: The "Better-or-Nothing" policy prevented the inferior model from being deployed. The system prioritized stability over novelty, ensuring users never see a degraded model.

## 4. Scenario: Missing Features in Logs
*   **Condition**: Production logs were missing required columns due to an upstream schema change.
*   **System Action**: The Drift Detector identified that the intersection of log columns and training columns was insufficient.
*   **Result**: **SAFE HALT**.
*   **Why it didn't break**: The system uses schema-matching logic. If it can't find the features it needs to measure drift, it stops the pipeline and logs a descriptive error, preventing "silent failures" where drift is reported as zero just because columns are missing.

---

## 📊 Summary of Robustness
| Failure Type | Primary Defense | Outcome |
| :--- | :--- | :--- |
| **Silent Degradation** | Deployment Gate (RMSE Comparison) | Rejected |
| **Data Corruption** | Pydantic & Range Validation | Aborted |
| **System Thrashing** | 24-hour Cooldown Period | Skipped |
| **Empty Logs** | File Existence Guards | Skipped |

**Conclusion**: The platform is built with a "Defense in Depth" strategy. Every step of the pipeline—from data merging to model promotion—is gated by validation and performance checks, making it virtually impossible for a "bad" model to reach production. 🛡️✨
