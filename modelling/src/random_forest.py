import os
import pickle
import json
import numpy as np
import cudf 
import cuml
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

# The RandomForest algorithm, part of the ensemble family of methods, is one of the most popular 
# and versatile machine learning algorithms, used for both classification and regression tasks. Its 
# main feature is the construction of a set of decision trees during training, and the model output 
# is determined by the average of the predictions of all trees in the case of regression, or by 
# majority voting in the case of classification. The name “RandomForest” derives from the way the 
# algorithm creates a “forest” of random decision trees during training.

# Each tree in the random forest is constructed from a random sample of the training data 
# (with replacement), known as bootstrapping, which gives the model the property of being an 
# “ensemble bagging.” In addition, at each split in a tree, a random subset of features is selected 
# for consideration, increasing the diversity among trees and making the model more robust against 
# overfitting. This randomness helps improve model accuracy and reduce variance, providing an effective 
# solution for handling large and complex datasets. Individual trees tend to learn and adapt 
# excessively to the training data (overfitting), but when combined in a random forest, the errors 
# of one tree are offset by the successes of the others, resulting in improved overall performance.

# RandomForest is widely appreciated for its ease of use, as it requires little hyperparameter 
# configuration and generally produces good results with default settings. It is also less prone 
# to overfitting than a single decision tree. In addition, it offers good interpretability through 
# feature importance, which measures how much each feature contributes to the accuracy of the model's 
# prediction. However, despite its many advantages, it can be computationally intensive and less 
# efficient in terms of time and memory, especially with a very large number of trees or a very 
# large dataset. 


# --- Config ---
ARTIFACTS_DIR = '../../pre_processing/artifacts/'
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'X_train.parquet')
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'X_test.parquet')
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'y_train.parquet')
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'y_test.parquet')

SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'artifacts/scaler.pkl')

SAVE_PATH = os.path.join("../artifacts")

# --------------------

# Functions to handle user inputs
def get_int_input(prompt_text, default = None):
    val_str = input(prompt_text).strip()
    if not val_str: # If user press Enter (Null)
        return default
    
    try:
        return int(val_str)
    except ValueError:
        print(f"Invalid input (not an integer). Using default: {default}")
        return default

def get_model_params():
    print("--- Define parameters for RandomForestClassifier: ---")
    
    # 1. Model name
    model_name = ""
    while not model_name:
        model_name = input("Enter a unique name for this model run (e.g., 'rf_v1_depth5'): ").strip()
        if not model_name:
            print("Error: Model name is required to save artifacts.")

    # 2. cuML RandomForestClassifier parameters
    n_estimators = get_int_input("Parameter 'n_estimators' (int, num trees) [Default: 100]: ", default = 100)
    max_depth = get_int_input("Parameter 'max_depth' (int) [Default: 16]: ", default = 16)
    min_samples_leaf = get_int_input("Parameter 'min_samples_leaf' (int) [Default: 1]: ", default = 1)
    min_samples_split = get_int_input("Parameter 'min_samples_split' (int) [Default: 2]: ", default = 2)
    random_state = get_int_input("Parameter 'random_state' (int) [Default: 42]: ", default = 42)

    # 3. Model params dict
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "random_state": random_state
    }
    
    print("\n--- Configuration Selected ---")
    print(f"Model Name: {model_name}")
    print(f"Parameters: {model_params}\n")
    
    return model_name, model_params

def train_model():
    
    # --- 1. Load Data ---
    print("Loading data directly into GPU memory (VRAM)...")
    X_train = cudf.read_parquet(X_TRAIN_PATH)
    X_test = cudf.read_parquet(X_TEST_PATH)
    y_train = cudf.read_parquet(Y_TRAIN_PATH)
    y_test = cudf.read_parquet(Y_TEST_PATH)

    print("------------------------------------------\n")
    print(f"X Train shape (on GPU): {X_train.shape}")
    print(f"y Train shape (on GPU): {y_train.shape}")
    print("------------------------------------------\n")

    # --- 2. Model Configuration ---
    model_name, params = get_model_params()
    MODEL_PATH = os.path.join(SAVE_PATH, f"{model_name}.pkl")
    METRICS_PATH = os.path.join(SAVE_PATH, f"{model_name}_metrics.json")

    # --- 3. Model Training ---
    print("Initializing cuML RandomForestRegressor model (GPU)...")
    model = RandomForestClassifier(**params)

    print("Fitting model on GPU...")
    model.fit(X_train, y_train)

    print("------------------------------------------\n")
    print("Model fitting complete.")
    print(f"Params: {model.get_params()}")
    print("------------------------------------------\n")

    # --- 4. Prediction & Evaluation ---
    print("Predicting on test set (GPU)...")
    y_pred = model.predict(X_test)

    # Returns a matrix where column 0 is the probability of being ‘0’ and column 1 is the probability of being ‘1’.
    y_proba = model.predict_proba(X_test)
    y_proba = y_proba.iloc[:, 1]

    print("Calculating final metrics on test set...")
    y_test_cpu = y_test['status'].to_numpy()
    y_pred_cpu = y_pred.to_numpy()
    y_proba_cpu = y_proba.to_numpy()

    class_report = classification_report(y_test_cpu, y_pred_cpu)
    class_report_dict = classification_report(y_test_cpu, y_pred_cpu, output_dict=True)

    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)

    cm_numpy = confusion_matrix(y_test_cpu, y_pred_cpu)
    cm_list = cm_numpy.tolist()

    roc_auc = roc_auc_score(y_test_cpu, y_proba_cpu)

    metrics = {
        "test_metrics": {
            "accuracy": round(float(accuracy), 4),
            "confusion_matrix": cm_list,
            "classification_report": class_report_dict,
            "roc_auc": round(float(roc_auc), 4)
        }
    }

    # --- 5. Print Metrics ---
    print(f"\n--- Model Metrics: {model_name} (GPU Accelerated) ---")
    print(f"Accuracy: {metrics['test_metrics']['accuracy']}")
    print(f"Confusion Matrix:\n{np.array(cm_list)}")
    print("\n--- Classification Report ---")
    print(class_report)
    print("\n--- ROC AUC Score ---")
    print(roc_auc)
    print("------------------------------------------\n")

    # --- 6. Cross-Validation --- 
    print("\nCross-Validation")
    X_train_cpu = X_train.to_numpy()
    y_train_cpu = y_train['status'].to_numpy()
    cv_scores = cross_val_score(model, X_train_cpu, y_train_cpu, cv = 5)
    print(f"Cross_validation scores: {cv_scores}")

    # These results provide a more robust view of the model's performance, as cross-validation 
    # assesses the model's ability to generalize to new data. The variation in accuracy scores 
    # across different folds indicates that the model may behave inconsistently across different 
    # subsets of the data. This may be due to data characteristics, such as class imbalance, or the 
    # need for finer tuning of the model's hyperparameters.

    print(f"Saving model artifact to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saving metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Training script complete.")

if __name__ == "__main__":
    train_model()