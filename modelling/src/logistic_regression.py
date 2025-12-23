import os
import pickle
import json
import numpy as np
import cudf 
import cuml
from cuml.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

# Logistic regression is a machine learning algorithm used to predict the probability of a 
# categorical dependent variable. Although the name suggests regression, it is most commonly 
# used in binary classification problems, where the goal is to predict one of two possible classes 
# (such as “yes” or “no,” “1” or “0,” ‘positive’ or “negative”). The algorithm models the probability 
# that a given input belongs to a specific category.

# The functioning of Logistic Regression is based on the concept of the logistic function, 
# also known as the sigmoid function. This function has an “S”-shaped curve, which can map any 
# real number to a value between 0 and 1, making it suitable for estimating probabilities. The model 
# calculates the probability that the input belongs to a specific class, and if that probability is 
# greater than a defined threshold (usually 0.5), the input is classified in that class. The logistic 
# function transforms the linear output (a linear combination of independent variables weighted by 
# coefficients, plus an intercept term) into a probability.


# In terms of training, Logistic Regression involves adjusting the coefficients of the independent 
# variables. This is done by maximizing the likelihood function, which seeks to find the coefficients 
# that make the predicted probabilities closest to the observed (actual) results. Essentially, the 
# algorithm attempts to draw a decision line that best separates the classes. To evaluate the model's 
# performance, metrics such as accuracy, confusion matrix, AUC-ROC (Area Under the Curve - Receiver 
# Operating Characteristic), and others are used. 


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
    
def get_float_input(prompt_text, default = None):
    val_str = input(prompt_text).strip()
    if not val_str:
        return default
    
    try:
        return float(val_str)
    except ValueError:
        print(f"Invalid input (not a float). Using default: {default}")
        return default

def get_model_params():
    print("--- Define parameters for LogisticRegression: ---")
    
    # 1. Model name
    model_name = ""
    while not model_name:
        model_name = input("Enter a unique name for this model run (e.g., 'lgr_v1_depth5'): ").strip()
        if not model_name:
            print("Error: Model name is required to save artifacts.")

    # 2. cuML LogisticRegression parameters
    C = get_float_input("Parameter 'C' (float) [Default: 1.0]: ", default = 1.0)
    penalty = input("Parameter 'penalty' (str: 'l1', 'l2', 'elasticnet', 'none') [Default: 'l2']: ").strip() or 'l2'
    max_iter = get_int_input("Parameter 'max_iter' (int) [Default: 1000]: ", default = 1000)

    # 3. Model params dict
    model_params = {
        "C": C,
        "penalty": penalty,
        "max_iter": max_iter
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
    print("Initializing cuML Logistic Regression model (GPU)...")
    model = LogisticRegression(**params)

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