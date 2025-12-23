# Hiring process management application

### About the project

This is a production-ready web application designed to automate candidate screening. Using a cuML-powered Logistic Regression, the system predicts admission probability based on test performance. The project follows a Modular Architecture (Controller-Service-Schema) to ensure scalability, robustness, and ease of maintenance.

### Problem

* **Operational Bottleneck**: Manual screening of hundreds of candidates is slow and inefficient.

* **Subjective Bias**: Human-only reviews are prone to unconscious bias and inconsistent standards.

* **Scaling Issues**: Hard to maintain evaluation quality as the volume of applicants increases.

### Solution

* **GPU-Accelerated Inference**: High-performance processing using NVIDIA RAPIDS (cuML).

* **Objective Evaluation**: Data-driven decisions based on english_proficiency, cognitive_ability, and aptitude_test_result.

* **Data Integrity**: Strict input validation using Pydantic to ensure model reliability.

* **High Performance**: Achieved a ROC-AUC of 0.92, providing high discriminative power for admission stages.

### Tech stack
* **Language:** Python
* **API Framework:** Flask
* **Machine Learning:** cuML Logistic Regression
* **Data Processing:** Pandas, NumPy

---

### Data source and Pipeline

The model relies on a **synthetic dataset** available for download in the repository in the directory `data_analysis`.

### Model

```bash
{
    "test_metrics": {
        "accuracy": 0.8167,
        "confusion_matrix": [
            [
                26,
                6
            ],
            [
                5,
                23
            ]
        ],
        "classification_report": {
            "0": {
                "precision": 0.8387096774193549,
                "recall": 0.8125,
                "f1-score": 0.8253968253968254,
                "support": 32.0
            },
            "1": {
                "precision": 0.7931034482758621,
                "recall": 0.8214285714285714,
                "f1-score": 0.8070175438596491,
                "support": 28.0
            },
            "accuracy": 0.8166666666666667,
            "macro avg": {
                "precision": 0.8159065628476085,
                "recall": 0.8169642857142857,
                "f1-score": 0.8162071846282373,
                "support": 60.0
            },
            "weighted avg": {
                "precision": 0.8174267704857249,
                "recall": 0.8166666666666667,
                "f1-score": 0.816819827346143,
                "support": 60.0
            }
        },
        "roc_auc": 0.9286
    }
}
```

---

## Getting Started

Follow these instructions to set up a local copy of the project.

### Prerequisites

In this project I used `conda` for environment management.
* Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation

1.  **Clone the repository** (or download the files to a local folder).
    ```bash
    git clone [git@github.com:ViniciusRubens/hiring_process_management_application.git](https://github.com/ViniciusRubens/hiring_process_management_application)
    cd your-repository-name
    ```

2.  **Create a new conda environment** (this example uses `project_env` as the name):
    ```bash
    conda create --name project_env python=3.12
    ```

3.  **Activate the new environment:**
    ```bash
    conda activate project_env
    ```

4.  **Install pip** into the environment:
    ```bash
    conda install pip
    ```

5.  **Install the required dependencies** from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

With your `project_env` environment still active, run the application using the following command to start WSGI server from Gunicorn in terminal:

```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 run:app
```

Or you can run in development mode by Flask in terminal:

```bash
flask run
```

You will see an interface on web like this in localhost (`http://127.0.0.1:5000/`):

![](/images/image.png)

---

## Cleanup

To deactivate and remove the conda environment (optional).

1.  **Deactivate the environment:**
    ```bash
    conda deactivate
    ```

2.  **Remove the environment (optional):**
    ```bash
    conda remove --name project_env --all
    ```

---

## License

Distributed under the MIT License. See `LICENSE` file for more information.