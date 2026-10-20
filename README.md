# Employee Attrition Prediction with an Artificial Neural Network (Classification)

A binary classification project that predicts whether an employee will leave the company, using a feed-forward Artificial Neural Network (ANN) built with Keras/TensorFlow and tuned with Optuna.

## Project Overview

| Item | Details |
|---|---|
| **Task** | Binary classification |
| **Target** | `LeaveOrNot` (1 = employee left, 0 = stayed) |
| **Model** | ANN (2 hidden layers, sigmoid output) |
| **Tuning** | Optuna (hyperparameter search) |
| **Metric** | ROC-AUC and Gini (Gini = 2 × AUC − 1) |
| **Notebook** | `ANN_Classification_task_G30.ipynb` |

## Dataset

**File:** `Employee.csv`, with 4,653 rows, 9 columns and no missing values.

| Column | Type | Description |
|---|---|---|
| `Education` | Categorical | Education level (Bachelors, Masters, PHD) |
| `JoiningYear` | Numeric | Year the employee joined (2012–2018) |
| `City` | Categorical | Office city (Bangalore, Pune, New Delhi) |
| `PaymentTier` | Numeric | Pay tier of the employee (1, 2 or 3) |
| `Age` | Numeric | Age of the employee (22–41) |
| `Gender` | Categorical | Male / Female |
| `EverBenched` | Categorical | Whether the employee has ever been kept out of projects (Yes / No) |
| `ExperienceInCurrentDomain` | Numeric | Years of experience in the current domain (0–7) |
| `LeaveOrNot` | Target | 1 if the employee left, 0 otherwise |

**Class balance:** about 65.6% stayed (0) and 34.4% left (1), so the classes are moderately imbalanced.

## Workflow

1. **Read libraries and data** — pandas, NumPy, TensorFlow/Keras, Optuna, scikit-learn.
2. **Descriptive statistics** — `data.describe(include='all')`.
3. **Define target and inputs, then split** — `train_test_split` (80% train / 20% test, `random_state=42`).
4. **Encode categorical variables** — `LabelEncoder` fitted on the training set and applied to the test set.
5. **Scale the inputs** — `StandardScaler` fitted on the training set only, to avoid data leakage.
6. **Build the ANN and tune it with Optuna** — 2 hidden layers (ReLU) and a sigmoid output with `binary_crossentropy` loss and the AUC metric. Optuna ran 10 trials, maximizing test ROC-AUC, over:
   - units in each hidden layer (6–32)
   - optimizer (Adam, SGD, RMSprop, Adagrad)
   - learning rate (1e-5 to 1e-2, log scale)
   - number of epochs (10–50)
   - batch size (16–64)
7. **Train and evaluate the final model** with the best hyperparameters and report Gini on the train and test sets.

## Results

**Final model (retrained with the best Optuna hyperparameters):**

| Dataset | Gini | ROC-AUC |
|---|---|---|
| Train | 77.82 | 0.889 |
| Test | 77.35 | 0.887 |

The train and test scores are almost identical (a gap of about 0.5 Gini points), so the model generalizes well and does not overfit.

**Best hyperparameters found by Optuna (10 trials, best trial ROC-AUC 0.892):**

| Hyperparameter | Value |
|---|---|
| Units in hidden layer 1 | 29 |
| Units in hidden layer 2 | 13 |
| Optimizer | Adam |
| Learning rate | 0.00137 |
| Epochs | 48 |
| Batch size | 63 |

**What the search showed:** Adam and RMSprop trials reached ROC-AUC of about 0.84 to 0.89. The SGD trials stayed near 0.50 (no better than random guessing), because with very small learning rates SGD barely learned in 40 or so epochs.

## Technologies

- Python 3
- pandas, NumPy
- scikit-learn
- TensorFlow / Keras
- Optuna
- Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy scikit-learn tensorflow optuna jupyter
jupyter notebook ANN_Classification_task_G30.ipynb
```

Keep `Employee.csv` in the same folder as the notebook.

## Repository Structure

```
├── ANN_Classification_task_G30.ipynb
├── Employee.csv
└── README.md
```

## Key Takeaways

- Categorical variables are label-encoded because neural networks need numeric input.
- Inputs are standardized so that features on larger scales do not dominate training.
- Encoders and scalers are fitted on the training data only, to prevent data leakage.
- Optuna replaces manual trial and error when choosing the network size, optimizer and learning rate.
