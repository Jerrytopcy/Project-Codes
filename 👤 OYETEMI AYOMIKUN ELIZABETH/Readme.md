Here’s a comprehensive and user-friendly README.md file for my project. It includes instructions on setup, running the code, and understanding the outputs—designed for both technical and non-technical users who might want to explore the project.


# 🧬 Breast Cancer Classification using Decision Tree and Genetic Algorithm Optimization

This project implements a decision tree classifier to predict breast cancer (Benign or Malignant) using the **Wisconsin Breast Cancer Dataset**. It includes a baseline model and an optimized version using a **Genetic Algorithm (GA)**. The project is built and tested in **Python** with Jupyter Notebook.

## 📂 Project Structure

📁 breast-cancer-classification/ │ ├── breast_cancer_decision_tree.ipynb ← Main Jupyter Notebook ├── plots/ │ ├── baseline_confusion_matrix.png │ ├── evolved_confusion_matrix.png │ ├── performance_metrics_comparison.png │ ├── hyperparameter_bar_plot.png │ ├── genetic_algorithm_learning_curve.png │ └── decision_tree_structure.png ├── README.md ← You are here └── requirements.txt ← List of dependencies



---

## 🧰 Requirements

Before running the project, ensure the following Python packages are installed:

```bash
pip install -r requirements.txt
requirements.txt includes:
pandas

numpy

matplotlib

seaborn

scikit-learn

deap

ucimlrepo

jupyterlab / notebook (if running via Jupyter)

## 🚀 Getting Started
Option 1: Run in Jupyter Notebook
Install Jupyter Notebook (if not already installed):


pip install notebook
Launch the notebook:


jupyter notebook
Navigate to the breast_cancer_decision_tree.ipynb file and run each cell sequentially.

Option 2: Run in VS Code or another IDE
Make sure you have Python and the required libraries installed.

Open the .ipynb file in your IDE.

Run the cells interactively.

## 📊 Features & Outputs
Data Loading & Preprocessing: Cleans missing values, encodes targets.

Baseline Decision Tree: Trained using default hyperparameters.

Genetic Algorithm Optimization: Tunes max_depth, min_samples_split, min_samples_leaf, and criterion.

Evaluation Metrics: Includes accuracy, precision, recall, F1-score.

Visualizations:

Confusion matrices (baseline vs evolved)

Metric comparisons

Hyperparameter bar chart

Genetic Algorithm learning curve

Full decision tree visualization

All saved plots can be found in the /plots directory.

## 📈 Dataset Info
The dataset is fetched directly from the UCI Machine Learning Repository:

Dataset ID: 15 (Breast Cancer Wisconsin – Original)

Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)

## 🧠 Model Summary
Baseline Accuracy: 93.57%

Optimized Accuracy: 94.29%

Best Parameters (via GA):

max_depth: 5

min_samples_split: 6

min_samples_leaf: 7

criterion: entropy

## 📸 Visuals and Reports
Each critical stage of the model's development and evaluation is visualized. These can be included in a report, thesis, or presentation. Visual outputs are stored in the plots/ folder and named accordingly.

## 💡 Notes
If you encounter an error about re-declaring classes from DEAP (e.g., FitnessMax or Individual), restart the kernel to clear previously defined classes.

Be sure to run the notebook from start to end without skipping cells for consistent results.

## 📬 Questions or Issues?
For any bug reports or clarification, feel free to open an issue or contact the project maintainer.

📁 Author: OYETEMI AYOMIKUN ELIZABETH
📅 Last Updated: April 2025

