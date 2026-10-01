# CSDL-SEN-XAI: Software Defect Prediction Framework

Implementation aligned with the research paper:
> **"An Interpretable Analytical Intelligence Architecture Delivering Reliable Detection of Software Defect Instances"**  
> *Srinivasa Rao Katragadda & Dr. Sirisha Potluri*  
> *(IJACSA) International Journal of Advanced Computer Science and Applications, Vol. 16, No. 12, 2025*

---

## 🌟 Key Architecture & Novel Components

```
+-----------------------------------------------------------------------------------------+
|                                    CSDL-SEN-XAI PIPELINE                                 |
+-----------------------------------------------------------------------------------------+
|  [NASA CM1 Dataset] -> [Missing Imputation & Normalization] -> [Train/Test Split 80:20] |
|                                                                          |              |
|                                                                   [SMOTE (Train Only)]  |
|                                                                          |              |
|                   [Enzyme Action Optimizer (EAO)] <----------------------+              |
|                   - Binding (Crossover)                                                 |
|                   - Cleavage (Mutation)                                                 |
|                   - Catalysis (Move towards Elite E)                                    |
|                   - Joint Feature Selection & SNN Hyperparameter Tuning                  |
|                                          |                                              |
|                 [Controlled Balanced Pair Construction (1:1)]                           |
|                 - Positive Pairs (k-Nearest Neighbors)                                  |
|                 - Negative Pairs (Random Cross-Class)                                   |
|                                          |                                              |
|            [Contrastive Siamese Defect Learning (CSDL) Network]                         |
|            - Shared Weight Twin Subnetworks                                             |
|            - Contrastive Loss Optimization                                              |
|                                          |                                              |
|                     [Distance-Based Defect Classifier]                                  |
|                     - Prototype Centroid Clustering (C0, C1)                             |
|                                          |                                              |
|                       [Explainability Layer (XAI)]                                      |
|                       - SHAP Global Feature Attribution (Eq. 10)                        |
|                       - Integrated Gradients Local Reasoning (Eq. 11)                   |
+-----------------------------------------------------------------------------------------+
```

### 1. Data Preprocessing & Class Balancing (`preprocess.py`)
- **Imputation**: Continuous features imputed via median/mean; categorical via mode.
- **Min-Max Normalization (Eq. 1)**: Rescales all software metrics to $[0, 1]$.
- **Pearson Collinearity Pruning**: Identifies and drops redundant features with Pearson correlation $> 0.98$.
- **SMOTE Balancing (Eq. 2, Eq. 3)**: Applied **only to the training set** to eliminate majority class bias without data leakage.

### 2. Enzyme Action Optimizer (EAO) (`eao_optimizer.py`)
- Simulates biochemical enzymatic reactions (*Binding, Cleavage, and Catalysis*) for joint feature selection and hyperparameter optimization.
- **Fitness Objective (Eq. 4)**:
  $$F = \alpha \cdot (1 - \text{Accuracy}) + \beta \cdot \frac{|S|}{|T|}$$
- **Catalytic Transition (Eq. 5)**:
  $$X_{t+1} = X_t + \eta \cdot (E - X_t) + \gamma \cdot R$$

### 3. Contrastive Siamese Defect Learning (CSDL) (`model.py` & `pair_generator.py`)
- Constructs 1:1 balanced positive ($k$-NN) and negative pairs.
- Maps input metrics into discriminative embedding spaces using twin subnetworks.
- **Contrastive Loss**:
  $$\mathcal{L} = \frac{1}{2}(1-y) D^2 + \frac{1}{2} y \max(0, m - D)^2$$
- **Decision Layer (Eq. 9)**: Classifies test instances based on Euclidean distance to prototype cluster centroids.

### 4. Dual Explainability Layer (`explainability.py`)
- **Global SHAP (Eq. 10)**: Identifies influential metrics (LOC, Cyclomatic Complexity, Coupling) across the entire software project.
- **Instance-Level Integrated Gradients (Eq. 11)**: Path integral attributions explaining why an individual module is flagged as defective.

---

## 📋 Experimental Simulation Parameters (Table II)

| Parameter | Value in Paper |
| :--- | :--- |
| **Dataset Benchmark** | NASA MDP `cm1.csv` (498 modules, 22 metrics) |
| **Train-Test Split** | 80% - 20% (Stratified) |
| **SMOTE $k$-neighbors** | 5 |
| **Epochs ($E$)** | 30 |
| **Batch Size ($B$)** | 32 |
| **Learning Rate ($\alpha$)** | 0.001 |
| **Embedding Dimension ($d$)** | 64 |
| **Margin ($m$)** | 1.0 |
| **Optimizer** | Adam |
| **Activation Function** | ReLU |
| **Cross-Validation ($K$)** | 5-Fold (Optimization) / 10-Fold (Final) |

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Full Pipeline & Run Benchmarks
```bash
python train.py
```
This runs EAO optimization, trains CSDL, performs 10-fold cross validation (Table III), benchmarks baseline models (Table IV), and saves artifacts in `models/`.

### 3. Test Predictions & Explanations via CLI
```bash
python predict.py
```

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view live inference, Integrated Gradients attributions, EAO convergence curves, PCA embedding projections, and benchmark tables.
