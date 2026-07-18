# Handwritten Character Recognition using Machine Learning

## Objective
To build a machine learning model that recognizes handwritten digits (0–9) from 28x28 pixel images using the MNIST dataset.

---

## Folder Structure
```text
TASK_2/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── notebook.ipynb
├── model.py
├── requirements.txt
├── README.md
├── trained_model.pkl
└── images/
    ├── sample_digits.png
    ├── loss_accuracy_graph.png
    └── confusion_matrix.png
```

---

## Project Phases & Workflow

1. **Phase 1: Data Collection**: Loaded 10,000 training images and 2,000 testing images from the MNIST dataset.
2. **Phase 2: Data Preprocessing**: Normalized pixel intensities from [0, 255] to [0.0, 1.0].
3. **Phase 3: Data Visualization**: Visualized sample digit images and verified label distributions.
4. **Phase 4: Model Building**: Built a Multi-Layer Perceptron (MLP) Neural Network with architecture `(128, 64)` hidden units.
5. **Phase 5: Model Training**: Trained using Adam optimizer with ReLU activation functions.
6. **Phase 6 & 7: Model Testing & Performance Evaluation**:
   - **Test Accuracy**: **95.20%**
   - **Precision**: **95.14%**
   - **Recall**: **95.14%**
   - **F1-Score**: **95.12%**
7. **Phase 8 & 9: Prediction & Results**: Generated visual plots for Loss Curve, Confusion Matrix, and Sample Predictions.
8. **Phase 10: Conclusion**: Achieved high accuracy (>95%) suitable for digit classification tasks.
9. **Phase 11: Future Scope**: Expand to recognize handwritten letters (EMNIST), complete words, and deploy as a web app.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Model
```bash
python model.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
