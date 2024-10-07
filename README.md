# Wine Classification with Decision Tree

This project demonstrates the classification of the Wine dataset using the Decision Tree algorithm. The Wine dataset is loaded from scikit-learn's built-in datasets and used to train and evaluate a classification model.

## Project Overview

- **Dataset**: Wine dataset from `sklearn.datasets`
- **Algorithm**: Decision Tree Classifier
- **Objective**: Classify the type of wine based on chemical analysis.
- **Tools Used**: Python, Pandas, scikit-learn

## Steps:
1. Load the Wine dataset.
2. Preprocess the data and create features and target variables.
3. Split the data into training and testing sets.
4. Train the model using Decision Tree Classifier.
5. Evaluate the model using accuracy score, confusion matrix, and classification report.

## Results:
- **Accuracy**: 94.44%
- **Confusion Matrix**:
    - [[x, y], [z, w], [a, b]]
- **Classification Report**:  
    ```
    precision    recall  f1-score   support
    0       x.xx      x.xx      x.xx      xxx
    1       y.yy      y.yy      y.yy      yyy
    2       z.zz      z.zz      z.zz      zzz
    ```

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository.
    ```bash
    git clone https://github.com/your-username/wine_classification.git
    ```
2. Navigate to the project directory.
    ```bash
    cd wine_classification
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `wine_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal.
    ```bash
    python wine_classification.py
    ```

## License
This project is licensed under the MIT License.
