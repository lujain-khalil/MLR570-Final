# Lecture 5 Model Evaluation

## Training, Validation, and Testing

- **Training Data**: Used to optimize model parameters.
- **Validation Data**: Separate dataset to tune hyperparameters and monitor performance. Evaluates generalization beyond training data. It's benefits are:
    - Prevents overfitting by monitoring generalization error.
    - Enables **early stopping** during training to avoid unnecessary epochs.
    - Facilitates **model selection and tuning** based on validation error.
- **Test Data**: Final evaluation on unseen data for generalization.


## Overfitting and Underfitting

| Scenario         | Description                                  | Error Characteristics             |
|------------------|----------------------------------------------|------------------------------------|
| **Underfitting** | Model too simple to capture patterns.        | High training and testing error.  |
| **Overfitting**  | Model too complex, captures noise.           | Low training error, high test error. |


## K-Fold Cross-Validation
- Splits data into $K$ subsets (folds).
- Model is trained $K$ times, using $K-1$ folds for training and 1 fold for validation.

- **Benefits**:
  - Reliable performance estimates.
  - Detects overfitting risks.
  - Facilitates model selection.


## Classification Metrics
1. **Accuracy**: Ratio of correct predictions. Could be misleading for imbalanced datasets. 

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

2. **Precision**: Ratio of true positive out of all positive predictions. Ignores false negatives.

$$ \text{Precision} = \frac{TP}{TP + FP} $$

3. **Recall**: Ratio of true positive out of all actual positives. May result in many false positives.

$$ \text{Recall} = \frac{TP}{TP + FN} $$

4. **F1 Score**: Harmonic mean of precision and recall:

$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

**Why Harmonic Mean?** Balances disproportionate precision and recall values.

**Weighted F1 Score**: If we want to put more weight to precision in F1-score:

$$ F1_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}} $$

## Regression Metrics
1. **Mean Squared Error (MSE)**: Heavily penalizes large errors.

$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$

2. **Mean Absolute Error (MAE)**: Less sensitive to outliers.

$$ MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
