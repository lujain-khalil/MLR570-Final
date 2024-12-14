# Lecture 2: Data Preprocessing

Data preprocessing involves transforming raw data into a clean and usable format for machine learning models. Key techniques include:

1. **Aggregation** reduces data size and reveals trends.
2. **Sampling** handles large datasets and class imbalances.
3. **Normalization** ensures consistent data scaling.
4. **Encoding** makes categorical data usable for models.
5. **Discretization** simplifies continuous features.

## **1. Aggregation**

- **Definition**: Summarizing or combining data points to reduce size and complexity.

- **Common Techniques**:
  - **Sum Aggregation**: Calculates the sum of values within a group.
  - **Mean Aggregation**: Computes the average value.
  - **Median Aggregation**: Finds the median value, robust to outliers.
  - **Windowed Aggregation**: Aggregates over a specific time window.
  - **Spatial Aggregation**: Combines geographically close data points (e.g., population density).

| Technique              | Advantages                        | Disadvantages                  | When to Use                        |
|-------------------------|------------------------------------|--------------------------------|------------------------------------|
| **Mean Aggregation**    | Simple, works for normal data.    | Sensitive to outliers.         | Symmetric data without outliers.  |
| **Median Aggregation**  | Robust to outliers.              | Ignores fine-grained details.  | Data with outliers or skewness.   |
| **Windowed Aggregation**| Tracks trends in time-series.    | Can introduce lag in real-time applications. | Time-series trend detection. |


## **2. Sampling**

- **Definition**: Selecting a subset of data to reduce size while maintaining representation.
- **Common Techniques**:
  - **Random Sampling**: Selects instances randomly.
  - **Stratified Sampling**: Proportional representation of groups/classes.
  - **Oversampling/Undersampling**: Balances imbalanced datasets by replicating or reducing samples.
  - **Systematic Sampling**: Selects data at regular intervals.

| Technique              | Advantages                        | Disadvantages                  | When to Use                        |
|-------------------------|------------------------------------|--------------------------------|------------------------------------|
| **Random Sampling**     | Simple, effective.               | May miss class representation. | Large homogeneous datasets.       |
| **Stratified Sampling** | Ensures proportional representation.| Requires class labels.         | Imbalanced datasets.              |
| **Oversampling**        | Balances minority classes.       | May cause overfitting.         | Imbalanced classification.        |


## **3. Normalization**

- **Definition**: Scaling numerical data to fit within a specific range.
- **Techniques**:
  - **Min-Max Normalization**: Scales data to [0, 1].

$$X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

  - **Z-Score Normalization**: Centers data with mean 0 and variance 1.

$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$

| Technique              | Advantages                        | Disadvantages                  | When to Use                        |
|-------------------------|------------------------------------|--------------------------------|------------------------------------|
| **Min-Max Normalization**| Ensures data in [0, 1] range.    | Sensitive to outliers.         | Data without outliers.            |
| **Z-Score Normalization**| Handles positive and negative values. | Less intuitive scaling.        | Normally distributed data.         |


## **4. Encoding**

- **Definition**: Converting categorical data into numerical format.
- **Techniques**:
  - **Label Encoding**: Assigns integer values to categories.
  - **One-Hot Encoding**: Creates binary columns for each category. 
  - **Frequency Encoding**: Replaces categories with their frequency.
  - **Binary Encoding**: Converts categories to binary digits.

> **_Example:_**  For a feature with 8 categories, one-hot encoding would create 8 new columns. Binary encoding would only introduce 3 new columns ($8 = 2^3$).

| Technique              | Advantages                        | Disadvantages                  | When to Use                        |
|-------------------------|------------------------------------|--------------------------------|------------------------------------|
| **Label Encoding**      | Simple, memory-efficient.         | Implies ordinal relationships. | Ordinal data (with order).         |
| **One-Hot Encoding**    | Avoids ordinal relationships.     | High-dimensional for many categories. | Nominal data (without order).      |
| **Binary Encoding**     | Reduces dimensionality.           | Less interpretable.            | High-cardinality data.             |


## **5. Discretization**

- **Definition**: Converts continuous features into discrete bins.
- **Techniques**:
  - **Equal-Width Binning**: Divides range into equal-width bins.
  - **Equal-Frequency Binning**: Ensures bins have equal number of data points.
  - **K-Means Binning**: Uses clustering to form bins.
