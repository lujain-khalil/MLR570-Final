# Lecture 9: Feedforward Neural Networks (FNNs)

Neural networks learn a mapping function $f: X \to Y$ where:
- **Classification**: $Y$ is discrete (e.g., $Y \in \{1, 2, \dots, k\}$).
- **Regression**: $Y$ is continuous (e.g., $Y \in \mathbb{R}$).

| Component            | Description                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| **Layers (Depth)**    | Number of layers: input, hidden, and output.                             |
| **Width**            | Number of neurons per layer.                                              |
| **Activation**       | Introduces non-linearity for complex modeling.                           |
| **Loss Function**     | Measures prediction error: MSE (regression) or CE (classification). |


## Forward Propagation

### Classification Example:
For input $X = [X_1, X_2, X_3]$, activations $a$ and outputs $o$, any hidden layer result is as follows:

$$a_1 = f(W_1 X + b_1)$$

where:
- $W_1 \in \mathbb{R}^{1 \times 3}$
- $b_1 \in \mathbb{R}$
- $f(z)$: Activation (e.g., ReLU, Sigmoid, Tanh) ($z = W_1 X + b_1$)

Using softmax for multi-class classification as an example, the output layer predictions are calculated as:
   
$$o_i = f(z) = \frac{e^z}{\sum_j e^z}$$


### Regression Example:
Similar to classification, for any input $X = [X_1, X_2, X_3] $, activations $a$ and outputs $o$, any hidden layer result is as follows:
   
$$a_1 = f(W_1 X + b_1)$$

Using ReLI for regression as an example, the output layer predictions are calculated as:

$$o_i = f(z) = \text{max}(0, z)$$

### Loss Functions

| **Loss Function**          | **Formula**                                                                 | **Use Case**                          | **Advantages**                           | **Disadvantages**                         |
|----------------------------|----------------------------------------------------------------------------|---------------------------------------|------------------------------------------|-------------------------------------------|
| **Mean Squared Error (MSE)** | $\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$                     | Regression                           | Simple, smooth gradients                 | Sensitive to outliers                     |
| **Mean Absolute Error (MAE)**| $\frac{1}{n} \sum_{i=1}^n \|y_i - \hat{y}_i\|$                       | Regression                           | Robust to outliers                       | Non-smooth gradient at $y_i = \hat{y}_i$ |
| **Cross-Entropy Loss**     | $- \sum_{i=1}^C y_i \log(\hat{y}_i)$                                | Classification                       | Well-suited for probabilistic outputs    | Requires softmax or sigmoid output       |
| **Binary Cross-Entropy**   | $- \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right)$      | Binary classification               | Handles 2-class problems effectively     | Sensitive to class imbalance             |

### Activation Functions

| **Function**          | **Formula**                      | **Advantages**                                | **Disadvantages**                          |
|------------------------|----------------------------------|----------------------------------------------|--------------------------------------------|
| **Sigmoid**            | $f(x) = \frac{1}{1 + e^{-x}}$ | Smooth gradient, output between $ [0, 1] $  | Vanishing gradient for large/small inputs. |
| **Tanh**               | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Zero-centered, faster convergence            | Still suffers from vanishing gradients.    |
| **ReLU**               | $f(x) = \max(0, x)$         | Solves vanishing gradient for positive values| Dead neurons, potential gradient explosion.|

## Backward Propagation

Taking SGD as our baseline, the objective when backpropagating through a neural network is to minimize the loss function $L(y, \hat{y})$ w.r.t the weights $W$:

$$\min_W L(y, \hat{y})$$

The update rule for weights and biases is as follows:

$$W := W - \alpha \frac{\partial}{\partial W} L(y, \hat{y})$$

$$b := b - \alpha \frac{\partial}{\partial b} L(y, \hat{y})$$

We know that our predtiction $\hat{y} = f(z)$, where $z = WX + b$. We can rewrite the loss function as $L(y, f(z))$, or more specifically, $L(y, f(WX + b))$. Regardless, deriving the loss function w.r.t $W$ and $b$, we use the chain rule and get the following:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial f(z)} \cdot \frac{\partial f(z)}{\partial z} \cdot \frac{\partial z}{\partial W}$$

### Optimizers

| **Optimizer**          | **Update Rule**                                                                                     | **Key Features**                       | **Advantages**                              | **Disadvantages**                       |
|-------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------|--------------------------------------------|-----------------------------------------|
| **SGD** | $\theta_t = \theta_{t-1} - \eta \nabla_\theta L(\theta_{t-1})$                       | Basic gradient descent optimization    | Simple, easy to implement                  | Slow convergence, noisy updates         |
| **SGD with Momentum**  | $\theta_t = \theta_{t-1} - \eta v_t$ | Adds "momentum" term to smooth updates | Reduces oscillations, accelerates learning | Requires tuning momentum $\beta$    |
| **Adam (Adaptive Moment Estimation)** | $\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ | Combines momentum and adaptive learning rates | Fast convergence, adaptive learning rates | Computationally expensive, more hyperparameters |


## Weight Initialization

- **Random Initialization**:
  - Drawn from uniform or normal distribution.
  - Can cause gradient issues (slow convergence).
- **He Initialization**: Suitable for **ReLU**.
    
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

- **Xavier Initialization**: Suitable for **Sigmoid/Tanh**.

$$W \sim \text{Uniform}\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right)$$


## Deeper vs Wider Networks
- Deeper networks generally have more **expressive power** than wider networks.
- Wide networks may fail to approximate complex functions without sufficient depth.

## Applications (MNIST Example)
- **MNIST Dataset**: Classifying handwritten digits.
- Importance of data shuffling for accuracy:
  - **0% shuffling**: Poor performance.
  - **100% shuffling**: Optimal test accuracy.

