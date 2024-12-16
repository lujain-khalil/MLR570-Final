# Lecture 10: Convolutional Neural Networks and Recurrent Neural Networks

## Convolutional Neural Networks

Input images are represented as pixel intensity values ranging from $[0, 255]$. For an RGB image, it is represented as a **matrix** of shape $H \times W \times 3$ (e.g., $1080 \times 1080 \times 3$). CNNs Exploit spatial structures by learning local patches and hierarchically build up features. A basic CNN would look like this:

![CNN architecture](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/CNN.png)

Some key advantages include:

1. **Parameter Sharing**: Reduces the number of parameters.
2. **Local Connectivity**: Learns spatial hierarchies.
3. **Translation Invariance**: Recognizes features anywhere in the input.

### Feature Extraction with Convolution

A **filter** (or kernel) is applied to local patches. Multiple filters extract different features. **Parameter sharing** is when filters are reused across the input. A visual of this can be seen in the [Lecture 10 Practice](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Practice/Lecture%2010.pdf) document.

For an $N \times N$ input, $F \times F$ kernel, $P$-padded rows/columns (if you add one 'border' of 0's, you're adding 1 row and 1 column to the image, hence $P = 1$), and stride = 1 (how much you're 'shifitng' the kernel across the input), the output size of a convolution is formulated as:

$$\text{Output size} = \frac{(N + 2P - F)}{\text{stride}} + 1$$  

**Example:** Input = $5 \times 5$, Filter = $3 \times 3$, Padding = $1$, Stride = $1$:

$$\text{Output Size} = \frac{(5 + 2(1) - 3)}{1} + 1 = 5 \implies 5 \times 5$$


### Pooling Layer

Pooling reduces spatial dimensions (downsampling) and provides spatial invariance. The two examples are:

| **Pooling Type**     | **Method**                       |
|-----------------------|----------------------------------|
| **Max Pooling**       | Retains the maximum value in a patch. |
| **Average Pooling**   | Computes the average of the values.   |

**Example**: For a $2 \times 2$ kernel with stride $2$, the output of a max pooling layer is:

$$
\begin{bmatrix}
1 & 1 & 4 & 2 \\
5 & 6 & 7 & 8 \\
3 & 2 & 0 & 1 \\
1 & 2 & 3 & 4
\end{bmatrix}
\quad \to \quad
\begin{bmatrix}
6 & 8 \\
3 & 4
\end{bmatrix}
$$

### Padding
- **Same Convolution**: Pads all inputs so that the features are the same size and the input throughout the network.
- **Valid-only Convolution**: Standard convolution concept but with strictly no padding in the network.

### CNN Architectures

| **Architecture** | **Year** | **Key Features**                                     | **Depth**      | **FC Layers**      | **Parameters**         |
|------------------|----------|------------------------------------------------------|----------------|------------------|------------------------|
| **LeNet-5**      | 1998     | - First successful CNN.<br>- Replaced manual feature extraction.<br>- Two conv and pooling layers. | 5 layers       | :heavy_check_mark:            | ~60,000                |
| **AlexNet**      | 2012     | - First use of ReLU.<br>- First breakthrough w/ ImageNet<br>- Dropout for regularization.<br>- Data augmentation. | 8 layers       | :heavy_check_mark:         | ~60 million            |
| **VGGNet**       | 2014     | - Smaller convolutions.<br>- Deeper network.<br> | 16–19 layers   | :heavy_check_mark:         | ~138 million           |
| **GoogleNet**    | 2014     | - Introduced **Inception module**.<br>- Reduced parameters.<br>- Much deeper than AlexNet<br>- Computationally efficient. | 22 layers      | :x:         | ~5 million             |
| **ResNet**       | 2015     | - Introduced **skip connections** (residual blocks).<br>- Much much deep networks.<br>- Solves vanishing/exploding gradient problem. | 50–152 layers  | :x:         | ~25 million (ResNet-50) |

### Inception module

- Introduced in **GoogleNet**
- Addresses the challenge of choosing the right filter size
- Combines multiple filter sizes and pooling operations in parallel within a single layer
- The inception module achieves the following:
   - **Multi-scale Feature Extraction**: Different filter sizes capture features at different scalesComputationally efficient. Pooling ensures spatial information is retained
   - **Parameter Efficiency**: The use of 1x1 convolutions reduces the number of channels before applying larger filters, making it _computationally efficient_
   - **Improved learning**: By combining outputs from different operations, the network learns richer features.

The inception module can be explained by the following example (using Same Convolutions for the sake of simplicity):

![Inception module](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/inception-module.png)

- **Input features** of size $28 \times 28 \times 196$ (i.e. $196$ channels)
- **Parallel convolutions** of size $28 \times 28 \times C_i$, where $C_i$ is the size of subset of channels used in branch $i$. A pooling layer would take all the channels ($28 \times 28 \times 196$)
- **Filter Concatentation**: $28 \times 28 \times \sum_{i=0}^B C_i$, where $B$ is the number of branches. This final layer just concatenates the outputs of all parallel convolution branches

### Skip Connection

- Introduced in **ResNet**
- Addresses the challenge of vanishing gradients
- Uses short-circuit connections that bypass one or more layers by adding the input of a layer directly to its output
- Skip connections achieves the following:
   - **Solves Vanishing Gradient Problem**: In very deep networks, gradients become very small (vanish) as they backpropagate through many layers. Skip connections provide an identity shortcut that allows gradients to flow directly through the network
   - **Learn Residuals**: Instead of learning the full mapping $H(x)$,the network learns a _residual function_ $F(x) = H(x) - x$
   - **Improved Training**: Enables the training of very deep architectures (e.g., ResNet-152) without degradation in performance (not caused by overfitting)

Here's a visual example that uses ReLU as the activation function $f(z) = \max(0, z)$:

![Skip Connections](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/skip-connection.png)


$$z_1 = W_1 \cdot a_0 + b_1$$

$$a_1 = f(z_1)$$

$$z_2 = W_2 \cdot a_1 + b_2$$


$$a_2 = f(z_2)$$

$$\implies a_2 = W_2 \cdot a_1 + b_2$$

If we introcude a skip connection:

$$a_2 = f(z_2 + a_0)$$

$$\implies a_2 = W_2 \cdot a_1 + b_2 + a_0$$

## Recurrent Neural Networks

RNNs model sequential data while retaining memory of past inputs. Hidden states allow information to flow across time steps. In FNNs, the input flows only forward. In RNNs, input also flows forward, but hidden states also depend on previous states.

| Type            | Input            | Output           | Example                  |
|-----------------|------------------|------------------|--------------------------|
| One-to-Many     | Fixed-size       | Sequence         | Image captioning         |
| Many-to-One     | Sequence         | Fixed-size       | Sentiment analysis       |
| Many-to-Many    | Sequence         | Sequence         | Language translation     |


### **Model Formulation**
At each time step $t$, the hidden state $h_t$ is computed as:

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

where:
- $W_h, W_x$: Weight matrices
- $h_{t-1}$: Hidden state from previous step
- $x_t$: Input at current step,
- $f(z)$: Activation function (e.g., tanh).


### **Vanishing/Exploding Gradient Problem**

> _Incomplete dervations_

Let's derive the gradient that will be used to update $W_h$. Given a loss function $L(y, \hat{y})$, gradients of the loss function $L$ are backpropagated through time:

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_h}$$

Since $h_t$ depends recursively on $h_{t-1}$, we get repeated matrix multiplications involving $W_h$. 

Using **$h_t = tanh(W_h h_{t-1} + W_x x_t + b)$:**

1. Gradient at step $t$:
   $$
   \frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t} \approx W_h^T \frac{\partial L}{\partial h_{t+1}}
   $$
2. Repeated chain rule gives:
   $$
   \frac{\partial L}{\partial h_0} = W_h^T W_h^T \dots W_h^T \frac{\partial L}{\partial h_T}
   $$
   - **If $\| W_h \| < 1$**: Gradients shrink exponentially → Vanishing gradient.
   - **If $\| W_h \| > 1$**: Gradients explode exponentially → Exploding gradient.

- **Vanishing Gradient**: Gradients shrink exponentially if $W_h$ has small eigenvalues (less than 1).
  $$
  \frac{\partial h_t}{\partial h_{t-1}} \approx W_h^T \quad \text{(repeated multiplications shrink values)}
  $$
- **Exploding Gradient**: Gradients explode if $W_h$ has large eigenvalues (greater than 1).


### **Gated Recurrent Units (GRUs): Solving Vanishing Gradient**
> _Incomplete dervations_

GRUs introduce **gates** to control the flow of information:
- **Update Gate** ($z_t$): Controls how much of the past information is carried forward.
- **Reset Gate** ($r_t$): Controls how much of the past hidden state to forget.

**GRU Equations**
1. Update gate:
   $$
   z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
   $$
2. Reset gate:
   $$
   r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
   $$
3. Candidate hidden state:
   $$
   \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
   $$
4. Final hidden state:
   $$
   h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
   $$

**Why GRUs Solve Vanishing Gradients**
- The **update gate** $z_t$ selectively carries forward long-term information.
- Gradients flow through $z_t$, preventing exponential shrinking.


In summary: 
- **RNNs** process sequential data using hidden states but face vanishing/exploding gradients due to recursive computations.
- **GRUs** solve these issues with update/reset gates, allowing selective memory retention and better gradient flow.
