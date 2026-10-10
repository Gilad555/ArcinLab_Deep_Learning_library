# ArcinLab Deep Learning Library

A deep learning library built from scratch on PyTorch tensors, without autograd or `torch.nn`. Every gradient is derived and implemented by hand.

## Why build it from scratch?

I believe that to become truly strong in AI, you have to build a deep learning library from scratch - more than once, and in more than one way.

This is not my first one. I have written at least three libraries before it, and this version is the product of that experience, knowledge, and personal touch.

Writing a library like this is how I get to feel the field in my hands.

## What's inside

- **Model** - `Sequence`: stacks layers, runs forward and backward propagation through them, and collects all trainable parameters
- **Layers** - fully connected layer with manual forward and backward propagation and shape validation
- **Activations** - Linear, ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softplus, SiLU, GELU, Softmax
- **Losses** - MSE, and Cross-Entropy fused with Softmax for a simple and numerically stable gradient
- **Optimizers** - SGD, Momentum, Adam (with bias correction)
- **Initialization** - He, Xavier, zeros, random normal
- **Parameters** - a wrapper that holds each tensor together with its accumulated gradient
- **Data** - `Dataset` with sample/label validation, and `BatchesMaker` for shuffled mini-batches
- **Hardware** - runs on CUDA, Apple Silicon (MPS), or CPU

## Quick start

```python
import torch
from ArcinLab_Deep_Learning_library import *

# data must live on the same device as the model's weights
device = initialization.get_device()
X = torch.randn(256, 4, device=device)
Y = torch.nn.functional.one_hot(torch.randint(0, 3, (256,), device=device), 3).float()

# model
model = Sequence()
model.add_layer(Layer(4, 16, activation=ReLU()))
model.add_layer(Layer(16, 3))          # logits - softmax is fused into the loss

loss_fn = CrossEntropy()
optimizer = Adam(model, learning_rate=1e-2)

# training loop
for epoch in range(20):
    for xb, yb in BatchesMaker(X, Y, batch_size=32):
        loss = loss_fn.compute(model.forward_propagation(xb), yb)   # yb is one-hot

        optimizer.reset_gradients()
        model.backward_propagation(loss_fn.backward())
        optimizer.apply()
```

## The math

Every gradient in this library is derived by hand. One highlight:

**Softmax + Cross-Entropy.** The Softmax Jacobian and the Cross-Entropy derivative are messy on their own, but together they collapse to a single subtraction - which is why the library implements them as one fused, numerically stable loss:

$$\frac{\partial L}{\partial z_j} = p_j - y_j$$

Full step-by-step derivations: [MATH.md](MATH.md)
