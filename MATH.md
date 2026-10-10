# The math behind ArcinLab

Full derivations of the gradients implemented by hand in this library.

## 1. Softmax + Cross-Entropy: why the gradient collapses to $p - y$

### Step 1 - The Jacobian of Softmax

First, Softmax is a function that takes a vector and returns a different vector, where every component of the output is affected by every component of the input:

$$p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$$

So in this case, the derivative of each input component is also a vector: each of its components is the influence of that input component on one of the output components - including the output in its own position.

The derivative of the whole input vector is then a matrix, built from the derivative vectors of each input component, according to its influence on each of the outputs. Since we are working with row vectors, in this matrix - which is called the Jacobian - each row is the influence of one input component on each of the output components.

Now, in Softmax there are two kinds of influence an input component has on an output component.

The first is when the input component is itself in the numerator - meaning the output is in the same position as the input. In this case the numerator is not a constant. This happens only once in its influence on all the output components.

The second is when it is only a part of the denominator - the sum. There, the numerator is a constant.

**The case where the numerator is not a constant.** By the quotient rule: the derivative of the numerator, which is $e$ to the power of the input component, times the whole denominator, which is the sum - minus the numerator times the derivative of the denominator with respect to the input component, which is simply $e$ to the power of the input component. And in the denominator, the sum squared:

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i}\sum_k e^{z_k} - e^{z_i}\,e^{z_i}}{\left(\sum_k e^{z_k}\right)^2}$$

I split this expression into two fractions with a subtraction between them.

In the first fraction, the numerator is $e$ to the power of the input component times the sum, and the denominator is the sum squared. We cancel the sum, and now we have an expression identical to the output component itself - so we substitute it with $p$:

$$\frac{e^{z_i}\sum_k e^{z_k}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} = p_i$$

The second fraction, the one being subtracted, is $e$ to the power of the input component times itself, over the sum squared. This is actually a fraction times a fraction, and each fraction is exactly the output component - the probability - so we substitute it times itself:

$$\frac{e^{z_i}\,e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} \cdot \frac{e^{z_i}}{\sum_k e^{z_k}} = p_i \cdot p_i$$

So we have $p$ minus itself times itself. We take $p$ out as a common factor, and we get $p$ times, in parentheses, one minus itself:

$$\frac{\partial p_i}{\partial z_i} = p_i - p_i \cdot p_i = p_i(1 - p_i)$$

This is the general case where an input component stands against its own output component. It happens only on the diagonal of the Jacobian.

**The case where the numerator is a constant.** Zero times anything doesn't matter, because it's zero - minus $e$ to the power of the input component times $e$ to the power of the input in the output component's position, over the sum squared:

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot \sum_k e^{z_k} - e^{z_i}\,e^{z_j}}{\left(\sum_k e^{z_k}\right)^2}$$

This is also a product of products. We can write it as minus the probability in the input component's position, times the $p$ we are differentiating, which belongs to the output component. We take $p$ of the output out as a common factor, and it comes out as $p$ times, in parentheses, zero minus the probability in the input component's position:

$$\frac{\partial p_i}{\partial z_j} = -p_i\,p_j = p_i(0 - p_j)$$

**Similar to the first case, isn't it?** What is the only thing that's different? The zero or the one in the middle.

So we can set a general formula for all the derivatives of an input component against an output component. It goes like this: the output component, times, in parentheses, $t$ of the input component's index against the output component's index - which says that if the components are the same it is one, and otherwise it is zero - minus the probability in the input component's position:

$$\frac{\partial p_i}{\partial z_j} = p_i\left(t_{ij} - p_j\right), \qquad t_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

Now we have a matrix.

### Step 2 - Cross-Entropy

Cross-Entropy is a function that takes a vector and returns a single number. Something happens to each input component, and then we add it to the same thing that happens to every other input component. Each input component is handled on its own, so here input and output components won't collide.

The formula for the general term is: put it inside $\ln$, and multiply by minus the matching component of the label vector:

$$L = -\sum_k y_k \ln p_k$$

This means that if we differentiate with respect to a specific component of the probability vector, we get one over the number inside the $\ln$, because that is its derivative, times minus the label component in the matching position. So the derivative of the loss with respect to each probability component is minus the label component in its position, over that probability component itself:

$$\frac{\partial L}{\partial p_k} = -\frac{y_k}{p_k}$$

That is the derivative of the loss with respect to one probability.

### Step 3 - Putting them together

Now, if we want the derivative of the loss with respect to an input component before the probabilities - a $z$ - we need to multiply each component in its derivative vector in the Jacobian by the derivative of the loss with respect to that specific probability. And we sum all of them, because $z$ contributed to the loss through several paths.

We write it as a sigma that runs over the number of classes. If there are $K$ classes, it runs $K$ times. What it sums is minus the label component $k$ over the probability $k$, times the probability $k$, times, in parentheses, our $t$ of $k$ against the index of the specific $z$, minus the probability in the position of that $z$:

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^{K} \frac{\partial L}{\partial p_k}\,\frac{\partial p_k}{\partial z_j} = \sum_{k=1}^{K} -\frac{y_k}{p_k}\;p_k\left(t_{kj} - p_j\right)$$

In every round, the $p_k$ in our denominator is identical to the $p_k$ that is now in the numerator after the multiplication. We cancel it, and we are left with minus sigma of label component $k$ times, in parentheses, our $t$ minus the probability in the position of the $z$ we are differentiating with respect to:

$$\frac{\partial L}{\partial z_j} = -\sum_{k=1}^{K} y_k\left(t_{kj} - p_j\right)$$

Now we open the parentheses, and we can write it as two separate sigmas:

$$\frac{\partial L}{\partial z_j} = -\left(\sum_{k=1}^{K} y_k\,t_{kj} \;-\; \sum_{k=1}^{K} y_k\,p_j\right)$$

Let's look at the first sigma. Because $t$ equals one only once - exactly when $k$ equals our index - there we simply get the label at the index:

$$\sum_{k=1}^{K} y_k\,t_{kj} = y_j$$

In the second sigma, we multiply the same number - the probability - each time by a different number. All those numbers together complete to one, so we simply get the number itself, which is the probability:

$$\sum_{k=1}^{K} y_k\,p_j = p_j \sum_{k=1}^{K} y_k = p_j$$

And here we have a result that is just a simple subtraction. The minus sign flips it, and we get the probability minus the label, at the index of the $z$ we are differentiating with respect to:

$$\frac{\partial L}{\partial z_j} = -\left(y_j - p_j\right) = p_j - y_j$$
