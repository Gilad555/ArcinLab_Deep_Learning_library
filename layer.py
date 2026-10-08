from .parameter import *
from . import initialization
from .activation import *
import torch

class Layer:
    """
    An artificial neural network layer.

    Parameters
    ----------
    in_features : int
        Number of layer's input sample's features.
    out_features : int
        Number of layer's output sample's features.
    activation : Activation
        Layer's activation function with forward and backward.
    
    Attributes
    ----------
    W : Parameter
        Layer's weight
    b : Parameter
        Layer's bias
    """

    def __init__(self, in_features, out_features, activation=None):
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(initialization.he((in_features, out_features)))
        self.b = Parameter(initialization.zeros( out_features))

        self.activation = activation if activation is not None else Linear()
        
    def forward_propagation(self, X):
        """
        Layer's forward propagation.

        Parameters
        ----------
        X : torch.Tensor
            Layer's input.
        
        Returns
        -------
        A : torch.Tensor
            Layer's output.

        Raises
        ------
        ValueError
            If input doesn't fit layer.
        """

        if(X.shape[-1] != self.in_features):
            raise ValueError(f"shape: {X.shape} does not fit layer's shape: {(self.in_features, self.out_features)}")

        self._layer_input = X

        Z = X @ self.W.parameter + self.b.parameter
        self._Z = Z

        A = self.activation.forward(Z)
        self._A = A

        return A

    def backward_propagation(self, dA):
        """
        Layer's backward propagation.

        Parameters
        ----------
        dA : torch.Tensor
            Layer's output gradient.

        Returns
        -------
        dX : torch.Tensor
            Layer's input gradient.

        Raises
        ------
        ValueError 
            If gradient is not the same shape as layer's output.
        """

        if(self._A.shape != dA.shape):
            raise ValueError(f"layer's gradient shape: {dA.shape} does not identical as layer's output shape.")

        dZ = self.activation.backward(self._Z) * dA

        # adding gradient to weight and bias:

        dW = self._layer_input.transpose(-2, -1) @ dZ
        dims = tuple(range(dW.dim() - 2))
        if(len(dims) > 0):
            dW = torch.sum(dW, dim=dims)
        self.W.add_to_gradient(dW)

        dims = tuple(range(dZ.dim() - 1))
        self.b.add_to_gradient(torch.sum(dZ, dim=dims))

        dX = dZ @ self.W.parameter.T

        return dX

    def params_list(self):
        """
        List of all layer's Parameters.

        Returns
        -------
        params : list
            List of all layer's Parameters

        """
        params = [self.W, self.b]
        return params