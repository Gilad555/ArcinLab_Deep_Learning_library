import torch
import math

class Activation:
    """ An activation function with derivative function. """

    def forward(self, X):
        raise NotImplementedError

    def backward(self, X):
        raise NotImplementedError


    

class Linear(Activation):
    """ A Linear activation function with derivative function. """

    def forward(self, X):
        """
        Linear function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return X
 
    def backward(self, X):
        """
        Linear function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.ones_like(X)




class ReLU(Activation):
    """ A ReLU activation function with derivative function. """

    def forward(self, X):
        """
        ReLU function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.clamp(X, min=0)
 
    def backward(self, X):
        """
        ReLU function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return (X > 0).to(X.dtype)




class LeakyReLU(Activation):
    """ A Leaky ReLU activation function with derivative function. """

    def forward(self, X, negative_slope=0.01):
        """
        Leaky ReLU function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
        negative_slope : float, optional
            Slope for negative inputs.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.where(X > 0, X, negative_slope * X)
 
    def backward(self, X, negative_slope=0.01):
        """
        Leaky ReLU function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
        negative_slope : float, optional
            Slope for negative inputs.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.where(X > 0, torch.ones_like(X), torch.full_like(X, negative_slope))




class ELU(Activation):
    """ A ELU activation function with derivative function. """

    def forward(self, X, alpha=1.0):
        """
        ELU function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
        alpha : float, optional
            Scale for negative inputs.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.where(X > 0, X, alpha * (torch.exp(X) - 1))
 
    def backward(self, X, alpha=1.0):
        """
        ELU function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
        alpha : float, optional
            Scale for negative inputs.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.where(X > 0, torch.ones_like(X), alpha * torch.exp(X))




class Sigmoid(Activation):
    """ A Sigmoid activation function with derivative function. """

    def forward(self, X):
        """
        Sigmoid function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.sigmoid(X)
 
    def backward(self, X):
        """
        Sigmoid function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        s = torch.sigmoid(X)
        return s * (1 - s)




class Tanh(Activation):
    """ A Tanh activation function with derivative function. """

    def forward(self, X):
        """
        Tanh function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.tanh(X)
 
    def backward(self, X):
        """
        Tanh function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        t = torch.tanh(X)
        return 1 - t ** 2




class Softplus(Activation):
    """ A Softplus activation function with derivative function. """

    def forward(self, X):
        """
        Softplus function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.logaddexp(X, torch.zeros_like(X))
 
    def backward(self, X):
        """
        Softplus function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return torch.sigmoid(X)




class SiLU(Activation):
    """ A SiLU activation function with derivative function. """

    def forward(self, X):
        """
        SiLU function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return X * torch.sigmoid(X)
 
    def backward(self, X):
        """
        SiLU function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        s = torch.sigmoid(X)
        return s * (1 + X * (1 - s))




class GELU(Activation):
    """ A GELU activation function with derivative function. """

    def forward(self, X):
        """
        GELU function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        return 0.5 * X * (1 + torch.erf(X / math.sqrt(2)))
 
    def backward(self, X):
        """
        GELU function derivative.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """
        cdf = 0.5 * (1 + torch.erf(X / math.sqrt(2)))
        pdf = torch.exp(-0.5 * X ** 2) / math.sqrt(2 * math.pi)
        return cdf + X * pdf




class Softmax(Activation):
    """ A Softmax activation function with derivative function. """

    def forward(self, X):
        """
        Softmax function.
 
        Parameters
        ----------
        X : torch.Tensor
            Function input.
 
        Returns
        -------
        torch.Tensor
            Function output.
        """

        X = X - torch.amax(X, dim=-1, keepdim=True)
        exp = torch.exp(X)
        return exp / torch.sum(exp, dim=-1, keepdim=True)