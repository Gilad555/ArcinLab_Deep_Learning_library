import torch

class Parameter:
    """
    A model's Parameter.

    Parameters
    ----------
    X : torch.tensor
        Initial values of parameter.

    Attributes
    ----------
    parameter : torch.Tensor
        Current values.
    gradient : torch.Tensor
        Current gradient, same shape as parameter.
    """

    def __init__(self, X):
        self.parameter = X
        self.gradient = torch.zeros_like(X)

    def reset_gradient(self):
        """
        Reset gradient values to zero.
        """

        self.gradient = torch.zeros_like(self.gradient)

    def add_to_gradient(self, X):
        """
        Add values to gradient.

        Parameters
        ----------
        X : torch.Tensor
            Values adding to gradient.
        
        Raises
        ------
        ValueError
            If the shape of X does not match the gradient shape.
        """

        if(X.shape == self.gradient.shape):
            self.gradient = self.gradient + X
        else:
            raise ValueError(f"Gradient shape {X.shape} does not match {self.gradient.shape}")

            
        