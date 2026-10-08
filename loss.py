import torch

class Loss:
    """ A loss function. """

    def compute(self, X, Y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError




class MSE(Loss):
    """ means sqaured loss function. """

    def compute(self, X, Y):
        """
        Compute loss of prediction.

        Parameters
        ----------
        X : torch.Tensor
            Prediction.
        Y : torch.Tensor
            Labeles of model's prediction.

        Returns
        -------
        loss : torch.Tensor
            Computed loss.
        """

        self._X = X
        self._Y = Y

        m = X.numel()
        self._m = m

        loss = torch.sum((X - Y)**2) / m

        return loss

    def backward(self):
        """
        Gradient of loss input.

        Returns
        -------
        gradient : torch.Tensor
            Gradient of loss input.
        """

        m = self._m

        gradient = 2 * (self._X - self._Y) / m

        return gradient




class CrossEntropy(Loss):
    """ CrossEntropy loss function. """

    def compute(self, X, Y):
        """
        Compute loss of prediction.
        
        Parameters
        ----------
        X : torch.Tensor
            Prediction.
        Y : torch.Tensor
            Labeles of model's prediction.
        
        Returns
        -------
        loss : torch.Tensor
            Computed loss.
        """

        m = X.numel() // X.shape[-1]
        self._m = m

        X = X - torch.max(X, dim=-1, keepdim=True).values
        exp = torch.exp(X)

        S = torch.sum(exp, dim=-1, keepdim=True)

        self._P = exp / S
        self._Y = Y

        loss = torch.sum(-Y * (X - torch.log(S))) / m

        return loss

    def backward(self):
        """
        Gradient of loss input.
        
        Returns
        -------
        gradient : torch.Tensor
            Gradient of loss input.
        """

        m = self._m

        gradient = (self._P - self._Y) / m

        return gradient