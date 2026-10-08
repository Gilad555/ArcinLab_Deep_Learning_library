class Sequence:
    """
    A sequence of artificial neural network Layers.

    Attributes
    ----------
    layer : list
        List of layers in order.
    """
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
         """
         Add layer to current layers sequence.
         """
         self.layers.append(layer)

    def forward_propagation(self, X):
        """
        Forward propagation through layers sequence.

        Parameters
        ----------
        X : torch.Tensor
            Sequence input.

        Returns
        -------
        torch.Tensor
            Sequence output.
        """
        for current_layer in self.layers:
            X = current_layer.forward_propagation(X)
        return X
    
    def backward_propagation(self, dX):
            """
            backward propagation through layers sequence.
            
            Parameters
            ----------
            dX : torch.Tensor
                Sequence gradient.
            
            Returns
            -------
            torch.Tensor
                Sequence input gradient.
            """
            
            for current_layer in reversed(self.layers):
                dX = current_layer.backward_propagation(dX)
            return dX

    def params_list(self):
        params = []

        for l in self.layers:
            params = params + l.params_list()

        return params
        

