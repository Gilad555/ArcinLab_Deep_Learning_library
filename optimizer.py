import torch

class Optimizer:
    """
    A model optimizer.

    Parameters
    ----------
    model : Sequence
        Neural network model.
    learning_rate : float
        Training learning rate.
    
    Attributes
    ----------
    params : list
        List of model's Parameters.
    """

    def __init__(self, model, learning_rate):
        self.params = model.params_list()
        self.learning_rate = learning_rate

    def apply(self):
        raise NotImplementedError

    def reset_gradients(self):
        for p in self.params:
            p.reset_gradient()




class SGD(Optimizer):
    """
    A model stochastic gradient descent method based optimizer.
    
    Parameters
    ----------
    model : Sequence
        Neural network model.
    learning_rate : float
        Training learning rate.
        
    Attributes
    ----------
    params : list
        List of model's Parameters.
    """
    def __init__(self, model, learning_rate):
        super().__init__(model, learning_rate)

    def apply(self):
        for p in self.params:
            p.parameter = p.parameter - self.learning_rate * p.gradient




class Adam(Optimizer):
    """
    A model adam method based optimizer.

    Parameters
    ----------
    model : Sequence
        Neural network model.
    learning_rate : float
        Training learning rate.
    beta1 : float, optional
        Exponential decay rate of training momentums.
    beta2 : float, optional.
        Exponential decay rate of training variances.
        
    Attributes
    ----------
    params : list
        List of model's Parameters.
    state : dictionary
        Dictionary that holds model's training state.
    t : int
        updating step.
    """

    def __init__(self, model, learning_rate, beta1=0.9, beta2=0.999):
        super().__init__(model, learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2

        # momentum and variance tensor for each model's Parameter
        
        self.state = {}
        for p in self.params:
            m = torch.zeros_like(p.parameter)
            v = torch.zeros_like(p.parameter)
        
            self.state[p] = {"m": m, "v": v}

        self.t = 0

    def apply(self):
        """ Update model's Parameters. """
    
        self.t += 1
    
        for p in self.params:
            m = self.state[p]["m"]
            v = self.state[p]["v"]
    
            m = m * self.beta1 + p.gradient * (1 - self.beta1)
            v = v * self.beta2 + p.gradient**2 * (1 - self.beta2)
    
            self.state[p]["m"] = m
            self.state[p]["v"] = v
    
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
    
            p.parameter = p.parameter - self.learning_rate * m_hat / (torch.sqrt(v_hat) + 1e-8)




class Momentum(Optimizer):
    """
    A model momentum method based optimizer.
    
    Parameters
    ----------
    model : Sequence
        Neural network model.
    learning_rate : float
        Training learning rate.
    momentum : float
        Learning momentum.
            
    Attributes
    ----------
    params : list
        List of model's Parameters.
    velocity : dictionary
        Dictionary that holds model's velocity state.
    """

    def __init__(self, model, learning_rate, momentum):
        super().__init__(model, learning_rate)

        self.momentum = momentum

        # velocity for each model's Parameter

        self.velocity = {}
        for p in self.params:
            self.velocity[p] = torch.zeros_like(p.parameter)


    def apply(self):
        """ Update model's Parameters. """

        for p in self.params:
            v = self.velocity[p]
            v = self.momentum * v + p.gradient

            self.velocity[p] = v

            p.parameter = p.parameter - self.learning_rate * v