import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def set_device(name):
    global device
    device = name

def get_device():
    return device

def he(shape):
    """
    Create and return weight array initialized with He method.

    Parameters
    ----------
    shape : tuple
        Shape of Parameter.

    Returns
    -------
    torch.Tensor
        Weight initialized.

    Raises
    ------
    ValueError
        If shape is one dimensional.
    """

    if(len(shape) < 2):
        raise ValueError(f"shape {shape} cannot be initialized with He.")
        
    in_features = shape[0]

    parameter = torch.randn(shape, device=device)
    parameter_initialized = parameter * (2/in_features) ** 0.5
        
    return parameter_initialized

def xavier(shape):
    """
    Create and return weight array initialized with Xavier method.
    
    Parameters
    ----------
    shape : tuple
        Shape of Parameter.
            
    Returns
    -------
    torch.Tensor
        Weight initialized.
        
    Raises
    ------
    ValueError
        If shape is one dimensional.
    """

    if(len(shape) < 2):
        raise ValueError(f"shape {shape} cannot be initialized with Xavier.")

    in_features = shape[0]
    out_features = shape[1]
    
    parameter = torch.randn(shape, device=device)
    parameter_initialized = parameter * (2/(in_features + out_features)) ** 0.5
            
    return parameter_initialized
    
def zeros(shape):
    """
    Create and return weight array initialized in zeros.
            
    Parameters
    ----------
    shape : tuple
        Shape of Parameter.
                    
    Returns
    -------
    torch.Tensor
        Weight initialized.
    """

    parameter_initialized = torch.zeros(shape, device=device)

    return parameter_initialized

def random_normal(shape, std=0.01):
    """
    Create and return weight array with random normal distribution.
                
    Parameters
    ----------
    shape : tuple
        Shape of Parameter.
    std : float, optional.
        Standard deviation of the distribution
                        
    Returns
    -------
    torch.Tensor
        Weight initialized.
    """
    
    parameter_initialized = torch.randn(shape, device=device) * std
    
    return parameter_initialized