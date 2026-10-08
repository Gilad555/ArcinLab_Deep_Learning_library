class Dataset:
    """
    A dataset.

    Parameters
    ----------
    data : torch.Tensor
        A tensor which contains all samples.
    labels : torch.Tensor
        A tensor which contains data labels corresponding to data's samples.
    
    Attributes
    ----------
    num_samples : int
        Data's number of samples.
    
    Raises
    ------
    ValueError
        If number of data samples does not match number of labels.
    """

    def __init__(self, data, labels):

        if(data.shape[0] != labels.shape[0]):
            raise ValueError(f"number of data samples: {data.shape[0]} does not match number of samples labels: {labels.shape[0]}")
        
        self.data = data
        self.labels = labels

        self.num_samples = data.shape[0]

    def batch(self, samples):
        """
        Return batch from data according to samples.

        Parameters
        ----------
        samples : list
            List of samples indices in data.
        
        Returns 
        -------
        batch : torch.Tensor
            Tensor of samples in order of samples list.
        batch_labels : torch.Tensor
            Tensor of batch labels.
        """

        batch = self.data[samples]
        batch_labels = self.labels[samples]
        return batch, batch_labels