from .dataset import *
import random

class BatchesMaker:
    """
    A data mini batches maker.

    Parameters
    ----------
    data : torch.Tensor
        A tensor which contains all samples.
    labels : torch.Tensor
        A tensor which contains data labels corresponding to data's samples.
    batch_size : int
        Number of samples in each mini batch.
    shuffle : bool, optional
        Whether to shuffle samples or not.
    
    Attributes
    ----------
    dataset : Dataset
        Contain data and labels for batch creating.
    """

    def __init__(self, data, labels, batch_size, shuffle=True):
        self.dataset = Dataset(data, labels)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """
        Create mini batch each call.

        Yields
        ------
        data : torch.Tensor
            Tensor of samples.
        labels : torch.Tensor
            Tensor of labels.
        """
        samples = list(range(self.dataset.num_samples))

        if(self.shuffle):
            random.shuffle(samples)
        for b in range(0, len(samples), self.batch_size):
            yield self.dataset.batch(samples[b:b+self.batch_size])