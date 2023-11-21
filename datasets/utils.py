from torch.utils.data import BatchSampler

class DeterministicBatchSampler(BatchSampler):
    def __init__(self, indices):
        '''Subclass of BatchSampler that always returns the list <indices> when called.'''
        self.indices = indices
        super().__init__(self, len(indices), drop_last=False)
    def __iter__(self):
        yield self.indices