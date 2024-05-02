import numpy as np

class WeightInitializer:
    def __init__(self, method='random'):
        self.method = method

    def initialize(self, shape):
        if self.method == 'random':
            return np.random.randn(*shape)
        elif self.method == 'xavier':
            return np.random.randn(*shape) / np.sqrt(shape[0])
        elif self.method == 'he':
            return np.random.randn(*shape) * np.sqrt(2 / shape[0])
        elif self.method == 'uniform':
            return np.random.uniform(-1, 1, shape)
        else:
            raise ValueError(f'Unknown initialization method: {self.method}')