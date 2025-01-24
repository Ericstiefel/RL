import numpy as np

class Net:
    def __init__(self, obs_size: int, actions: int):
        self.IOsize = (obs_size, actions)
        self.w1 = np.random.randn(obs_size, 16)
        self.w2 = np.random.randn(16, actions)
        self.weights = [self.w1, self.w2]

    def forward(self, obs):
        assert isinstance(obs, np.ndarray),'Obs is not a Numpy Array'

        firstLayer = np.dot(obs, self.w1)
        firstAct = np.array([x if x > 0 else 0 for x in firstLayer], dtype=np.float32)
        secondLayer = np.dot(firstAct, self.w2)
        return np.argmax(secondLayer)