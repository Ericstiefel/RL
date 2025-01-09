import numpy as np
import cv2
import typing as tt
import gymnasium as gym
from collections import deque


def grayscale(image: np.array) -> np.array:
    assert image.shape[2] == 3
    std_grayscale_weights = np.array([0.2989, 0.5870, 0.1140])
    grayscaleImage = np.dot(image[..., :3], std_grayscale_weights)
    return grayscaleImage

def resize_image(image: np.array, new_shape: tt.Tuple[int, ...]) -> np.array:
    resized_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    return resized_image

class AtariWrapper(gym.ObservationWrapper):
    def __init__(self, obs, stack_size=4, new_shape=(84,84)):
        super().__init__(obs)
        self.stack_size=stack_size
        self.frames = deque()
        self.new_shape=new_shape

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(new_shape[0], new_shape[1], stack_size),
            dtype=np.float32
        )
    
    def observation(self, obs):
        grayImage = grayscale(obs)
        resXgray = resize_image(grayImage, new_shape=self.new_shape)
        if len(self.frames) == self.stack_size:
            self.frames.popleft()
        while len(self.frames) < self.frames:
            self.frames.append(resXgray / 255.0) #Normalizing for CNN
        return np.stack(self.frames, axis=-1)

        
        
