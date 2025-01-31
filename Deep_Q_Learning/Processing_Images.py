import numpy as np
import cv2
import typing as tt
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 3:
        std_grayscale_weights = np.array([0.2989, 0.5870, 0.1140])
        grayscaleImage = np.dot(image[..., :3], std_grayscale_weights)
        return grayscaleImage
    return image

def resize_image(image: np.array, new_shape: tt.Tuple[int, ...]) -> np.array:
    if image.size != new_shape:
        resized_image = cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

class AtariWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stack_size=4, new_shape=(84,84)):
        super().__init__(env)
        self.stack_size=stack_size
        self.frames = deque(maxlen=stack_size)
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

        while len(self.frames) < self.stack_size:
            self.frames.append(resXgray / 255.0) #Normalizing for CNN

        return np.stack(self.frames, axis=-1)

        
        
