#Modules to test
from Processing_Images import AtariWrapper, grayscale, resize_image
from Conv import CNN, torchify
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#For specific Atari Environment
from ale_py import ALEInterface
import ale_py
import shimmy

#Processing_Images Test
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
obs, _ = env.reset()
"""
print(obs.shape)

gray = grayscale(obs)
print(gray.shape)


reshaped = resize_image(gray, new_shape=(84, 84))
print(reshaped.shape)
"""
wrapped_env = AtariWrapper(env, stack_size=4, new_shape=(84, 84))

# Reset environment and get the first observation
obs, _ = wrapped_env.reset()
processed_obs = wrapped_env.observation(obs)
"""
print('Full processed Environment shape', processed_obs.shape)

# Display one of the processed frames
plt.imshow(processed_obs[..., 0], cmap='gray')  # Show the first frame
plt.title("Processed Observation (Grayscale & Resized)")
plt.show()
"""

#CNN Test
convModel = CNN((84, 84, 4), output_shape=(1,), dropout=0.0)

randomTensor = np.random.rand(4, 84, 84, 4).astype(np.float32)

output = convModel.forward(randomTensor)
print(output)

out = convModel.forward(processed_obs)
print(out)

