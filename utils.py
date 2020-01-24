import numpy as np
from PIL import Image

def preprocess(observation, image_size):
    state = Image.fromarray(np.uint8(observation))
    state = state.resize((image_size, image_size))
    state = state.convert('L')
    state = np.asarray(state)
    state = state / 255.0
    state = state[np.newaxis, :, :]
    return state
