import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

with np.load("../NeuroLearning/neuro.npz") as data:
    weights_input_to_hidden = data['arr_0']
    weights_hidden_to_output = data['arr_1']
    bias_input_to_hidden = data['arr_2']
    bias_hidden_to_output = data['arr_3']

p = Path("../CustomImages")
for file in p.rglob("*"):
    image = plt.imread(file, format="jpeg")

    gray = lambda rgb: np.dot(rgb[... , :3], np.array([0.299, 0.587, 0.114]))
    image = 1 - (gray(image).astype(np.float32) / 255)

    image = np.reshape(image, (-1, 1))

    hidden_raw = np.dot(weights_input_to_hidden, image) + bias_input_to_hidden
    hidden = 1 / (1 + np.exp(-hidden_raw))

    output_raw = np.dot(weights_hidden_to_output, hidden) + bias_hidden_to_output
    output = 1 / (1 + np.exp(-output_raw))

    plt.imshow(image.reshape(28, 28), cmap="Greys")
    plt.title(f"Number: {output.argmax()}")
    plt.show()

