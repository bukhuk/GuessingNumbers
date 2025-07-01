import numpy as np

import utils
from constants import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, EPOCHS, LEARNING_RATE

images, labels = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (HIDDEN_SIZE, INPUT_SIZE))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (OUTPUT_SIZE, HIDDEN_SIZE))

bias_input_to_hidden = np.zeros((HIDDEN_SIZE, 1))
bias_hidden_to_output = np.zeros((OUTPUT_SIZE, 1))

for epoch in range(EPOCHS):
    print(f"Epoch №{epoch}")

    e_loss = 0
    e_correct = 0

    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        hidden_raw = np.dot(weights_input_to_hidden, image) + bias_input_to_hidden
        hidden = 1 / (1 + np.exp(-hidden_raw))

        output_raw = np.dot(weights_hidden_to_output, hidden) + bias_hidden_to_output
        output = 1 / (1 + np.exp(-output_raw))

        e_loss += 1 / len(output) * np.sum((label - output) ** 2, axis=0)
        e_correct += int(np.argmax(label) == np.argmax(output))

        delta_output = output - label
        weights_hidden_to_output += -LEARNING_RATE * np.dot(delta_output, np.transpose(hidden))
        bias_hidden_to_output += -LEARNING_RATE * delta_output

        delta_hidden = np.dot(np.transpose(weights_hidden_to_output), delta_output) * (hidden * (1 - hidden))
        weights_input_to_hidden += -LEARNING_RATE * np.dot(delta_hidden, np.transpose(image))
        bias_input_to_hidden += -LEARNING_RATE * delta_hidden

    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%\n")

# SAVE NEURO

np.savez("neuro.npz", weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden, bias_hidden_to_output)