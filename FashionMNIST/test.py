import random
import torch
import matplotlib.pyplot as plt

from main import model_path, NeuralNetwork, test_data


classes = [
    "T-shirt/top",
    "Trouser    ",
    "Pullover   ",
    "Dress      ",
    "Coat       ",
    "Sandal     ",
    "Shirt      ",
    "Sneaker    ",
    "Bag        ",
    "Ankle boot ",
]

model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()


def test(a, b):
    l = len(test_data)
    i = random.randint(0, l - a * b)
    with torch.no_grad():
        for j in range(a * b):
            index = i + j
            x, y = test_data[index][0], test_data[index][1]
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            flag = "Success" if predicted == actual else "Failure"
            print(f"[{flag}] Predicted: {predicted}, Actual: {actual}")

            plt.subplot(a, b, j + 1)
            plt.tight_layout()
            plt.imshow(x[0], cmap='gray', interpolation='none')
            plt.title(f"P: {predicted}\nT: {actual}")
            plt.xticks([])
            plt.yticks([])
        plt.show()


if __name__ == "__main__":
    test(3, 4)
