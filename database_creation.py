from sklearn.datasets import fetch_openml
import json
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")



mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(int)


answers = mnist["target"]
images = [[] for i in range(70000)]
for pixel_col in range(1, 785):
    for data, i in zip(mnist["data"][f"pixel{pixel_col}"], range(len(mnist["data"][f"pixel{pixel_col}"]))):
        images[i].append(data)
    print(pixel_col)

file = open("Mnist_databse.txt", "w")
file.write(json.dumps({
    "data": [{"image": list(images[i]),
              "answer": int(answers[i])} for i in range(len(images))]
}, indent=4))
file.close()
