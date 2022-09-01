from sklearn.datasets import fetch_openml
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

file = open("Mnist_database.txt", "w")
data = [(np.array(images[i]) - 127, answers[i]) for i in range(70000)]
data.sort(key=lambda x: x[1])
splited_numbers = []
j = 0
for i in range(len(data) - 1):
    if data[i][1] != data[i + 1][1] or i + 2 == len(data):
        splited_numbers.append(data[j:i + 1].copy())
        j = i + 1
ready_data = []
print([len(i) for i in splited_numbers])
s = 0
while len(splited_numbers[1]) > 0:
    s += 1
    if len(splited_numbers[s % 10]):
        ready_data.append(splited_numbers[s % 10].pop())
ready_data = ready_data[::-1].copy()
ready_data = [(ready_data[i][0], [1 if ready_data[i][1] == j else 0 for j in range(10)]) for i in range(len(ready_data))]
print([i[1] for i in ready_data])
print(len(ready_data))
file.write(json.dumps({
    "data": [{"image": list(ready_data[i][0]),
              "answer": list(ready_data[i][1])} for i in range(len(ready_data))]
}, indent=4))
file.close()
