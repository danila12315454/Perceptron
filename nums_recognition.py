from perceptron import *
import numpy as np
from progress.bar import IncrementalBar
export_file = "nums_recognition_MNIST3"

f = open("Mnist_databse.txt", "r")
json_d = json.loads(f.read())
f.close()
print("file read")
inputs = [(np.array(json_d["data"][i]["image"]) - 127, [1 if json_d["data"][i]["answer"] == j else 0 for j in range(10)]) for i in range(70000)]
split_point = 60000
learn_inputs = inputs[:split_point]

test_inputs = inputs[split_point:]
print("Inputs ready")
network = NeuralNetwork(learn_speed=0.8, sensors_amount=784, hidden_layer_amount=2, neuron_amount_in_hidden_layers=[16] * 2, result_neurons_amount=10)
print(f"Network ready(after learning weights can de seen on {export_file})")
r_ans = 0
for era in range(10):
    network.learn_speed *= 0.85
    learn_progress = IncrementalBar(f"Learn progress for Era(stochastic):{era}", max=len(learn_inputs))
    for learn_input in learn_inputs:
        network.set_input(input_data=list(learn_input[0]), answer=learn_input[1])
        network.forward_propagation()
        network.stochastic_backward_propagation()
        learn_progress.next()
    learn_progress.finish()
    r_ans = 0
    test_progress = IncrementalBar(f"Test progress for Era:{era}", max=1000)
    for test in test_inputs[:1000]:
        network.set_input(input_data=list(test[0]), answer=test[1])
        network.forward_propagation()
        arr = [network.result_neurons_arr[i].value for i in range(len(network.result_neurons_arr))]
        arr1 = network.answer
        if arr.index(max(arr)) == arr1.index(max(arr1)):
            r_ans += 1
        test_progress.next()
    test_progress.finish()
    print(f"Era:{era} answered right on {r_ans / 10}%", "\n")
network.learn_speed = 10
for era in range(100):
    network.learn_speed *= 0.956
    learn_progress = IncrementalBar(f"Learn progress for Era(batch):{era + 10}", max=len(learn_inputs))
    for learn_input in learn_inputs:
        network.set_input(input_data=list(learn_input[0]), answer=learn_input[1])
        network.forward_propagation()
        network.batch_delta_weights_calculation()
        learn_progress.next()
    network.batch_weights_adjustment()
    learn_progress.finish()
    r_ans = 0
    test_progress = IncrementalBar(f"Test progress for Era:{era}", max=10000)
    for test in test_inputs[:10000]:
        network.set_input(input_data=list(test[0]), answer=test[1])
        network.forward_propagation()
        arr = [network.result_neurons_arr[i].value for i in range(len(network.result_neurons_arr))]
        arr1 = network.answer
        if arr.index(max(arr)) == arr1.index(max(arr1)):
            r_ans += 1
        test_progress.next()
    test_progress.finish()
    print(f"Era:{era} answered right on {r_ans / 100}%", "\n")


for test in test_inputs[:100]:
    network.set_input(input_data=list(test[0]), answer=test[1])
    network.forward_propagation()
    arr = [network.result_neurons_arr[i].value for i in range(len(network.result_neurons_arr))]
    print(f"network is {max(arr) * 100 // 1} % sure:", arr.index(max(arr)))
    arr = network.answer
    print("answer:", arr.index(max(arr)))
network.export_weights(export_file)

#comand line command
"""
cd WP\Python\AI_trening
venv\Scripts\activate
color 2
cls
python nums_recognition.py
"""