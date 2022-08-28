from perceptron import *

f = open("Mnist_databse.txt", "r")
json_d = json.loads(f.read())
f.close()
print("file read")
# inputs = [(json_d["data"][i]["image"], [1 if json_d["data"][i]["answer"] == j else 0 for j in range(10)]) for i in range(70000)]
inputs = [(np.array(json_d["data"][i]["image"]) - 127, [1 if json_d["data"][i]["answer"] == j else 0 for j in range(10)]) for i in range(70000)]
learn_inputs = inputs[:60000]

test_inputs = inputs[60000:]
print("Inputs ready")

network = NeuralNetwork(learn_speed=0.8, sensors_amount=784, hidden_layer_amount=2, neuron_amount_in_hidden_layers=[20] * 2, result_neurons_amount=10)
network.import_weights("nums_recognition_MNIST3")
r_ans = 0
print("Network ready")
for test in test_inputs:
    network.set_input(input_data=list(test[0]), answer=test[1])
    network.forward_propagation()
    arr = [network.result_neurons_arr[i].value for i in range(len(network.result_neurons_arr))]
    print(f"network is {max(arr) * 100 // 1} % sure:", arr.index(max(arr)))
    arr1 = network.answer
    print("answer:", arr1.index(max(arr1)))
    if arr.index(max(arr)) == arr1.index(max(arr1)):
        r_ans += 1
        print(r_ans)

print("Ai gave " + str(r_ans / len(test_inputs) * 100 // 1) + "% of right answers")

# network.show_all_data()
