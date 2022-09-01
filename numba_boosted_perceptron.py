import numpy as np
import random
import math
import json
from numba import njit


def sigmoid(value):
    return 1 / (1 + math.e ** (-value))


def sigmoid_derivative(value):
    return value * (1 - value)


class SenorNeuron:
    def __init__(self, number_in_layer, value=0, activation_function=sigmoid):
        self.value = value
        self.number_in_layer = number_in_layer
        self.activation_function = activation_function

    def set_value(self, value):
        self.value = self.activation_function(value)

    def __str__(self):
        return str(self.number_in_layer) + "(SenorNeuron value=" + str(self.value) + ")"


class HiddenNeuron:
    def __init__(self, number_in_layer, previous_layer_neurons, activation_function=sigmoid):
        self.previous_layer_neurons = previous_layer_neurons
        self.weights = np.array([random.random() * 2 - 1 for _ in range(len(previous_layer_neurons))])
        self.activation_function = activation_function
        self.number_in_layer = number_in_layer
        self.delta_weights = np.array([0] * len(self.weights))
        self.value = 0
        self.error = 0
        self.bias = 0
        self.delta_bias = 0
        self.examples_number = 0
        self.era_number = 0

    def __str__(self):
        return f"{str(self.number_in_layer)}(HiddenNeuron value={str(self.value)} error={self.error})"

    def calculate(self, clear_error=True):
        self.value = sigmoid(sum(np.array([i.value for i in self.previous_layer_neurons]) * self.weights) + self.bias)
        if clear_error:
            self.error = 0

    def count_stochastic_error(self, next_neurons_layer):
        self.error = sum((np.array([next_neuron.error for next_neuron in next_neurons_layer]) * np.array([next_neuron.weights[self.number_in_layer] for next_neuron in next_neurons_layer]))) * sigmoid_derivative(
            self.value)

    def count_batch_weights_adjustment(self, previous_neuron_layer, next_neurons_layer):  # You should multiply on ls further
        self.error = sum((np.array([next_neuron.error for next_neuron in next_neurons_layer]) * np.array([next_neuron.weights[self.number_in_layer] for next_neuron in next_neurons_layer]))) * sigmoid_derivative(
            self.value)
        self.delta_weights = self.delta_weights - np.array([sensor.value for sensor in previous_neuron_layer]) * self.error
        self.delta_bias -= self.error
        self.era_number += 1


class ResultNeuron(HiddenNeuron):
    def __str__(self):
        return f"{str(self.number_in_layer)}(ResultNeuron value={str(self.value)} error={self.error})"

    def count_stochastic_error(self, answer):
        self.error = (self.value - answer) * sigmoid_derivative(self.value)

    def count_batch_weights_adjustment(self, previous_neuron_layer, answer):
        self.error = (self.value - answer) * sigmoid_derivative(self.value)
        self.delta_weights = self.delta_weights - np.array([sensor.value for sensor in previous_neuron_layer]) * self.error
        self.delta_bias -= self.error
        self.era_number += 1


class NeuralNetwork:
    def __init__(self, learn_speed=0.1, sensors_amount=4, hidden_layer_amount=2, neuron_amount_in_hidden_layers=None, result_neurons_amount=2, education_type="stochastic"):  # education_type="stochastic" or "batch"
        if neuron_amount_in_hidden_layers is None:
            neuron_amount_in_hidden_layers = [3, 3]
        self.learn_speed = learn_speed
        self.sensors_arr = np.array([SenorNeuron(i) for i in range(sensors_amount)])
        self.hidden_layers_arr = [
            [HiddenNeuron(_, self.sensors_arr) for _ in range(neuron_amount_in_hidden_layers[0])]]
        for i in range(1, hidden_layer_amount):
            self.hidden_layers_arr.append(
                [HiddenNeuron(_, self.hidden_layers_arr[i - 1]) for _ in range(neuron_amount_in_hidden_layers[i])])
        self.hidden_layers_arr = np.array(self.hidden_layers_arr, dtype=object)
        self.result_neurons_arr = np.array([ResultNeuron(i, self.hidden_layers_arr[-1]) for i in range(result_neurons_amount)])
        self.answer = [0, 1]
        self.education_type = education_type

    def calculate_network_error(self):
        return sum([(self.result_neurons_arr[i].value - self.answer[i]) ** 2 for i in range(len(self.result_neurons_arr) - 1)]) / (len(self.result_neurons_arr) - 1)

    def forward_propagation(self, clear_error=True):
        for hidden_layer in self.hidden_layers_arr:
            for hidden_neuron in hidden_layer:
                hidden_neuron.calculate(clear_error=clear_error)
        for answer_neuron in self.result_neurons_arr:
            answer_neuron.calculate(clear_error=clear_error)

    def stochastic_calculate_errors(self):
        for result_neuron in range(len(self.result_neurons_arr)):
            self.result_neurons_arr[result_neuron].count_stochastic_error(self.answer[result_neuron])
        for hidden_layer in range(len(self.hidden_layers_arr))[::-1]:
            if hidden_layer == len(self.hidden_layers_arr) - 1:
                for hidden_neuron in self.hidden_layers_arr[hidden_layer]:
                    hidden_neuron.count_stochastic_error(self.result_neurons_arr)
            else:
                for hidden_neuron in self.hidden_layers_arr[hidden_layer]:
                    hidden_neuron.count_stochastic_error(self.hidden_layers_arr[hidden_layer + 1])

    def batch_calculate_errors(self):
        for result_neuron in range(len(self.result_neurons_arr)):
            self.result_neurons_arr[result_neuron].count_batch_error(self.answer[result_neuron])
        for hidden_layer in range(len(self.hidden_layers_arr))[::-1]:
            for hidden_neuron in self.hidden_layers_arr[hidden_layer]:
                if hidden_layer == len(self.hidden_layers_arr) - 1:
                    for sinaps in self.result_neurons_arr:
                        hidden_neuron.count_batch_error(sinaps)
                else:
                    for sinaps in self.hidden_layers_arr[hidden_layer + 1]:
                        hidden_neuron.count_batch_error(sinaps)

    def export_weights(self, file_name):
        conf_file = open(file_name + "(configuration_information).txt", "w")
        json_conf_info = json.dumps({
            "sensors_amount": len(self.sensors_arr),
            "hidden_layer_amount": len(self.hidden_layers_arr),
            "neuron_amount_in_hidden_layers": [len(hidden_layer) for hidden_layer in self.hidden_layers_arr],
            "result_neurons_amount": len(self.result_neurons_arr)
        }, indent=4)
        conf_file.write(json_conf_info)
        conf_file.close()
        weights_file = open(file_name + "(weights_information).txt", "w")
        json_conf_info = json.dumps({
            "hidden_layers_neurons": [
                [(list(hidden_neuron.weights), hidden_neuron.bias) for hidden_neuron in hidden_layer] for
                hidden_layer in self.hidden_layers_arr],
            "result_neurons": [(list(result_neuron.weights), result_neuron.bias) for result_neuron in self.result_neurons_arr]
        }, indent=4)
        weights_file.write(json_conf_info)
        weights_file.close()

    def import_weights(self, file_name):
        conf_file = open(file_name + "(configuration_information).txt", "r")
        conf_json = json.loads(conf_file.read())
        conf_file.close()
        if conf_json["sensors_amount"] == len(self.sensors_arr) and conf_json["hidden_layer_amount"] == len(self.hidden_layers_arr) \
                and conf_json["neuron_amount_in_hidden_layers"] == [len(hidden_layer) for hidden_layer in self.hidden_layers_arr] and conf_json["result_neurons_amount"] == len(self.result_neurons_arr):
            print("import is being applied")
            weights_file = open(file_name + "(weights_information).txt", "r")
            weights_file_json = json.loads(weights_file.read())
            for hidden_layer in range(len(self.hidden_layers_arr)):
                for hidden_neuron in range(len(self.hidden_layers_arr[hidden_layer])):
                    self.hidden_layers_arr[hidden_layer][hidden_neuron].weights = np.array(weights_file_json["hidden_layers_neurons"][hidden_layer][hidden_neuron][0])
                    self.hidden_layers_arr[hidden_layer][hidden_neuron].bias = weights_file_json["hidden_layers_neurons"][hidden_layer][hidden_neuron][1]
            for result_neuron in range(len(self.result_neurons_arr)):
                self.result_neurons_arr[result_neuron].weights = np.array(weights_file_json["result_neurons"][result_neuron][0])
                self.result_neurons_arr[result_neuron].bias = weights_file_json["result_neurons"][result_neuron][1]
            weights_file.close()
            print("import applied")
        else:
            raise Exception(f"""
import configuration does not match your configuration
import configuration:
sensors_amount={conf_json["sensors_amount"]}
hidden_layer_amount={conf_json["hidden_layer_amount"]}
neuron_amount_in_hidden_layers={conf_json["neuron_amount_in_hidden_layers"]}
result_neurons_amount={conf_json["result_neurons_amount"]}""")

    def stochastic_weights_adjustment(self):
        for hidden_neuron in self.hidden_layers_arr[0]:
            hidden_neuron.weights -= np.array([sensor.value for sensor in self.sensors_arr]) * hidden_neuron.error * self.learn_speed
            hidden_neuron.bias -= hidden_neuron.error * self.learn_speed
        if len(self.hidden_layers_arr) > 1:
            for i in range(len(self.hidden_layers_arr))[1:]:
                for hidden_neuron in self.hidden_layers_arr[i]:
                    hidden_neuron.weights -= np.array([sensor.value for sensor in self.hidden_layers_arr[i - 1]]) * hidden_neuron.error * self.learn_speed
                    hidden_neuron.bias -= hidden_neuron.error * self.learn_speed
        for result_neuron in self.result_neurons_arr:
            result_neuron.weights -= np.array([sensor.value for sensor in self.hidden_layers_arr[-1]]) * result_neuron.error * self.learn_speed
            result_neuron.bias -= result_neuron.error * self.learn_speed

    def batch_delta_weights_calculation(self):
        for result_neuron_id in range(len(self.result_neurons_arr)):
            self.result_neurons_arr[result_neuron_id].count_batch_weights_adjustment(self.hidden_layers_arr[-1], self.answer[result_neuron_id])
        if len(self.hidden_layers_arr) > 1:
            for i in range(len(self.hidden_layers_arr))[1:][::-1]:
                if i == len(self.hidden_layers_arr) - 1:
                    for hidden_neuron in self.hidden_layers_arr[i]:
                        hidden_neuron.count_batch_weights_adjustment(self.hidden_layers_arr[i - 1], self.result_neurons_arr)
                else:
                    for hidden_neuron in self.hidden_layers_arr[i]:
                        hidden_neuron.count_batch_weights_adjustment(self.hidden_layers_arr[i - 1], self.hidden_layers_arr[i])
        if len(self.hidden_layers_arr) > 1:
            for hidden_neuron in self.hidden_layers_arr[0]:
                hidden_neuron.count_batch_weights_adjustment(self.sensors_arr, self.hidden_layers_arr[1])
        else:
            for hidden_neuron in self.hidden_layers_arr[0]:
                hidden_neuron.count_batch_weights_adjustment(self.sensors_arr, self.result_neurons_arr)

    def batch_weights_adjustment(self):
        for hidden_layer in self.hidden_layers_arr:
            for hidden_neuron in hidden_layer:
                hidden_neuron.weights += hidden_neuron.delta_weights * self.learn_speed / hidden_neuron.era_number
                hidden_neuron.delta_weights *= 0
                hidden_neuron.bias += hidden_neuron.delta_bias * self.learn_speed / hidden_neuron.era_number
                hidden_neuron.delta_bias = 0
                hidden_neuron.era_number = 0
        for result_neuron in self.result_neurons_arr:
            result_neuron.weights += result_neuron.delta_weights * self.learn_speed / result_neuron.era_number
            result_neuron.delta_weights *= 0
            result_neuron.bias += result_neuron.delta_bias * self.learn_speed / result_neuron.era_number
            result_neuron.delta_bias = 0
            result_neuron.era_number = 0

    def stochastic_backward_propagation(self):
        self.stochastic_calculate_errors()
        self.stochastic_weights_adjustment()

    def set_input(self, input_data, answer):
        self.answer = answer
        for sensor in self.sensors_arr:
            sensor.set_value(input_data.pop())

    def show_all_data(self, text_limit=80, show_sensors=False):
        for i in range(max([len(self.sensors_arr) if show_sensors else 0, len(self.result_neurons_arr)] + [len(j) for j in self.hidden_layers_arr])):
            text_limit_x = text_limit
            s = ""
            if show_sensors:
                if len(self.sensors_arr) > i:
                    s += str(self.sensors_arr[i])
                    s += " " * (text_limit_x - len(s))
                    text_limit_x += text_limit
                else:
                    s += " " * text_limit
                    text_limit_x += text_limit
            for hidden_layer in self.hidden_layers_arr:
                if len(hidden_layer) > i:
                    s += str(hidden_layer[i])
                    s += " " * (text_limit_x - len(s))
                    text_limit_x += text_limit
                else:
                    s += " " * text_limit
                    text_limit_x += text_limit
            if len(self.result_neurons_arr) > i:
                s += str(self.result_neurons_arr[i])
                s += " " * (text_limit_x - len(s))
                text_limit_x += text_limit
            else:
                s += " " * text_limit
                text_limit_x += text_limit
            print(s)
        # print(*[str(i) + " " * (text_limit - 1 - len(str(i))) for i in self.biases])
