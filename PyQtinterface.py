from numba_boosted_perceptron import *
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QPoint
import random
import traceback


def log_uncaught_exceptions(ex_cls, ex, tb):
    text = '{}: {}:\n'.format(ex_cls.__name__, ex)

    text += ''.join(traceback.format_tb(tb))

    print(text)
    QtWidgets.QMessageBox.critical(None, 'Error', text)

    sys.exit()


sys.excepthook = log_uncaught_exceptions


class UI_Design(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(424, 335)
        self.formLayoutWidget = QtWidgets.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(290, 0, 131, 285))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label)
        self.progressBar = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_2)
        self.progressBar_1 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_1.setProperty("value", 24)
        self.progressBar_1.setObjectName("progressBar_1")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_1)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_3)
        self.progressBar_2 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_2)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_4)
        self.progressBar_3 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_3.setProperty("value", 24)
        self.progressBar_3.setObjectName("progressBar_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_3)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_5)
        self.progressBar_4 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_4.setProperty("value", 24)
        self.progressBar_4.setObjectName("progressBar_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_4)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_6)
        self.progressBar_5 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_5.setProperty("value", 24)
        self.progressBar_5.setObjectName("progressBar_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_5)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_7)
        self.progressBar_6 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_6.setProperty("value", 24)
        self.progressBar_6.setObjectName("progressBar_6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_6)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_8)
        self.progressBar_7 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_7.setProperty("value", 24)
        self.progressBar_7.setObjectName("progressBar_7")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_7)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_10)
        self.progressBar_9 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_9.setProperty("value", 24)
        self.progressBar_9.setObjectName("progressBar_9")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_9)
        self.progressBar_8 = QtWidgets.QProgressBar(self.formLayoutWidget)
        self.progressBar_8.setProperty("value", 24)
        self.progressBar_8.setObjectName("progressBar_8")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.FieldRole, self.progressBar_8)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(0, 290, 421, 41))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "0"))
        self.label_2.setText(_translate("Dialog", "1"))
        self.label_3.setText(_translate("Dialog", "2"))
        self.label_4.setText(_translate("Dialog", "3"))
        self.label_5.setText(_translate("Dialog", "4"))
        self.label_6.setText(_translate("Dialog", "5"))
        self.label_7.setText(_translate("Dialog", "6"))
        self.label_8.setText(_translate("Dialog", "7"))
        self.label_9.setText(_translate("Dialog", "8"))
        self.label_10.setText(_translate("Dialog", "9"))
        self.pushButton.setText(_translate("Dialog", "CLEAR"))


class App(QWidget, UI_Design):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.point = None
        self.input = [-127] * 784
        self.pushButton.clicked.connect(self.clear)
        self.arr = [self.progressBar, self.progressBar_1, self.progressBar_2, self.progressBar_3, self.progressBar_4, self.progressBar_5, self.progressBar_6, self.progressBar_7, self.progressBar_8, self.progressBar_9]
        self.network = NeuralNetwork(learn_speed=0.8, sensors_amount=784, hidden_layer_amount=2, neuron_amount_in_hidden_layers=[32] * 2, result_neurons_amount=10)
        self.network.import_weights("nums_recognition_MNIST3")
        self.ans = [0] * 10
        for bar in range(len(self.arr)):
            self.arr[bar].setValue(0)

    def mousePressEvent(self, event):
        self.point = event.pos()
        if event.pos().x() <= 280 and event.pos().y() <= 280:
            self.input[event.pos().y() // 10 * 28 + event.pos().x() // 10 % 28] = 127
            self.input[event.pos().y() // 10 * 28 + (event.pos().x() - 10) // 10 % 28] = 127
            self.input[event.pos().y() // 10 * 28 + (event.pos().x() + 10) // 10 % 28] = 127
            self.input[(event.pos().y() - 10) // 10 * 28 + event.pos().x() // 10 % 28] = 127
            self.input[(event.pos().y() + 10) // 10 * 28 + event.pos().x() // 10 % 28] = 127
        self.num_recognition(self.input.copy())
        for bar in range(len(self.arr)):
            self.arr[bar].setValue(self.ans[bar])
        self.update()

    def num_recognition(self, input_data):
        self.network.set_input(input_data, 0)
        self.network.forward_propagation()
        self.ans = [int(i.value * 100) for i in self.network.result_neurons_arr]

    def mouseMoveEvent(self, event):
        self.point = event.pos()
        if event.pos().x() <= 280 and event.pos().y() <= 280:
            self.input[event.pos().y() // 10 * 28 + event.pos().x() // 10 % 28] = 127
            self.input[event.pos().y() // 10 * 28 + (event.pos().x() - 10) // 10 % 28] = 127
            self.input[event.pos().y() // 10 * 28 + (event.pos().x() + 10) // 10 % 28] = 127
            self.input[(event.pos().y() - 10) // 10 * 28 + event.pos().x() // 10 % 28] = 127
            self.input[(event.pos().y() + 10) // 10 * 28 + event.pos().x() // 10 % 28] = 127
        self.num_recognition(self.input.copy())
        for bar in range(len(self.arr)):
            self.arr[bar].setValue(int(self.ans[bar]))
        self.update()

    def clear(self):
        self.input = [-127] * 784
        self.update()

    def mouseReleaseEvent(self, event):
        self.point = None

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 0, 0, 255), 10))
        for i in range(784):
            if self.input[i] == 127:
                painter.drawPoint(QPoint(i % 28 * 10 + 5, i // 28 * 10 + 5))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = App()
    myapp.show()
    sys.exit(app.exec())
