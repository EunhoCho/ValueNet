import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

TARGET = ['1_1_1_1_1_4']
COLUMNS = 17
DECISION_LENGTH = 9
PLOT_NUM = [0, 5, 10, 70, 99]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device


class ValueNet(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(ValueNet, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size * seq_length, 128)  # fully connected 1
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)  # internal state

        # Propagate input through LSTM output
        (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.reshape(-1, self.hidden_size * self.seq_length)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output return out

        return out


if __name__ == "__main__":
    for target in TARGET:
        x_file = open('data/x/' + target + '.csv', 'r', newline='')
        x = np.array(list(csv.reader(x_file))).astype(int)
        x_file.close()

        y_file = open('data/y/' + target + '.csv', 'r', newline='')
        y = np.array(list(csv.reader(y_file))).astype(int)
        y_file.close()

        mm = MinMaxScaler()
        ss = StandardScaler()
        x_ss = ss.fit_transform(x)
        y_mm = mm.fit_transform(y)

        x_train = x_ss[:569646, :]
        x_test = x_ss[569646:, :]
        y_train = y_mm[:63294, :]
        y_test = y_mm[63294:, :]

        x_train_tensors = Variable(torch.Tensor(x_train))
        x_test_tensors = Variable(torch.Tensor(x_test))
        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))

        x_train_tensors_final = torch.reshape(x_train_tensors,
                                              (int(x_train_tensors.shape[0] / 9), 9, x_train_tensors.shape[1]))
        x_test_tensors_final = torch.reshape(x_test_tensors, (int(x_test_tensors.shape[0] / 9), 9, x_test_tensors.shape[1]))

        print("Training Shape", x_train_tensors_final.shape, y_train_tensors.shape)
        print("Testing Shape", x_test_tensors_final.shape, y_test_tensors.shape)

        num_epochs = 10000  # 1000 epochs
        learning_rate = 0.00001  # 0.001 lr

        input_size = 17  # number of features
        hidden_size = 2  # number of features in hidden state
        num_layers = 1  # number of stacked lstm layers

        num_classes = 1  # number of output classes
        valueNet = ValueNet(num_classes, input_size, hidden_size, num_layers, x_train_tensors_final.shape[1]).to(DEVICE)

        loss_function = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(valueNet.parameters(), lr=learning_rate)  # adam optimizer

        for epoch in range(num_epochs):
            outputs = valueNet.forward(x_train_tensors_final.to(DEVICE))  # forward pass
            optimizer.zero_grad()  # calculate the gradient, manually setting to 0

            # obtain the loss function
            loss = loss_function(outputs, y_train_tensors.to(DEVICE))
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e backpropagation

            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


        # PLOT
        for plot_num in PLOT_NUM:
            x_plot = x[plot_num * 8631:(plot_num + 1) * 8631]
            y_plot = y[plot_num * 959:(plot_num + 1) * 959]
            x_plot_ss = Variable(torch.Tensor(ss.transform(x_plot)))
            y_plot_tensors = Variable(torch.Tensor(mm.transform(y_plot)))
            x_plot_tensors = torch.reshape(x_plot_ss, (int(x_plot_ss.shape[0] / 9), 9, x_plot_ss.shape[1]))

            train_predict = valueNet(x_plot_tensors.to(DEVICE))
            data_predict = train_predict.data.detach().cpu().numpy()
            data_y_plot = y_plot_tensors.data.numpy()

            data_predict = mm.inverse_transform(data_predict)
            data_y_predict = mm.inverse_transform(data_y_plot)

            plt.plot(data_y_predict, label='Actual Data')  # actual plot
            plt.plot(data_predict, label='Predicted Data')  # predicted plot
            plt.legend()
            plt.show()
