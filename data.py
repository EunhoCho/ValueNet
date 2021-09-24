import numpy as np
import csv

TARGET = ['1_1_1_1_1_4']
COLUMNS = 17
DECISION_LENGTH = 9

if __name__ == "__main__":
    for target in TARGET:
        x = []
        y = []
        with open('data/' + target + '.csv', 'r', newline='') as data_file:
            data = list(np.array(list(csv.reader(data_file))[1:]).astype(int))

            data_length = []
            for i in range(len(data)):
                x.append(list(data[i][:COLUMNS]))
                if i % DECISION_LENGTH == 0:
                    y.append(data[i][COLUMNS])

        with open('x/' + target + '.csv', 'w', newline='') as x_file:
            csv_writer = csv.writer(x_file)
            csv_writer.writerows(x)

        with open('y/' + target + '.csv', 'w', newline='') as y_file:
            csv_writer = csv.writer(y_file)
            for line in y:
                csv_writer.writerow([line])
