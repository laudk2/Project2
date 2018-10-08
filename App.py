import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

epsilon = 0.00005
alpha = 0.1
errors = []
total_errors = []
weights = [[], [], []]
# Average male height (US): 5.78 ft
# Average female height (US): 5.34 ft
# https://en.wikipedia.org/wiki/List_of_average_human_height_worldwide (2011 - 2014)

# Average male weight (US): 195.8 lb
# Average female weight (US): 168.4 lb
# https://en.wikipedia.org/wiki/Human_body_weight (2011 - 2014)


# *******************
# Generates random data and writes to text file
# *******************
def generate_data():
    male_average_height = 5.78
    male_average_weight = 195.8
    female_average_height = 5.34
    female_average_weight = 168.4
    file = open('SampleData.txt', 'w')
    for x in range(0, 2):
        for y in range(0, 2000):
            if x == 0:
                height = np.random.normal(male_average_height, 0.3)
                weight = np.random.normal(male_average_weight, 20)
                file.write(str(height) + "," + str(weight) + "," + str(x) + "\n")
            else:
                height = np.random.normal(female_average_height, 0.3)
                weight = np.random.normal(female_average_weight, 20)
                file.write(str(height) + "," + str(weight) + "," + str(x) + "\n")
    file.close()


# *******************
# Main function to case each specific scenario (Run 1 at a time)
# *******************
def main():
    # Uncomment if need to generate new dataset
    # generate_data()
    begin_learning(True, 0.25)
    # begin_learning(True, 0.75)

    # begin_learning(False, 0.25)
    # begin_learning(False, 0.75)


# *******************
# Function to set up and begin learning process
# *******************
def begin_learning(hard, fraction):

    normalized_data = normalize_data(pd.read_csv("SampleData.txt", header=None))

    training_data = normalized_data.sample(frac=fraction)

    testing_data = normalized_data[~normalized_data.isin(training_data)]

    random_separation = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    # random_separation = [1, -1, 0.5]

    plt.figure(1)
    final_separation = learning_algorithm(1000, training_data, testing_data, random_separation, hard)
    plot_graphs(normalized_data)
    plt.show()

    plt.figure(2)
    plot_graphs(normalized_data)
    plot_separation(normalized_data, final_separation, color="red")
    plt.axis((0, 1, 0, 1))
    plt.show()

    print_errors_and_weights()


# *******************
# Normalize the passed in dataset to between 0 and 1
# *******************
def normalize_data(all_data):
    normalized_data = all_data.copy()

    normalized_data[0] = (all_data[0] - all_data[0].min()) / (all_data[0].max() - all_data[0].min())
    normalized_data[1] = (all_data[1] - all_data[1].min()) / (all_data[1].max() - all_data[1].min())

    return normalized_data


def plot_graphs(data_frame):
    males = data_frame[data_frame[2] == 0]
    females = data_frame[data_frame[2] == 1]

    male_height = males[0]
    male_weight = males[1]

    female_height = females[0]
    female_weight = females[1]

    male_graph = plt.scatter(male_height, male_weight, c=np.full(males[2].shape, 'orange'), alpha=0.1,  s=50)
    female_graph = plt.scatter(female_height, female_weight, c=np.full(females[2].shape, 'blue'), alpha=0.1, s=50)

    plt.title("Male & Female - Weight vs Height")
    plt.xlabel("Height (ft)")
    plt.ylabel("Weight (lbs)")
    plt.legend((male_graph, female_graph), ('Male', 'Female'), loc='upper right', ncol=3, fontsize=6, scatterpoints=1)

    return plt


def print_errors_and_weights():
    print("----------------------------------------")
    print("Training Error Start: %.4f" % errors[0])
    print("Training Error End: %.4f" % errors[len(errors) - 1])
    print("Training Error Best: %.4f" % min(errors))
    print("----------------------------------------")
    print("Testing Error Start: %.4f" % total_errors[0])
    print("Testing Error End: %.4f" % total_errors[len(total_errors) - 1])
    print("Testing Error Best: %.4f" % min(total_errors))
    print("----------------------------------------")
    print("Initial X-Weight (Random): %.4f" % weights[0][0])
    print("Initial Y-Weight (Random): %.4f" % weights[1][0])
    print("Initial Bias (Random): %.4f" % weights[2][0])
    print("End X-Weight %.4f" % weights[0][len(weights) - 1])
    print("End Y-Weight: %.4f" % weights[1][len(weights) - 1])
    print("End Bias: %.4f" % weights[2][len(weights) - 1])
    print("----------------------------------------")


def plot_separation(data, separation_line, color="0.15"):
    bias = separation_line[2]
    weight_x = separation_line[0]
    weight_y = separation_line[1]

    center = (((data[0].max() - data[0].min()) / 2) + data[0].min())

    y_center = -(((weight_x * center) / weight_y) + (bias / weight_y))
    y1 = -((bias / weight_y) + ((weight_x * data[0].min()) / weight_y))
    y2 = -((bias / weight_y) + ((weight_x * data[0].max()) / weight_y))

    direction = plt.axes()
    direction.arrow(center, y_center, 0.05, 0.05, fc='k', ec='k', color="blue", head_width=0.03, head_length=0.03)

    plt.plot([data[0].min(), data[0].max()], [y1, y2], color=color)
    plt.axis((0, 1, 0, 1))
    return plt


def calculate_new_weight(weight, number, hard=True):
    net = ((weight[0] * number[0]) + (weight[1] * number[1]) + weight[2])

    if hard:
        if net > 0:
            result = 1
        else:
            result = 0
        delta = (number[2] - result) * 0.1
    else:
        result = (1 + np.tanh(net * 0.5)) / 2
        delta = (number[2] - result) * 0.1

    number[0] = number[0] * delta
    number[1] = number[1] * delta
    number[2] = delta

    weight[0] = weight[0] + number[0]
    weight[1] = weight[1] + number[1]
    weight[2] = weight[2] + number[2]

    return weight


def learning_algorithm(iterations, train_data, test_data, separation_input, hard):
    final = None

    for i in range(0, iterations):
        print("iteration", i)
        error1 = get_error(test_data, separation_input)
        total_errors.append(1 - ((error1[1] + error1[0]) / (test_data[0].count())))
        if total_errors[i] < 0.00005:
            break
        plot_separation(test_data, separation_input, color=str(i / iterations))

        error2 = get_error(train_data, separation_input)
        errors.append(1 - ((error2[1] + error2[0]) / (train_data[0].count())))

        weights[0].append(separation_input[0])
        weights[1].append(separation_input[1])
        weights[2].append(separation_input[2])

        train_data = train_data.sample(frac=1)

        for x, row in train_data.iterrows():
            learned_weight = calculate_new_weight(separation_input, row, hard=hard)

            separation_input[0] = learned_weight[0]
            separation_input[1] = learned_weight[1]
            separation_input[2] = learned_weight[2]

        final = separation_input

    return final


def get_error(data_matrix, separation_matrix):
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for x in data_matrix.iterrows():
        temp = x[1]
        if ((separation_matrix[0] * temp[0]) + (separation_matrix[1] * temp[1]) + separation_matrix[2]) > 0:
            if temp[2] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if temp[2] == 0:
                TN = TN + 1
            else:
                FN = FN + 1

    return TP, TN, FP, FN


if __name__ == "__main__":
    main()
