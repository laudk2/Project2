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
    # begin_learning(True, 0.25)
    begin_learning(True, 0.75)

    # begin_learning(False, 0.25)
    # begin_learning(False, 0.75)


# *******************
# Function to set up and begin learning process
# *******************
def begin_learning(hard, sample_fraction):

    normalized_data = normalize_data(pd.read_csv("SampleData.txt", header=None))

    training_data = normalized_data.sample(frac=sample_fraction)

    testing_data = normalized_data[~normalized_data.isin(training_data)]

    # random_separation = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    random_separation = [1, -1, 0.5]

    original_separation = random_separation

    plt.figure(1)
    final_separation = learn(training_data, testing_data, random_separation, 10, hard)
    plot_male_and_females(normalized_data)
    plot_separation(original_separation, normalized_data, color="green")
    plot_separation(random_separation, normalized_data, color="blue")
    plt.show()

    plt.figure(2)
    plot_male_and_females(normalized_data)
    plot_separation(final_separation, normalized_data, color="blue")
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


def plot_male_and_females(data_frame):
    males = data_frame[data_frame[2] == 0]
    females = data_frame[data_frame[2] == 1]

    male_x = males[0]
    male_y = males[1]

    female_x = females[0]
    female_y = females[1]

    male = plt.scatter(male_x, male_y, s=50, c=np.full(males[2].shape, 'red'), alpha=0.1)
    female = plt.scatter(female_x, female_y, s=50, c=np.full(females[2].shape, 'green'), alpha=0.1)

    plt.legend((male, female), ('Male', 'Female'), scatterpoints=1, loc='lower left', ncol=3, fontsize=6)

    plt.title("Male & Female - Weight vs Height")
    plt.xlabel("Height (ft)")
    plt.ylabel("Weight (lbs)")

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




def plot_separation(separation_line, data, color="0.18"):
    weight_x = separation_line[0]
    weight_y = separation_line[1]
    bias = separation_line[2]

    center = (((data[0].max() - data[0].min()) / 2) + data[0].min())

    # Formula: y_weight = x_weight + bias
    # Y = (x _weight / a) * y_weight + (bias / y_weight)
    y1 = -(((weight_x * data[0].min()) / weight_y) + (bias / weight_y))
    y2 = -(((weight_x * data[0].max()) / weight_y) + (bias / weight_y))
    y_mid = -(((weight_x * center) / weight_y) + (bias / weight_y))

    ax = plt.axes()
    ax.arrow(center, y_mid, 0.05, 0.05, head_width=0.025, head_length=0.025, fc='k', ec='k', color="b")

    plt.plot([data[0].min(), data[0].max()], [y1, y2], color=color)
    plt.axis((0,1,0,1))
    return plt


def calculate_weight_after_delta_d(current_weight, current_pattern, hard_activation=True, alpha=alpha, k=0.5):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1] * current_pattern[1] +
           current_weight[2])

    if hard_activation:
        output = 1 if net > 0 else 0
        delta_d = alpha * (current_pattern[2] - output)
    else:
        output = (np.tanh(net * k) + 1) / 2
        delta_d = alpha * (current_pattern[2] - output)

    current_pattern[0] *= delta_d
    current_pattern[1] *= delta_d
    current_pattern[2] = delta_d

    current_weight[0] += current_pattern[0]
    current_weight[1] += current_pattern[1]
    current_weight[2] += current_pattern[2]

    return current_weight


def learn(train_df, test_df, sep_line, number_of_iterations, hard):
    final_sep_line = None

    for i in range(0, number_of_iterations):
        print("iteration", i)

        total_error = calculate_error(test_df, sep_line)
        total_errors.append(total_error)

        plot_separation(sep_line, test_df, color=str(i / number_of_iterations))

        err = calculate_error(train_df, sep_line)
        errors.append(err)

        weights[0].append(sep_line[0])
        weights[1].append(sep_line[1])
        weights[2].append(sep_line[2])

        if total_error < 0.00005:
            break

        # mix up the test data frame so that we learn in different ways(?)
        train_df = train_df.sample(frac=1)
        # For each element in the data_frame `train_df`
        for index, row in train_df.iterrows():
            new_weights = calculate_weight_after_delta_d(sep_line, row, hard_activation=hard)

            sep_line[0] = new_weights[0]
            sep_line[1] = new_weights[1]
            sep_line[2] = new_weights[2]

        final_sep_line = sep_line

    return final_sep_line


def calculate_error(data_frame, sep_line):
    error_matrix = get_confusion_matrix(data_frame, sep_line)
    return 1 - ((error_matrix[1] + error_matrix[0]) / (data_frame[0].count()))


def get_output_for_row(current_pattern, current_weight):
    net = (current_weight[0] * current_pattern[0] +
           current_weight[1] * current_pattern[1] +
           current_weight[2])

    return 1 if net > 0 else 0


def get_confusion_matrix(data_frame, sep_line):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for row in data_frame.iterrows():
        r = row[1]

        if len(sep_line) == 3:
            gender = r[2]

            if get_output_for_row(r, sep_line):
                if gender == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if gender == 0:
                    true_negative += 1
                else:
                    false_negative += 1
        else:
            height = r[0]
            weight = r[1]
            gender = r[2]
            x_weight = sep_line[0]
            bias = sep_line[1]

            # 0 <= bx - c
            net = x_weight * height - bias * 1

            if net < 0:
                if gender == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if gender == 0:
                    true_negative += 1
                else:
                    false_negative += 1

    return (true_positive,
            true_negative,
            false_positive,
            false_negative)


if __name__ == "__main__":
    main()
