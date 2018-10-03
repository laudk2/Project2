import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import datetime

# Average male height (US): 5.78 ft
# Average female height (US): 5.34 ft
# https://en.wikipedia.org/wiki/List_of_average_human_height_worldwide (2011 - 2014)

# Average male weight (US): 195.8 lb
# Average female weight (US): 168.4 lb
# https://en.wikipedia.org/wiki/Human_body_weight (2011 - 2014)


def main(plt):
    # Uncomment if need to generate new dataset
    # generate_data()
    create(plt, True, 0.25)
    # create(plt, True, 0.5)
    # create(plt, True, 0.75)

    # create(plt, False, 0.25)
    # create(plt, False, 0.5)
    # create(plt, False, 0.75)


def create(plt, hard, sample_fraction):
    all_data = pd.read_csv("SampleData.txt", header=None)

    # male_data = only male data [ 0 - 1999 ]
    # male_data[x] ; x = 0,1,2 same as above
    male_data = all_data[all_data[2] == 0]

    # female_data = only male data [ 2000 - 3999 ]
    # female_data[x] ; x = 0,1,2 same as above
    female_data = all_data[all_data[2] == 1]

    df = normalize_data(all_data)

    # smaller amount of random items
    train_df = df.sample(frac=sample_fraction)

    test_df = df[~df.isin(train_df)]

    plt.figure(1)
    plt = plot_male_and_females(df)
    plt.figure(2)
    plt = plot_male_and_females(df)

    rand_x = 0.1
    sep_line = [random.uniform(-rand_x, rand_x), random.uniform(-rand_x, rand_x), random.uniform(-rand_x, rand_x)]
    original_sep_line = sep_line

    plt.figure(1)
    final_sep_line = learn(train_df, test_df, sep_line, 100, hard)

    plt.figure(1)
    plot_male_and_females(df)
    plot_separation(original_sep_line, df, color="g")
    plot_separation(sep_line, df, color="b")

    plt.figure(2)
    plot_male_and_females(all_data)
    plot_separation(original_sep_line, all_data, color="g")
    plot_separation(final_sep_line, all_data, color="b")
    plt.figure(2)
    plt.axis((0, 1, 0, 1))
    plt.show()



def plot_male_and_females(data_frame):
    area = 50
    alpha = 0.1
    males = data_frame[data_frame[2] == 0]
    females = data_frame[data_frame[2] == 1]

    male_x = males[0]
    male_y = males[1]

    female_x = females[0]
    female_y = females[1]

    male_plot = plt.scatter(male_x, male_y, s=area, c=np.full(males[2].shape, 'r'), alpha=alpha)
    female_plot = plt.scatter(female_x, female_y, s=area, c=np.full(females[2].shape, 'g'), alpha=alpha)

    plt.legend((male_plot, female_plot),
               ('Male', 'Female'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    plt.title("Weight and Height for Male vs Female")
    plt.xlabel("Height (ft)")
    plt.ylabel("Weight (lbs)")

    return plt


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


def normalize_data(all_data):
    normalized_data = all_data.copy()

    normalized_data[0] = (all_data[0] - all_data[0].min()) / (all_data[0].max() - all_data[0].min())
    normalized_data[1] = (all_data[1] - all_data[1].min()) / (all_data[1].max() - all_data[1].min())

    return normalized_data


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

    return plt

# def calculate_weight_after_delta_d(current_weight, current_pattern, hard_activation=True, alpha=alpha, k=0.5):


def calculate_weight_after_delta_d(current_weight, current_pattern, hard_activation=True):
    alpha = 0.1
    k = 0.5

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


errors = []
total_errors = []
weights = [[], [], []]


def learn(train_df, test_df, sep_line, number_of_iterations, hard):
    epsilon = 0.00005
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

        if epsilon > total_error:
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
    main(plt)
