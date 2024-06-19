import os
import pandas as pd
import numpy as np


def generate_synthetic_data(num_samples_train, num_samples_test, input_size, activation_function):
    np.random.seed(0)

    # Generate random data for train
    x_train = np.random.rand(num_samples_train, input_size)
    y_train = np.sum(x_train, axis=1).reshape(-1, 1)

    # Use the last `num_samples_test` entries from the training set for the test set
    x_test = x_train[-num_samples_test:]
    y_test = y_train[-num_samples_test:]

    # Create DataFrame and save to CSV
    train_data = pd.DataFrame(np.hstack((x_train, y_train)),
                              columns=[f'Input{i + 1}' for i in range(input_size)] + ['Output'])
    test_data = pd.DataFrame(x_test, columns=[f'Input{i + 1}' for i in range(input_size)])

    # Save to CSV files
    train_filename = f'train_data_{activation_function}.csv'
    test_filename = f'test_data_{activation_function}.csv'

    train_data.to_csv(train_filename, index=False)
    test_data.to_csv(test_filename, index=False)

    return train_filename, test_filename, y_test


def main():
    output_dir = "../tests/datasets"
    os.makedirs(output_dir, exist_ok=True)

    activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    num_samples_train = 100
    num_samples_test = 10
    input_size = 5

    results = {}

    for activation_function in activation_functions:
        train_filename, test_filename, expected_results = generate_synthetic_data(
            num_samples_train, num_samples_test, input_size, activation_function)

        # Move files to output directory
        os.rename(train_filename, os.path.join(output_dir, train_filename))
        os.rename(test_filename, os.path.join(output_dir, test_filename))

        results[activation_function] = expected_results

    # Display the expected results for the test data
    for activation_function, expected in results.items():
        print(f"Expected results for {activation_function} activation:")
        print(expected.flatten()[:5])  # Show only first 5 results for brevity


if __name__ == "__main__":
    main()
