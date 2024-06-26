import pandas as pd

class CSVDataLoader:
    def __init__(self, file_path, input_size, output_size):
        """
        Initialize the CSVDataLoader class
        :param file_path: Path to the CSV file
        :param input_size: Number of input features
        :param output_size: Number of output features
        """
        self.file_path = file_path
        self.input_size = input_size
        self.output_size = output_size
        self.data = None
        self.x_data = None
        self.y_data = None

    def load_and_check_data(self):
        """
        Load the CSV file and check for missing values and invalid dimensions
        """

        # Load CSV file
        self.data = pd.read_csv(self.file_path)

        # Check for missing values
        if self.data.isnull().values.any():
            raise ValueError("Data contains missing values")

        # Check dimensions
        if self.data.shape[1] != (self.input_size + self.output_size):
            raise ValueError("Data dimensions do not match the specified input and output sizes")

    def prepare_data(self):
        """
        Prepare the data for training
        :return: A tuple containing the 2 data arrays
        """
        self.x_data = self.data.iloc[:, :self.input_size].values
        self.y_data = self.data.iloc[:, self.input_size:].values

        # Reshape x_data to match the format used in cli()
        self.x_data = self.x_data.reshape(-1, 1, self.input_size)

        return self.x_data, self.y_data
