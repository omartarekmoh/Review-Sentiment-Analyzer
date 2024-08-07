# Review Sentiment Prediction

This repository contains a Review Sentiment Analyzer that predicts the sentiment of text based on the Yelp dataset. The project is implemented using PyTorch and includes the following components:

- `Vocabulary`: A class for managing the mapping between tokens and their corresponding integer indices.
- `ReviewVectorizer`: A class for converting text reviews into numerical vectors.
- `ReviewDataset`: A class for handling the dataset and splitting it into training, validation, and test sets.
- `ReviewClassifier`: A PyTorch neural network model for sentiment classification.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/omartarekmoh/Review-Sentiment-Prediction.git
    cd Review-Sentiment-Prediction
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preparation:**

    Download the Yelp dataset from [here](https://www.yelp.com/dataset) and place it in the `data/` directory -The one in the directory is a smaller version of the full dataset-.

2. **Training the Model:**

    Run the training script to train the model on the dataset:

    ```bash
    python train.py
    ```

3. **Making Predictions:**

    Use the trained model to predict the sentiment of new reviews:

    ```bash
    python test.py
    ```

## Project Structure

- `data/`: Directory for storing the dataset.
- `train.py`: Script for training the model.
- `test.py`: Script for making predictions using the trained model.
- `classes.py`: Contains the core classes (`Vocabulary`, `ReviewVectorizer`, `ReviewDataset`).
- `model.py`: Contains the `ReviewClassifier` class.
- `helpers.py`: Contains utility functions for data processing and model training.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
