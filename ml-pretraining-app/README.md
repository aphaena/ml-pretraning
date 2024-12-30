# README.md

# ML Pretraining App

This project is designed for pre-training machine learning models using custom datasets. It provides a structured approach to data preprocessing, model architecture definition, and training.

## Project Structure

```
ml-pretraining-app
├── src
│   ├── data
│   │   ├── preprocessing.py
│   │   └── loader.py
│   ├── models
│   │   └── model.py
│   ├── config
│   │   └── config.yaml
│   ├── utils
│   │   └── helpers.py
│   └── train.py
├── tests
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-pretraining-app
   ```



2. Install the required dependencies:

    Create virtual environment
    python -m venv .venv

   Activate virtual environment (Windows)   
   .\.venv\Scripts\activate
   
   pip install -r requirements.txt
   

3. Configure your dataset paths and hyperparameters in `src/config/config.yaml`.

## Usage

To start the training process, run the following command:
```
python test_run.py
```

## Overview

This application includes:
- Data preprocessing functions for normalization and augmentation.
- A `DataLoader` class for loading datasets.
- A customizable model architecture.
- Utility functions for logging and metrics.
- Unit tests to ensure the correctness of preprocessing steps.

Feel free to contribute to the project or reach out for any questions!