# README.md

# ML Pretraining App

This project is designed for pre-training machine learning models using custom datasets. It provides a structured approach to data preprocessing, model architecture definition, and training.

Install cuda toolkit
https://developer.nvidia.com/cuda-downloads

## Project Structure

```
ml-pretraining-app
├── .venv
├── src
│   ├── config
│   │   └── config.yaml
│   ├── data
│   │   └── loader.py
│   ├── models
│   │   ├── mistral_handler.py
│   │   └── mistral_trainer.py
│   ├── utils
│   │   └── helpers.py
│   └── __pycache__
├── README.md
├── requirements.txt
│── test_run.py
└── .gitignore

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

change prompt in config.yaml 
exemple:
  - instruction: "Résume ce passage du document."
  - instruction: "Extrais les informations principales de ce passage."
  - instruction: "Identifie les problèmes en te basant sur le document."
  - instruction: "Extrais les personnes et leur fonction dans le document"
  - instruction: "Extrais les mots-clés du document suivant."
  - instruction: "Extrais les acronymes et leur définition du document suivant."

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


# Copy your PDF files to the data/pdfs directory
# On Windows:
copy your_documents/*.pdf data/pdfs/


