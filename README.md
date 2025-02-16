# ScratchML
My understanding and implementation of ML algorithms that I would write from scratch in Python.

Following is how the code will be structured:
Scratch ML/
│
├── data/                     # Raw and processed data files
│   ├── raw/                  # Unprocessed data (often from external sources)
│   ├── processed/            # Cleaned/processed data (ready for modeling)
│   ├── external/             # External datasets (e.g., pre-trained models, API data)
│   └── test/                 # Test data or small datasets used for testing
│
├── notebooks/                # Jupyter notebooks for exploration and prototyping
│   ├── 01_exploration.ipynb  # Data exploration
│   ├── 02_modeling.ipynb     # Model training and validation
|   └── 03_testing.ipynb      # Random testing used as a scratchpad
│
├── src/                      # Core source code for all algorithms
│   ├── __init__.py           # To create a python package out of it
│   ├── data_preprocessing/   # Scripts for data processing
│   │   ├── clean_data.py     # Data cleaning functions
│   │   ├── feature_engineering.py # Feature extraction and selection
│   │   └── data_splitter.py  # Splitting the data into train/test sets
│   ├── models/               # Scripts for model implementation
│   │   ├── base_model.py     # Base class for models
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── xgboost.py
│   │   ├── neural_network.py
|   |   └── etc...
│   ├── utils/                # Helper functions and utilities
│   │   ├── metrics.py        # Functions for performance metrics (e.g., accuracy, precision)
│   │   ├── logger.py         # Custom logger for tracking experiments
│   │   └── config.py         # Configuration file for parameters and constants
│   └── pipeline/             # Pipeline code for training and inference
│       ├── training_pipeline.py
│       └── inference_pipeline.py
│
├── tests/                    # Unit and integration tests
│   ├── test_data_preprocessing.py
│   ├── test_models.py
│   └── test_utils.py
│
├── scripts/                  # Scripts for running experiments or batch jobs
│   ├── train_model.py        # Script to train a model
│   └── evaluate_model.py     # Script to evaluate a trained model
│
├── requirements.txt          # Dependencies (use `pip freeze > requirements.txt`)
├── setup.py                  # Installation script
├── config/                   # Configuration files (e.g., for hyperparameters)
│   └── model_config.yaml     # YAML/JSON files for model configuration
└── logs/                     # Logs (for experiments, training, etc.)
    └── training.log          # Log of training process
