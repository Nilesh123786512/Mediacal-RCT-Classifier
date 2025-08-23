# 🔬 Medical RCT Abstract Classifier (Skimlit)

This Streamlit application leverages a deep learning model to classify sentences in medical abstracts from Randomized Controlled Trials (RCTs). By identifying the role of each sentence (e.g., **Objective**, **Methods**, **Results**, **Conclusions**), it helps researchers and students read and understand scientific papers more quickly and efficiently.

The model is a "tribrid" implementation combining token, character, and positional embeddings to achieve high accuracy.


## ✨ Features

-   **Sentence Classification**: Automatically labels each sentence into one of five categories: `OBJECTIVE`, `BACKGROUND`, `METHODS`, `RESULTS`, or `CONCLUSIONS`.
-   **Structured Output**: Groups classified sentences under their respective labels in a clean, readable Markdown format.
-   **Confidence Scores**: Displays the model's confidence for each classification, giving you an insight into the prediction's reliability.
-   **Intelligent Text Handling**: Properly splits text into sentences, even when numbers with decimals are present.
-   **Built with TensorFlow & Streamlit**: Uses a powerful TensorFlow/Keras model with a user-friendly web interface created with Streamlit.

## 🛠️ Setup and Installation

To run this application on your local machine, please follow these steps.

### Prerequisites

-   Python 3.9+
-   `pip` (Python package installer)

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/Nilesh123786512/Mediacal-RCT-Classifier
cd MEDIACAL-RCT-CLASSIFIER
```

### 2. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Model File

This application requires the pre-trained Keras model file (`skimlit_tribrid_200k.keras`), which should be in the root directory of this project.

## 🚀 How to Run the App

Once the setup is complete, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default web browser.

## 📁 Project Structure

```.
├── app.py                      # The main Streamlit application script
├── experiments.ipynb           # Jupyter notebook with model training and experimentation
├── LICENSE                     # Project license file
├── README.md                   # This file
├── requirements.txt            # List of Python dependencies
└── skimlit_tribrid_200k.keras  # The pre-trained deep learning model
```

##  acknowledgments and Citations

This work is an implementation based on the architectures and datasets described in the following papers:

1.  **Dataset Paper**:
    Franck Dernoncourt, and Ji Young Lee. ["PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts."](https://arxiv.org/abs/1710.06071) *arXiv preprint arXiv:1710.06071* (2017).

2.  **Implementation Paper**:
    Jason P. D. Walsh, and Szilárd Z. Kiss. ["Neural Networks for Joint Sentence Classification in Medical Paper Abstracts."](https://arxiv.org/pdf/1612.05251) *arXiv preprint arXiv:1612.05251* (2016).