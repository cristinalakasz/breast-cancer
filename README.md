# Breast Cancer Prediction System

This project is a machine learning-based GUI application for predicting breast cancer diagnosis (Benign or Malignant) based on 30 features extracted from cell nuclei images.

## Features

- **Input Fields**: Allows users to input values for 30 features manually.
- **Prediction**: Uses a trained machine learning model to predict whether the case is benign or malignant.
- **Sample Data**: Load sample benign or malignant data for testing.
- **Results Display**: Shows prediction results, confidence levels, and recommendations.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `tkinter`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cristinalakasz/breast-cancer.git
   cd breast-cancer
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Setting Up the Environment with Conda

1. Install Anaconda or Miniconda:

   - Download from: https://www.anaconda.com/products/distribution

2. Create the environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate breast_cancer_env
   ```

4. Ensure the trained model file (`breast_cancer_model.pkl`) is present in the project directory.

## Running the Application

1. Run the GUI application:
   ```bash
   python gui.py
   ```

## Running the Notebook

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the notebook file (`breast_cancer_prediction.ipynb`) from the project directory.

3. Run the cells sequentially to:
   - Load the trained model.
   - Input feature values manually or load sample data.
   - Perform predictions and view results.

## Using the Application

### Input Fields

- Enter values for all 30 features manually in the input fields.
- Feature descriptions are provided next to each field for better understanding.

### Sample Data

- Click the **Load Sample Data** button to load predefined benign or malignant sample data.
- You will be prompted to choose between benign and malignant samples.

### Prediction

- Click the **Predict** button after entering the values or loading sample data.
- The results will display:
  - Diagnosis (Benign or Malignant)
  - Confidence level
  - Probability of malignancy and benignity
  - Recommendations

### Clearing Fields

- Click the **Clear All** button to reset all input fields.

## Important Notes

- This tool is for educational purposes only and should not replace professional medical diagnosis.
- Always consult with qualified healthcare professionals for accurate diagnosis and treatment.

## Disclaimer

- The model has an accuracy of approximately 97% on test data.
- False negatives and false positives are possible.

## Project Details

This project leverages machine learning to predict breast cancer diagnosis using features extracted from cell nuclei images. The GUI application simplifies the process by allowing users to input data, load sample cases, and view predictions with confidence levels and recommendations.

The trained model (`breast_cancer_model.pkl`) is based on the Wisconsin Breast Cancer dataset and achieves approximately 97% accuracy on test data. The project demonstrates the potential of machine learning in assisting medical professionals with diagnostic support.

## Project Structure

```
breast-cancer/
├── README.md                 # Project documentation
├── gui.py                    # Main GUI application
├── breast_cancer_model.pkl   # Trained model file
├── requirements.txt          # Python dependencies
└── data/                     # Directory containing data used for model training
```

- **README.md**: Contains project documentation and usage instructions.
- **gui.py**: Implements the GUI application for breast cancer prediction.
- **breast_cancer_model.pkl**: Serialized machine learning model used for predictions.
- **requirements.txt**: Lists required Python libraries for the project.
- **data/**: Contains the dataset used for training the machine learning model.
