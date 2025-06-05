# Breast Cancer Prediction using Machine Learning

This project implements a comprehensive machine learning solution for predicting breast cancer diagnosis using the Wisconsin Breast Cancer dataset. The project compares multiple algorithms and provides detailed analysis with clinical relevance.

## Project Structure

```
breast-cancer/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_env.bat            # Virtual environment setup script
├── activate_env.bat         # Activate environment script
├── start_jupyter.bat        # Start Jupyter Notebook script
├── data/
│   └── data.csv            # Dataset (if using external data)
└── breast_cancer_env/      # Virtual environment (created after setup)
```

## Quick Start

### 1. Set Up Virtual Environment

Run the setup script to create and configure the virtual environment:

```bash
# Double-click or run from command prompt
setup_env.bat
```

This will:

- Create a Python virtual environment named `breast_cancer_env`
- Install all required packages from `requirements.txt`
- Set up a Jupyter kernel for the project
- Configure everything needed to run the notebook

### 2. Activate Environment

To work with the project, activate the virtual environment:

```bash
# Option 1: Use the activation script
activate_env.bat

# Option 2: Manual activation
breast_cancer_env\Scripts\activate.bat
```

### 3. Start Jupyter Notebook

Launch Jupyter Notebook to work with the project:

```bash
# Option 1: Use the start script
start_jupyter.bat

# Option 2: Manual start (after activating environment)
jupyter notebook
```

## Dependencies

The project requires the following Python packages:

### Core Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### Machine Learning

- **scikit-learn**: Machine learning algorithms and tools
- **joblib**: Model serialization

### Jupyter Environment

- **jupyter**: Jupyter Notebook environment
- **ipykernel**: Jupyter kernel support
- **notebook**: Notebook interface

### Additional Tools

- **plotly**: Interactive visualizations
- **scipy**: Scientific computing
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## Project Features

### 1. Comprehensive Data Analysis

- Exploratory data analysis with visualizations
- Feature correlation analysis
- Distribution analysis between malignant and benign cases

### 2. Advanced Feature Engineering

- Creation of ratio features
- Polynomial feature generation
- Feature scaling and normalization

### 3. Multiple ML Algorithms

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Gradient Boosting
- Neural Networks (MLP)

### 4. Model Optimization

- Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Cross-validation for robust evaluation
- Feature selection techniques

### 5. Clinical Evaluation

- Comprehensive performance metrics
- Confusion matrices and ROC curves
- Clinical significance analysis
- Sensitivity, specificity, and predictive values

## Usage Instructions

### Step 1: Environment Setup

1. Ensure Python 3.7+ is installed on your system
2. Run `setup_env.bat` to create the virtual environment
3. Wait for all packages to install (this may take a few minutes)

### Step 2: Working with the Project

1. Run `activate_env.bat` to activate the environment
2. Run `start_jupyter.bat` to launch Jupyter Notebook
3. Open the notebook file in Jupyter
4. Run cells sequentially or all at once

### Step 3: Customization

- Modify hyperparameters in the optimization sections
- Add new algorithms or feature engineering techniques
- Adjust visualization parameters for better insights

## Troubleshooting

### Common Issues

**Issue**: Virtual environment creation fails
**Solution**: Ensure Python is properly installed and added to PATH

**Issue**: Package installation errors
**Solution**: Upgrade pip using `python -m pip install --upgrade pip`

**Issue**: Jupyter kernel not found
**Solution**: Reinstall the kernel using:

```bash
python -m ipykernel install --user --name=breast_cancer_env --display-name="Breast Cancer ML"
```

**Issue**: Import errors in notebook
**Solution**: Ensure the correct kernel is selected in Jupyter (Kernel → Change Kernel → Breast Cancer ML)

### Environment Management

To update packages:

```bash
# Activate environment first
activate_env.bat

# Update specific package
pip install --upgrade package_name

# Update all packages
pip install --upgrade -r requirements.txt
```

To deactivate the environment:

```bash
deactivate
```

To remove the environment:

```bash
# Delete the breast_cancer_env folder
rmdir /s breast_cancer_env
```

## Results Expected

The notebook will provide:

- **Model Accuracy**: >95% classification accuracy
- **Clinical Metrics**: High sensitivity and specificity
- **Feature Insights**: Identification of most predictive features
- **Comparative Analysis**: Performance comparison across algorithms
- **Optimized Models**: Hyperparameter-tuned best performers

## Clinical Relevance

This project demonstrates:

- **Diagnostic Support**: Potential to assist medical professionals
- **Risk Assessment**: Identification of high-risk cases
- **Quality Assurance**: Second-opinion validation system
- **Screening Enhancement**: Improved screening program effectiveness

## Disclaimer

This project is for educational and research purposes only. Any clinical application would require extensive validation, regulatory approval, and should always be used in conjunction with professional medical judgment.

## License

This project is provided under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the project repository.
