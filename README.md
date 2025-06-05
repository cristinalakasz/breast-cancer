# Breast Cancer Prediction using Machine Learning

This project implements a comprehensive machine learning solution for predicting breast cancer diagnosis using the Wisconsin Breast Cancer dataset. The project compares multiple algorithms and provides detailed analysis with clinical relevance.

## Project Structure

```
breast-cancer/
├── README.md                 # This file
├── environment.yml           # Conda environment specification
├── requirements.txt          # Alternative pip requirements (legacy)
├── setup_env.bat            # Conda environment setup script
├── activate_env.bat         # Activate environment script
├── start_jupyter.bat        # Start Jupyter Notebook script
├── data/
│   └── data.csv            # Dataset (if using external data)
└── models/                 # Saved models (created after training)
```

## Quick Start

### Prerequisites

- **Anaconda** or **Miniconda** installed on your system
- Download from: https://www.anaconda.com/products/distribution

### 1. Set Up Conda Environment

Run the setup script to create and configure the Conda environment:

```bash
# Double-click or run from command prompt
setup_env.bat
```

This will:

- Create a Conda environment named `breast_cancer_ml`
- Install all required packages from `environment.yml`
- Set up a Jupyter kernel for the project
- Configure everything needed to run the notebook

### 2. Activate Environment

To work with the project, activate the Conda environment:

```bash
# Option 1: Use the activation script
activate_env.bat

# Option 2: Manual activation
conda activate breast_cancer_ml
```

### 3. Start Jupyter Notebook

Launch Jupyter Notebook to work with the project:

```bash
# Option 1: Use the start script
start_jupyter.bat

# Option 2: Manual start (after activating environment)
jupyter notebook
```

## Environment Management

### Conda Environment Details

- **Name**: `breast_cancer_ml`
- **Python Version**: 3.9
- **Package Manager**: Conda + pip (for select packages)

### Dependencies

The project requires the following packages (automatically installed via `environment.yml`):

#### Core Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

#### Machine Learning

- **scikit-learn**: Machine learning algorithms and tools
- **joblib**: Model serialization

#### Jupyter Environment

- **jupyter**: Jupyter Notebook environment
- **ipykernel**: Jupyter kernel support
- **notebook**: Notebook interface

#### Additional Tools

- **plotly**: Interactive visualizations (via pip)
- **scipy**: Scientific computing
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **numba**: Performance optimization

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

1. Ensure Anaconda/Miniconda is installed on your system
2. Run `setup_env.bat` to create the Conda environment
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

**Issue**: Conda command not found
**Solution**: Ensure Anaconda/Miniconda is properly installed and added to PATH

**Issue**: Environment creation fails
**Solution**:

```bash
# Clean conda cache and try again
conda clean --all
conda env create -f environment.yml
```

**Issue**: Package conflicts
**Solution**:

```bash
# Remove existing environment and recreate
conda env remove -n breast_cancer_ml
conda env create -f environment.yml
```

**Issue**: Jupyter kernel not found
**Solution**: Reinstall the kernel using:

```bash
conda activate breast_cancer_ml
python -m ipykernel install --user --name=breast_cancer_ml --display-name="Breast Cancer ML"
```

**Issue**: Import errors in notebook
**Solution**: Ensure the correct kernel is selected in Jupyter (Kernel → Change Kernel → Breast Cancer ML)

### Environment Management Commands

**List all conda environments:**

```bash
conda env list
```

**Update the environment:**

```bash
conda activate breast_cancer_ml
conda env update -f environment.yml
```

**Export current environment:**

```bash
conda activate breast_cancer_ml
conda env export > environment_backup.yml
```

**Remove the environment:**

```bash
conda env remove -n breast_cancer_ml
```

**Install additional packages:**

```bash
conda activate breast_cancer_ml
conda install package_name
# or
pip install package_name
```

## Alternative Setup Methods

### Method 1: Manual Conda Setup

```bash
# Create environment manually
conda create -n breast_cancer_ml python=3.9
conda activate breast_cancer_ml
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn jupyter
pip install plotly
```

### Method 2: Using pip (if Conda not available)

```bash
# Use the legacy requirements.txt
python -m venv breast_cancer_env
breast_cancer_env\Scripts\activate
pip install -r requirements.txt
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

## Performance Optimization

### Conda vs pip advantages:

- **Faster installs**: Binary packages vs compilation
- **Better dependency resolution**: Conda handles complex dependencies
- **Cross-platform compatibility**: Consistent environments across OS
- **Scientific computing focus**: Optimized for data science workflows

## Disclaimer

This project is for educational and research purposes only. Any clinical application would require extensive validation, regulatory approval, and should always be used in conjunction with professional medical judgment.

## License

This project is provided under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the project repository.
