@echo off
echo Creating Conda environment for Breast Cancer ML Project...

REM Check if conda is installed
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda is not installed or not in PATH.
    echo Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

REM Create conda environment from environment.yml
echo Creating environment from environment.yml...
conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment.
    echo Please check the environment.yml file and try again.
    pause
    exit /b 1
)

REM Activate the environment and set up Jupyter kernel
echo Activating environment and setting up Jupyter kernel...
call conda activate breast_cancer_ml
python -m ipykernel install --user --name=breast_cancer_ml --display-name="Breast Cancer ML"

echo.
echo Conda environment setup complete!
echo.
echo To activate the environment, run: conda activate breast_cancer_ml
echo To deactivate, run: conda deactivate
echo To start Jupyter Notebook, run: jupyter notebook
echo.
echo Environment name: breast_cancer_ml
echo.
pause