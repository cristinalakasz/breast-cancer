@echo off
echo Creating virtual environment for Breast Cancer ML Project...

REM Create virtual environment
python -m venv breast_cancer_env

REM Activate virtual environment
call breast_cancer_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Create Jupyter kernel
python -m ipykernel install --user --name=breast_cancer_env --display-name="Breast Cancer ML"

echo.
echo Virtual environment setup complete!
echo.
echo To activate the environment, run: breast_cancer_env\Scripts\activate.bat
echo To deactivate, run: deactivate
echo To start Jupyter Notebook, run: jupyter notebook
echo.
pause