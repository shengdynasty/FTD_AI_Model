# Team Setup (Windows/macOS/Linux)

## 1) Clone the repo
- GitHub Desktop: File → Clone Repository
- or CLI:
  git clone <REPO_URL>
  cd <REPO_FOLDER>

## 2) Create + activate a virtual environment
### Windows (PowerShell)
run ts in powershell:
python -m venv .venv
.venv\Scripts\activate

## 3) Install dependencies
run ts in vs code terminal: 
pip install -r requirements.txt

## 4) Generate synthetic data
run ts in terminal: 
python generate_data.py
# creates cell_data.csv locally (not real data)

## 5) Train + evaluate model (with SHAP plots)
run the main file: 
python main_model.py  
#creates the plot 
