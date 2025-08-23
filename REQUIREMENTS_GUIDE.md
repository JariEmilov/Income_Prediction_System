# Requirements Files Documentation

## 📋 Overview
This project includes comprehensive dependency management with multiple configuration approaches:

### 1. `requirements.txt` - **Frozen Production Environment**
- **Purpose**: Exact snapshot of current working environment
- **Versioning**: Pinned exact versions (==) for perfect reproducibility
- **Generated from**: `uv pip list` output with all 291 packages
- **Usage**: `pip install -r requirements.txt` or `uv pip install -r requirements.txt`

### 2. `pyproject.toml` - **Modern Python Project Configuration**
- **Purpose**: Main project configuration with flexible dependency management
- **Versioning**: Uses minimum version constraints (>=) for flexibility
- **Organization**: Core packages only (28 primary dependencies)
- **Usage**: `uv sync` or `pip install -e .`

### 3. `03_production_system/pyproject.toml` - **AI Agent Sub-Module**
- **Purpose**: Dedicated configuration for the CrewAI income prediction agent
- **Versioning**: Minimal dependencies specific to the agent system
- **Usage**: For isolated agent development and deployment

## 🏗️ Project Structure & Dependencies

### Main Project Dependencies (`pyproject.toml`):

**Core Data Science & ML Libraries:**
- pandas, numpy, scikit-learn, scipy, statsmodels, imbalanced-learn

**Machine Learning Models:**
- xgboost, lightgbm

**Data Visualization:**
- matplotlib, seaborn, plotly, sweetviz

**Jupyter & Analysis Environment:**
- jupyter, ipykernel

**Web Framework:**
- streamlit (for the AI agent interface)

**AI/ML Agents & Tools:**
- crewai[tools] (with extended tool support)

**Data Processing & Utilities:**
- category-encoders, openpyxl, ucimlrepo

### AI Agent Sub-Module (`03_production_system/pyproject.toml`):

**Minimal Agent Dependencies:**
- crewai (core agent framework)
- pandas, numpy, scikit-learn (data processing)
- xgboost (ML model inference)
- joblib (model loading)
- python-dotenv (environment management)

### Complete Environment (`requirements.txt`):
- **Total packages**: 291 (all dependencies with exact versions)
- **Primary packages**: 28 (defined in pyproject.toml)
- **Transitive dependencies**: 263 (automatically resolved)

## 🔧 Installation Instructions

### For New Environment Setup (Recommended - UV):
```bash
# Clone/extract the project
cd "Data-Scientist Test"

# Install with UV (faster and more reliable)
uv sync

# Activate environment (if needed)
# UV automatically manages virtual environments
```

### For New Environment Setup (Standard pip):
```bash
# Create new virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install with exact frozen requirements
pip install -r requirements.txt
```

### For Development with Flexible Versions:
```bash
# Install from pyproject.toml (flexible versions)
pip install -e .

# Or with UV
uv pip install -e .
```

### For AI Agent Sub-Module Only:
```bash
cd 03_production_system

# Install minimal agent dependencies
pip install -e .

# Or with UV
uv pip install -e .
```

## ⚠️ Known Requirements Notes

### Python Version Compatibility:
- **Main project**: Requires Python >=3.13 (as specified in pyproject.toml)
- **AI Agent**: Requires Python >=3.10,<3.14 (more flexible for agent deployment)
- **Current environment**: Python 3.13.x (confirmed working)

### Key Package Versions (Frozen):
- **CrewAI**: 0.165.1 (AI agent framework)
- **XGBoost**: 3.0.4 (ML model)
- **Streamlit**: 1.48.1 (web interface)
- **Pandas**: 2.2.3 (data processing)
- **Scikit-learn**: 1.6.2 (ML utilities)

### UV Lock File:
- `uv.lock` contains the complete dependency resolution
- Ensures consistent installs across different environments
- Automatically managed by UV package manager

## 🚀 Usage in Different Scenarios

### 1. **Complete Development Environment**
```bash
# Full environment with all 291 packages
pip install -r requirements.txt
```
Use for complete data science workflow including notebooks, visualization, and AI agents.

### 2. **Flexible Development** 
```bash
# Install from pyproject.toml with latest compatible versions
uv sync
```
Use for ongoing development where you want automatic updates to compatible versions.

### 3. **Production Deployment**
```bash
# Exact versions for reproducible deployment
pip install -r requirements.txt
```
Use for production where you need exact reproducibility.

### 4. **AI Agent Only**
```bash
cd 03_production_system
pip install -e .
```
Use for minimal deployment of just the income prediction agent.

### 5. **CI/CD Pipelines**
```bash
# Use UV for faster, more reliable builds
uv sync --frozen
```
Use UV with frozen lock file for consistent CI/CD builds.

### 6. **Docker-Based Deployment**
```bash
# Build and run the project in a container
docker build -t income-prediction-system .
docker run -p 8501:8501 income-prediction-system
```
Use for quick, reproducible setup on any machine with Docker installed.

## 📦 Project Structure Summary

```
📁 Data-Scientist Test/
├── 📋 requirements.txt                 # Frozen environment (291 packages)
├── 📋 pyproject.toml                   # Main project config (28 core deps)
├── 🔒 uv.lock                         # UV dependency resolution
├── 📊 01_process_and_clean_data/       # Data analysis notebooks
├── 🤖 02_ml_development/               # ML model development
├── 🎯 03_production_system/            # AI agent system
│   ├── 📋 pyproject.toml              # Agent-specific config (7 core deps)
│   ├── 📱 streamlit_app.py            # Web interface
│   └── 🤖 src/income_prediction_agent/ # CrewAI agent code
├── 🗃️ models/                         # Trained ML models
└── 📈 training_data/                   # Processed datasets
```

## 📊 Package Statistics
- **Total installed packages**: 291 (with all dependencies)
- **Main project core packages**: 28 (in root pyproject.toml)
- **AI agent core packages**: 7 (in 03_production_system/pyproject.toml)
- **Transitive dependencies**: 256 (automatically resolved)

This comprehensive dependency setup ensures your income prediction project has everything needed for:
- ✅ Statistical data analysis and visualization
- ✅ Machine learning model development and optimization
- ✅ AI agent functionality with CrewAI and RAG
- ✅ Professional Streamlit web application
- ✅ Jupyter notebook development environment
- ✅ Production deployment with exact reproducibility
