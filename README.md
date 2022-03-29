[![Python application](https://github.com/fleuryc/Template-Python/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/Template-Python/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/Template-Python/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/Template-Python/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/b03fbc514ea44fce83fe471896566cfd)](https://www.codacy.com/gh/fleuryc/Template-Python/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/b03fbc514ea44fce83fe471896566cfd)](https://www.codacy.com/gh/fleuryc/Template-Python/dashboard)

- [Project](#project)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

* * *

# Project

-   What ?
-   Why ?
-   How ?

## Installation

### Prerequisites

-   [Python 3.9](https://www.python.org/downloads/)

### Virtual environment

```bash
# python -m venv env
# > or just :
make venv
source env/bin/activate
```

### Dependencies

```bash
# pip install kaggle jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests matplotlib seaborn plotly shap numpy statsmodels pandas sklearn nltk gensim pyLDAvis spacy transformers tensorflow
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

### Environment variables

- Set environment variable values in [.env](.env) file.


## Usage

### Download data

Download, extract and upload to Azure Cityscape zip files.

```bash
make dataset
```

### Run Notebook

```bash
jupyter-lab notebooks/main.ipynb
```

### Quality Assurance

```bash
# make isort
# make format
# make lint
# make bandit
# make mypy
# make test
# > or just :
make qa
```

## Troubleshooting

-   Fix Plotly issues with JupyterLab

cf. [Plotly troubleshooting](https://plotly.com/python/troubleshooting/#jupyterlab-problems)

```bash
jupyter labextension install jupyterlab-plotly
```

-   If using Jupyter Notebook instead of JupyterLab, uncomment the following lines in the notebook

```python
import plotly.io as pio
pio.renderers.default='notebook'
```
