[![Python application](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ec70c0d336f545b2ab13682841ac44ef)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ec70c0d336f545b2ab13682841ac44ef)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/dashboard)

- [Project](#project)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

---

# My Content : Books recommandation mobile app

Repository of OpenClassrooms' [AI Engineer path](https://openclassrooms.com/fr/paths/188-ingenieur-ia), project #8

Goal : use _Azure Machine Learning_ and _Azure Functions_ services, a _Recommander system_  embedded in a *React-Native* mobile app to produce the MVP of a books recommandation mobile app.

You can see the results here :

- [Presentation](https://fleuryc.github.io/OC_AI-Engineer_P9_Books-recommandation-mobile-app/index.html "Presentation")
- [Notebook : HTML page with interactive plots](https://fleuryc.github.io/OC_AI-Engineer_P9_Books-recommandation-mobile-app/notebook.html "HTML page with interactive plots")
- [Mobile App](https://www.clementflery.me "Mobile App")

## Goals

- [ ] create a first *Recommder System* based on **Content-Based Filtering**
- [ ] improve the *Recommder System* with **Collaborative Filtering**
- [ ] create the *React-Native* mobile app
- [ ] integrate the *Recommder System* in *Azure Functions*

## Installation

### Prerequisites

- [Python 3.9](https://www.python.org/downloads/)

### Virtual environment

```bash
# python -m venv env
# > or just :
make venv
source env/bin/activate
```

### Dependencies

```bash
# pip install kaggle jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests matplotlib seaborn plotly numpy statsmodels pandas sklearn transformers tensorflow
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

### Environment variables

- Set environment variable values in [.env](.env) file.

### Azure resources

- AzureML [Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace#-create-a-workspace "Create a workspace")
- ...

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

- Fix Plotly issues with JupyterLab

cf. [Plotly troubleshooting](https://plotly.com/python/troubleshooting/#jupyterlab-problems)

```bash
jupyter labextension install jupyterlab-plotly
```

- If using Jupyter Notebook instead of JupyterLab, uncomment the following lines in the notebook

```python
import plotly.io as pio
pio.renderers.default='notebook'
```
