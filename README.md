[![Python application](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/actions/workflows/codeql-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ec70c0d336f545b2ab13682841ac44ef)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ec70c0d336f545b2ab13682841ac44ef)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app/dashboard)

- [My Content : Books recommandation mobile app](#my-content--books-recommandation-mobile-app)
  - [Goals](#goals)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
    - [Environment variables](#environment-variables)
    - [Azure resources](#azure-resources)
  - [Usage](#usage)
    - [Download data](#download-data)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

---

# My Content : Books recommandation mobile app

Repository of OpenClassrooms' [AI Engineer path](https://openclassrooms.com/fr/paths/188-ingenieur-ia), project #9

Goal : use _Azure Machine Learning_ and _Azure Functions_ services, a _Recommander system_ embedded in a _React-Native_ mobile app to produce the MVP of a books recommandation mobile app.

You can see the results here :

- [Presentation](https://fleuryc.github.io/OC_AI-Engineer_P9_Books-recommandation-mobile-app/index.html "Presentation")
- [Notebook : HTML page with interactive plots](https://fleuryc.github.io/OC_AI-Engineer_P9_Books-recommandation-mobile-app/main.html "HTML page with interactive plots")

This repository is part of a 3-repos project :

- Main repo : [My Content : Books recommandation mobile app](https://github.com/fleuryc/OC_AI-Engineer_P9_Books-recommandation-mobile-app) : **this repo**
- [Azure Function](https://github.com/fleuryc/oc_p9_function "Azure Function")
- [Mobile App](https://github.com/fleuryc/oc_p9_mobile-app "Mobile App")

![Current MVP architecture](docs/img/current-architecture.svg "Current MVP architecture")

## Goals

- [x] create a first _Recommder System_ based on **Content-Based Filtering**
  - [Main notebook (EDA)](notebooks/main.ipynb "Exploratory Data Analysis")
  - [Content-Based Filtering](notebooks/content-based-filtering.ipynb "Content-Based Filtering")
- [x] improve the _Recommder System_ with **Collaborative Filtering**
  - [Collaborative Filtering with Surprise](notebooks/surprise-collaborative-filtering.ipynb "Collaborative Filtering with Surprise")
  - [Collaborative Filtering with Implicit](notebooks/implicit-collaborative-filtering.ipynb "Collaborative Filtering with Implicit")
- [x] improve the _Recommder System_ with **Hybrid Filtering**
  - [Hybrid Filtering with LightMF](notebooks/lightmf-hybrid-filtering.ipynb "Hybrid Filtering with LightMF")
- [x] create the _React-Native_ mobile app
  - [Mobile App](https://github.com/fleuryc/oc_p9_mobile-app "Mobile App")
- [x] integrate the _Recommder System_ in _Azure Functions_
  - [Store recommendations in Azure CosmosDB](notebooks/azure-store-recommendations.ipynb "Store recommendations in Azure CosmosDB")
  - [Azure Function](https://github.com/fleuryc/oc_p9_function "Azure Function")

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

- Set environment variable values in [.env](.env) file (copy or rename [.env.example](.env.example)).

### Azure resources

The app will query an [Azure CosmosDB](https://azure.microsoft.com/en-us/services/cosmos-db/ "Azure CosmosDB") database and an [Azure Function](https://azure.microsoft.com/en-us/services/functions/ "Azure Functions") that will return a list of relevant articles.

- [Azure Cosmos DB input binding for Azure Functions 2.x and higher - HTTP trigger, look up ID from route data](https://docs.microsoft.com/en-us/azure/azure-functions/functions-bindings-cosmosdb-v2-input?tabs=in-process%2Cfunctionsv2&pivots=programming-language-python#http-trigger-look-up-id-from-route-data-python "Azure Cosmos DB input binding for Azure Functions 2.x and higher - HTTP trigger, look up ID from route data")
- [Quickstart: Create a function in Azure with Python using Visual Studio Code](https://docs.microsoft.com/fr-fr/azure/azure-functions/create-first-function-vs-code-python "Quickstart: Create a function in Azure with Python using Visual Studio Code")

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
