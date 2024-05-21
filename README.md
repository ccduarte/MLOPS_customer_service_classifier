[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quantumfinancesac.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repositório%20app-black)](https://github.com/ccduarte/MLOPS_customer_service_classifier-front)
[![GitHub](https://img.shields.io/badge/GitHub-Repositório%20API-black)](https://github.com/ccduarte/MLOPS_customer_service_classifier-api)

Modelo de classificação de atendimento.
==============================

Este repositório contém o desenvolvimento de um modelo de classificação textual de atendimento da startup QuantumFinance e faz parte do projeto da disciplina de MLOps do curso MBA Data Science & Artificial Intelligence da FIAP.

O objetivo desse modelo é colocar em prática tecnicas de MLOps, o foco não é o modelo em si apresentado. 

O experimentos são devidamente rastreáveis pelo DVC. 
[![DVC](https://img.shields.io/badge/DVC-Registro%20Experimentos-blue)](https://studio.iterative.ai/user/ccduarte/projects/MLOPS_customer_service_classifier-dcvyyf364b)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
