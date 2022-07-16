# MLOps: From Models to Production

A code repo for the [MLOps: From Models to Production](https://corise.com/course/mlops/)


## Getting Started

This projects relies on [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/).

1. Install the required Python version:

   ```bash
   pyenv install
   ```

2. Install dependencies

   ```bash
   poetry install --no-dev
   ```

3. Prepare the AG News data set

   ```bash
   poetry run python -m app.data.prepare_agnews
   ```
