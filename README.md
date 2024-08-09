<p align="center">
  <img src="asset/logo.png" alt="RelGen v0.1" width="500">
</p>

# RelGen

[![Unit Tests](https://github.com/ruc-datalab/RelGen/actions/workflows/unit.yml/badge.svg)](https://github.com/ruc-datalab/RelGen/actions/workflows/unit.yml)
[![E2E Tests](https://github.com/ruc-datalab/RelGen/actions/workflows/e2e.yml/badge.svg)](https://github.com/ruc-datalab/RelGen/actions/workflows/e2e.yml)
[![Colab](https://img.shields.io/badge/Tutorials-Try%20now!-orange?logo=googlecolab)](https://github.com/ruc-datalab/RelGen/blob/main/tutorial/census_synthesis.ipynb)
[![PyPi Latest Release](https://img.shields.io/pypi/v/relgen)](https://pypi.org/project/relgen/)
[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](./LICENSE)

**RelGen** is the abbreviation of **Rel**ation **Gen**eration. This tool is used to generate relational data in databases. 
Interestingly, the pronunciation of "Rel" closely resembles "Real," underscoring the fact that the relational data produced by RelGen is remarkably authentic and reliable. 

## Overview

RelGen is a Python library designed to generate real relational data for users. 
RelGen uses a variety of advanced deep generative models and algorithms to learn data distribution from real data and generate high-quality simulation data. s

<p align="center">
  <img src="asset/framework2.png" alt="RelGen v0.1" width="600">
  <br>
  <b>Figure</b>: RelGen Overall Architecture
</p>

## Features
* ✨ **Supports multiple fields and scenarios.** RelGen is suitable for a variety of scenarios, including private data release, data augmentation, database testing and so on.

* ✨ **Advanced relational data generation models and algorithms.** RelGen provides users with a variety of deep generative models to choose from, and uses effective relational data generation algorithms to generate high-quality relational data.

* ✨ **Comprehensive quality evaluation for generated relational data.** RelGen comprehensively evaluates the quality of generated relational data from multiple dimensions, 
and visualizes the difference between real relational data and generated relational data.

## Installation
RelGen requires Python version 3.7 or later.

### Install from pip

```bash
pip install relgen
```

### Install from source
```bash
git clone https://github.com/ruc-datalab/RelGen.git && cd RelGen
pip install -r requirements.txt
```

## Quick-Start

<font color='red'>这块可以进步的空间很大，写得更加详细一点。具体介绍如何load data generate data和evaluate synthesis data</font>

### Loading Dataset
Load a demo dataset to get started. This dataset is a single table describing the census.

Load metadata for the census dataset.
```python
from relgen.data.metadata import Metadata

metadata = Metadata()
metadata.load_from_json("datasets/census/metadata.json")
```

Load data for the census dataset.
```python
import pandas as pd

data = {
    "census": pd.read_csv("datasets/census/census.csv")
}
```

<p align="center">
  <img src="asset/census.png" alt="RelGen v0.1">
</p>

Encapsulate the census dataset and process it.
```python
from relgen.data.dataset import Dataset

dataset = Dataset(metadata)
dataset.fit(data)
```

### Generating Data

Train the synthesizer.
```python
from relgen.synthesizer.arsynthesizer import MADESynthesizer

synthesizer = MADESynthesizer(dataset)
synthesizer.fit(data)
```

Generate relational data.
```python
sampled_data = synthesizer.sample()
```

<p align="center">
  <img src="asset/synthetic_census.png" alt="RelGen v0.1">
</p>

### Evaluating Data

Compare real data and generated data to evaluate the quality of generated data.
```python
from relgen.evaluator import Evaluator

evaluator = Evaluator(data["census"], sampled_data["census"])
```

Show comparison histogram of data distribution between real data and generated data.
```python
evaluator.eval_histogram(columns=["age", "sex", "relationship"])
```

<p align="center">
  <img src="asset/histogram.png" alt="RelGen v0.1">
</p>

Show comparison t-SNE plot of data distribution between real data and generated data.
```python
evaluator.eval_tsne()
```

<p align="center">
  <img src="asset/t-SNE.png" alt="RelGen v0.1" width="300">
</p>

## Cite
If you find RelGen useful for your research or development, please cite the following paper: [Tabular data synthesis with generative adversarial networks: design space and optimizations](https://link.springer.com/article/10.1007/s00778-023-00807-y).

```bibtex
@article{liu2024tabular,
  title={Tabular data synthesis with generative adversarial networks: design space and optimizations},
  author={Liu, Tongyu and Fan, Ju and Li, Guoliang and Tang, Nan and Du, Xiaoyong},
  journal={The VLDB Journal},
  volume={33},
  number={2},
  pages={255--280},
  year={2024},
  publisher={Springer}
}
```
