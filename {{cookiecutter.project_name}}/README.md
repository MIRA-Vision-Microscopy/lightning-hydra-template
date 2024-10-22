<div align="center">

# Project: {{cookiecutter.project_name}}

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/MIRA-Vision-Microscopy/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

This project was created from our [cookiecutter template](https://github.com/MIRA-Vision-Microscopy/lightning-hydra-template) in version {{cookiecutter.template_version}}.
</div>


## Description

What it does


## Installation

If you are using pyenv for venv management, set local Python version
```bash
pyenv local 3.x
```

Select  Python environment
```bash
poetry env use python3.x
# or
poetry env use /full/path/to/python
```

Install packages
```bash
poetry install 
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Template documentation and key features

For a detailed documentation of the repository features and dynamic instantiation of config files, please visit the [template repository](https://github.com/MIRA-Vision-Microscopy/lightning-hydra-template).  
