# Assignment for the course of Distributed Systems II

Daniele Parmeggiani, a.y. 2021/2022.

### Setup

```bash
# python = python 3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running a single experiment

This will launch the experiment in a new browser tab.

```bash
python main.py run-experiment cyclon geo 1000 20 20 5 0.50
```

See `run-experiment --help` for information on parameters.

### Running all experiments

This will launch all experiments in batch-mode, providing no user interaction.

Please, beware of the high memory and compute consumption.

```bash
python main.py run-all
```

You may also specify the amount of CPU cores to use in order to reduce
impact on your system (by default, the amount of available cores will be used).

```bash
python main.py run-all --cores 3
```
