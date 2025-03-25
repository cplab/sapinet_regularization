# Heterogeneous Quantization Regularizes Spiking Neural Network Activity

Spiking neural network simulations from [Moyal et al.](
https://doi.org/10.48550/arXiv.2409.18396) (2025), tested on Ubuntu 22.04.


Installation
------------
If you would like to run, modify, or extend this project:

* Clone the repository:

      git clone https://github.com/cplab/sapinet_regularization

* Create a conda virtual environment (optional):

      conda create -n <env_name> python=3.11
      conda activate <env_name>

* Install with pip:

      cd sapinet_regularization
      pip install -e .

See ``setup.py`` for dependency information.


Simulation
----------
* To reproduce the results, run the master script (multi-run):

      cd src/projects
      python simulation.py -multirun yaml/multi.yaml

* You may modify project, model, and component YAMLs to your liking and run single-use pipelines, e.g.:

      cd src/projects
      python simulation.py -experiment yaml/synthetic.yaml

Plots and diagnostic output will be written to `results/<run>`.


Analysis
--------
To reproduce the statistical analyses:

* Edit the following R file so that `run_dir` and `meta_dir` point to the run directory
generated in the previous step:

      src/utils/analysis/analysis.r

* Run `analysis.r` and inspect the results generated under `analysis/<run>`.

You may also directly inspect the output reported in the manuscript, included in this repository under:

      analysis/SciRep-Final


Note
----
Sapicore 0.4 is compatible with both Linux and Windows.
