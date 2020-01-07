# DMF Simulation, Modeling, and Analysis #

Code for simulating, modeling, and analyzing [Digital Microfluidics][1]
force and velocity data.

# Installation

The easiest way to install the required dependencies is to create a new
[conda](https://docs.conda.io/en/latest/miniconda.html) environment:

```sh
conda create --name dmf-sma
conda activate dmf-sma
conda config --env --add channels conda-forge
conda install numpy pandas sympy matplotlib
```

# Use

Run the 'electromechanical_model.py' submodule as a script to see the
different types of plots that can be generated:

```sh
python electroemechanical_model.py
```

# Authors #

Ryan Fobel <ryan@fobel.net>  
Christian Fobel <christian@fobel.net>

[1]: http://microfluidics.utoronto.ca/dropbot
