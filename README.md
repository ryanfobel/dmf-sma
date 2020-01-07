# DMF Simulation, Modeling, and Analysis #

Code for simulating, modeling, and analyzing [Digital Microfluidics][1]
force and velocity data.

# Installation

The easiest way to install the required dependencies is to create a new conda
environment:

```sh
conda create --name dmf-sma numpy pandas sympy matplotlib
conda activate dmf-sma
```

# Use

You can run the 'electromechanical_model.py' submodule as a script to see the
different types of plots that can be generated:

```sh
python electroemechanical_model.py
```

# Authors #

Ryan Fobel <ryan@fobel.net>  
Christian Fobel <christian@fobel.net>

[1]: http://microfluidics.utoronto.ca/dropbot
