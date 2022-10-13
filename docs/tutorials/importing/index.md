These tutorials will help with reading / importing data

# Configuration Settings

Configuration objects contain all of the hardware and software settings used
during your measurement as well as settings for processing the data.  
This can include laser settings, DAQ settings, filter settings, etc...
Config objects can be created manually or imported from .yaml files:

1. [Manually](config/manually.ipynb)
2. [From a .yaml file](config/from_yaml.ipynb) 

# Measured Data 

Several classes exist to store measured data of different types relevant for wms.
The measurement objects made from these classes can be created in several ways:

1. [From a .tdms file](meas/from_tdms.ipynb)
2. [From a .csv file](meas/from_csv.ipynb) 

# Spectral Parameters 

Spectral data can be imported from Hitran / Hitemp format .par files
or with custom parameters from .yaml files.

1. [From hitemp / hitran .par file](spectral/from_hitran.ipynb)
2. [From .yaml file](spectral/from_yaml.ipynb)

# Molecular Parameters 

Molecular parameters must be imported from a .yaml file (or created manually). 

1. [From .yaml file](tips/from_yaml.ipynb) 

