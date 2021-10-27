# Standard Traffic Surveillance Video Dataset for Carla Simulator (STSVDC)

The purpose of this repo is to provide a standard dataset for testing traffic surveillance software on videos from the [Carla Simulator](http://carla.org/). The Python code in this repo creates a number of standard videos with ground truth positions for road users in world coordinates. Hopefully, this will allow a more fair comparison between different road user tracking methods.  

The following road user classes are supported:
1. Pedestrians
1. Bicyclists
1. Cars
1. Trucks
1. Buses

### License
To be decided...

### Instructions
1. Install Carla Simulator 0.9.12, both the Python package and the server
1. Start the server with `./CarlaUE4.sh` in the appropriate folder
1. Run `python stsvdc.py`