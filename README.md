# Standard Traffic Surveillance Video Dataset for Carla Simulator (STSVDC)

The purpose of this repo is to provide a standard dataset for testing traffic surveillance software on videos from the [Carla Simulator](http://carla.org/). The Python code in this repo creates a number of standard videos with ground truth positions for road users in world coordinates. Hopefully, this will allow a more fair comparison between different road user tracking methods.  

The following road user classes are supported:
1. Pedestrians
1. Bicyclists/Motorbikes
1. Cars
1. Trucks
1. Buses

The videos are stored as 720p .jpg files, as 25 FPS videos. By default, monocular videos are generated, but an option exists for stereo/trinocular or even more cameras through the `--cam_setup` parameter. There are 24 sequences, each 4000 frames long. These are divided into training/validation/test sets, see `sets.txt`.

### License
To be decided...

### Instructions
1. Install Carla Simulator 0.9.12, both the Python package and the server
1. Start the server with `./CarlaUE4.sh` in the appropriate folder
1. Install Python dependencies with `python3 -m pip install carla`, or to include the optional dependencies for visualization do instead `python3 -m pip install carla imageio imageio-ffmpeg opencv-python`
1. While the server is running, execute `python3 start.py`. Use the `-h` flag to see options.
1. To run the optional visualization, run `python3 visualize_samples.py`. Again, use `-h` to see options.