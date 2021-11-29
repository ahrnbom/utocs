# Urban Traffic Observation in Carla Simulator (UTOCS) Dataset

The purpose of this repo is to provide a standard dataset for testing traffic observation software on videos from the [Carla Simulator](http://carla.org/). The Python code in this repo creates a number of standard videos with truly accurate ground truth positions for road users in world coordinates. Hopefully, this will allow a more fair comparison between different road user tracking methods. While Carla Simulator does not create perfectly realistic traffic videos, they are hopefully realistic enough to be useful for computer vision and traffic researchers.

The following road user classes are supported:
1. Pedestrians
1. Bicyclists/Motorbikes
1. Cars
1. Trucks
1. Buses

The videos are stored as 720p .jpg files, at 25 FPS. By default, monocular videos are generated, but an option exists for stereo/trinocular or even more cameras through the `--cam_setup` parameter. There are 24 sequences, each 4000 frames long. These are divided into training/validation/test sets, see `sets.txt`. The sequences are recorded across several different Carla maps, with a wide variety of camera angles, chosen to be somewhat feasible in a traffic surveillance setting, with the camera placed several metres above the ground and observing an active traffic environment. For each frame in each sequence, there exists a text file containing the ground truth position, in world coordinates, of each road user. The road users have consistent IDs across all frames. Thus this dataset can be used to evaluate world coordinate tracking methods designed for use on road users seen in surveillance cameras, of the kind that is of importance for traffic safety researchers.

### Example image
![Example image from UTOCS](https://raw.githubusercontent.com/ahrnbom/utocs/main/examples/example1.jpg)

### License
*LICENSE COMING SOON*

### Instructions
1. Install Carla Simulator 0.9.12, both the Python package and the server
1. Start the server with `./CarlaUE4.sh` in the appropriate folder
1. Install Python dependencies with `python3 -m pip install carla==0.9.12 shapely motmetrics scipy numpy`, or to include the optional dependencies for visualization do instead `python3 -m pip install carla==0.9.12 imageio imageio-ffmpeg opencv-python shapely motmetrics scipy numpy`
1. While the server is running, execute `python3 start.py`. Use the `-h` flag to see options. 
1. To run the optional visualization, run `python3 visualize_samples.py`. Again, use `-h` to see options.

### Evaluation
To evaluate a method against the UTOCS dataset, make sure your method exports to the following JSON format:

A folder, typically named after the method itself, should contain one subfolder for each sequence (`0000`, `0001` and so on). Each such folder should contain .json files for each frame, called `000000.json`, `000001.json` and so on. These files should be JSON formatted, as a list containing objects with (at least) the following entries:
1. `"type"` - the road user class ("car", "bicyclist", "pedestrian", "truck" or "bus").
1. `"id"` - ID of that road user, should be consistent across frames. Does obviously not need to match the ground truth IDs!
1. `"x"`, `"y"` and `"z"`: The x, y and z position of the lower center point of the road user (touching the ground).
1. `"l"`, `"w"` and `"h"`: The length (in the forward direction), width and height (upwards) of the road user. The height is currently not used.
1. `"forward_x"`, `"forward_y"` and `"forward_z"`: A normalized vector for the direction the road user is facing, representing the orientation of the road user.

The format is essentially the same as how UTOCS stores the ground truth. 

To run evaluation, simply run `python3 eval.py --folder PATH/TO/METHOD/FOLDER` and the average MOTA across all sequences will be presented in the terminal.

### Notes
- Note that Carla Simulator has a tendency to crash if left to run for several hours. If that happens, use the `--range` parameter to `start.py` to specify which scenarios still need to be generated, so that you do not need to run all sequences in a single go.
- When generating images with multiple cameras, make sure to use quotes (`'`) around the argument. Otherwise, your shell will likely interpret the semicolons (`;`) as splitting the different commands. An example of a correct command for stereo images is `python3 start.py --cam_setup '0,0,0;0.5,0,0'`, which will give you a stereo setup with 0.5 m between the two cameras, horizontally. The format uses three coordinates to allow exotic camera configurations; to make a stereo camera where the two cameras are placed above each other, use something like `0,0,0;0,0.5;0` and similarly, `0,0,0;0,0,0.5` will give a setup where one of the cameras is placed in front of the other one (which is somewhat unrealistic).
- For a monocular view, the entire dataset is approximately 60 GB in size. 
- Only moving road users are included in the ground truth. Your tracking method should remove any tracks without significant movement before evaluation. 
- The tracking method to be evaluated is allowed (and supposed) to use the camera calibration and ground points for computing the tracks. The method is not allowed to use the positions text files from the test set in any way prior to evaluation. 

### Dataset Download
If you are unable or unwilling to run the UTOCS code to generate the dataset, you can also download the dataset from this link, with only a monocular view:

*LINK COMING SOON*

### Citation
If you use the UTOCS dataset in your research, please cite this paper:

*BIBTEX COMING SOON*