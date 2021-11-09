"""
    Simple script to get the position and orientation of camera written to the terminal. 
    This script was used to create the default scenarios.
"""

import carla
import argparse
import time 

from stsvdc import loc_dist

host = '127.0.0.1'
port = 4000
client = carla.Client(host, port)
client.set_timeout(10.0)

args = argparse.ArgumentParser()
args.add_argument("--map", type=str, default="Town10")
args = args.parse_args()
map_name = args.map

client.load_world(map_name)
time.sleep(1.0)

world = client.get_world()

old_pos = None 
def_length = 2000
while True:
    transform = world.get_spectator().get_transform()
    pos = transform.location
    rot = transform.rotation
    if old_pos and loc_dist(pos, old_pos) > 0.001:
        print(f"Scenario(map='{map_name}', cam_pos=({pos.x},{pos.y},{pos.z}), cam_dir=({rot.pitch},{rot.yaw},{rot.roll}), length={def_length})")
    old_pos = pos