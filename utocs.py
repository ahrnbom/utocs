"""
    Creates standard videos 
"""

import argparse
from pathlib import Path 
import sys
import traceback
import random
import time 
from dataclasses import dataclass
from queue import Queue
from math import tan, pi, sqrt
import json
import numpy as np 
import imageio as iio

import carla
from carla import VehicleLightState as vls

from util import loc_dist, vector_normalize, long_str, vector_from_to, scalar_product
from cameras import build_cam, is_visible, is_obj_visible

@dataclass
class Scenario:
    map:str
    cam_pos:tuple
    cam_dir:tuple
    length:int

@dataclass 
class Camera:
    obj:carla.Sensor
    id:int
    loc:carla.Location
    dir:carla.Rotation
    name:str
    transform:carla.Transform
    segobj:carla.Sensor

@dataclass
class Pedestrian:
    id:int
    speed:float 
    con:int

def default_scenarios():
    scenarios = list()

    # 0 
    s = Scenario(map='Town04', cam_pos=(193.04466247558594,-257.2019958496094,7.6563401222229), cam_dir=(-43.125633239746094,50.72932815551758,2.5735223971423693e-05), length=3000)
    scenarios.append(s)

    # 1 
    s = Scenario(map='Town10HD', cam_pos=(-51.66246032714844,154.75738525390625,13.823832511901855), cam_dir=(-27.47414779663086,-86.13775634765625,0.00011643866309896111), length=3000)
    scenarios.append(s)

    # 2 
    s = Scenario(map='Town10HD', cam_pos=(-63.60895919799805,105.49251556396484,14.712573051452637), cam_dir=(-51.93895721435547,1.8427033424377441,6.924353215254087e-07), length=3000)
    scenarios.append(s)

    # 3
    s = Scenario(map='Town10HD', cam_pos=(-47.0649299621582,-40.33244705200195,11.453038215637207), cam_dir=(-36.29405975341797,-88.5447998046875,0.00011016575444955379), length=3000)
    scenarios.append(s)

    # 4
    s = Scenario(map='Town10HD', cam_pos=(-112.13251495361328,119.8772201538086,12.588258743286133), cam_dir=(-48.381961822509766,-8.009498596191406,6.395316449925303e-05), length=3000)
    scenarios.append(s)

    # 5
    s = Scenario(map='Town02', cam_pos=(37.02091598510742,246.881103515625,9.387005805969238), cam_dir=(-41.42270278930664,-33.86310958862305,3.529641617205925e-05), length=3000)
    scenarios.append(s)

    # 6
    s = Scenario(map='Town02', cam_pos=(122.1256332397461,183.66986083984375,9.662296295166016), cam_dir=(-40.59231185913086,47.09394836425781,7.083312084432691e-05), length=3000)
    scenarios.append(s)

    # 7
    s = Scenario(map='Town03', cam_pos=(-14.29495620727539,31.68166160583496,16.79800033569336), cam_dir=(-51.126773834228516,-25.994699478149414,3.4009499358944595e-05), length=3000)
    scenarios.append(s)

    # 8 
    s = Scenario(map='Town03', cam_pos=(-99.9942626953125,13.837921142578125,18.240234375), cam_dir=(-50.94935989379883,-21.88389778137207,3.7945083022350445e-05), length=3000)
    scenarios.append(s)

    # 9
    s = Scenario(map='Town03', cam_pos=(-156.63992309570312,-11.973634719848633,8.718941688537598), cam_dir=(-40.18968963623047,46.936527252197266,2.570556716818828e-05), length=3000)
    scenarios.append(s)

    # 10 
    s = Scenario(map='Town10HD', cam_pos=(-91.20193481445312,37.65811538696289,13.142256736755371), cam_dir=(-49.871089935302734,-139.0921173095703,0.00019340438302606344), length=3000)
    scenarios.append(s)

    # 11 
    s = Scenario(map='Town05', cam_pos=(21.694469451904297,-13.837503433227539,14.421462059020996), cam_dir=(-47.781150817871094,65.34835815429688,-2.5411281967535615e-06), length=3000)
    scenarios.append(s)

    # 12 
    s = Scenario(map='Town05', cam_pos=(19.721576690673828,-102.66522979736328,16.00419807434082), cam_dir=(-48.494869232177734,55.754634857177734,1.5460213035112247e-05), length=3000)
    scenarios.append(s)

    # 13 
    s = Scenario(map='Town10HD', cam_pos=(116.15340423583984,77.90415954589844,12.076078414916992), cam_dir=(-48.265743255615234,-147.64822387695312,9.106213110499084e-05), length=3000)
    scenarios.append(s)

    # 14 
    s = Scenario(map='Town10HD', cam_pos=(-56.86170959472656,41.37298583984375,13.404866218566895), cam_dir=(-44.32744216918945,69.0594482421875,7.877043390180916e-05), length=3000)
    scenarios.append(s)

    # 15 
    s = Scenario(map='Town10HD', cam_pos=(51.79150390625,56.6481819152832,8.31950855255127), cam_dir=(-38.744956970214844,137.55307006835938,2.8461327019613236e-05), length=3000)
    scenarios.append(s)

    # 16
    s = Scenario(map='Town02', cam_pos=(173.74713134765625,119.1712646484375,7.785608291625977), cam_dir=(-28.29818344116211,-26.61002540588379,6.108825618866831e-05), length=3000)
    scenarios.append(s)

    # 17 
    s = Scenario(map='Town02', cam_pos=(-9.025032043457031,104.06327819824219,7.86112117767334), cam_dir=(-29.11554527282715,44.28169631958008,6.645367102464661e-05), length=3000)
    scenarios.append(s)

    # 18
    s = Scenario(map='Town05', cam_pos=(-140.45082092285156,80.63668823242188,24.408430099487305), cam_dir=(-62.266754150390625,33.93044662475586,0.0005559046985581517), length=3000)
    scenarios.append(s)

    # 19 
    s = Scenario(map='Town10HD', cam_pos=(-59.03020477294922,35.21854782104492,13.662759780883789), cam_dir=(-47.19268035888672,-47.13734436035156,6.533325358759612e-05), length=3000)
    scenarios.append(s)

    # 20 
    s = Scenario(map='Town10HD', cam_pos=(33.16043472290039,42.02176284790039,24.430490493774414), cam_dir=(-53.3353271484375,-63.22784423828125,4.861298293690197e-05), length=3000)
    scenarios.append(s)

    # 21 
    s = Scenario(map='Town10HD', cam_pos=(7.524465084075928,-40.997764587402344,24.158056259155273), cam_dir=(-51.83544158935547,-91.22041320800781,0.00016856557340361178), length=3000)
    scenarios.append(s)

    # 22 
    s = Scenario(map='Town04', cam_pos=(72.52837371826172,-180.96755981445312,9.338313102722168), cam_dir=(-42.46043395996094,147.30682373046875,0.0), length=3000)
    scenarios.append(s)

    # 23
    s = Scenario(map='Town04', cam_pos=(23.82340431213379,-165.28225708007812,10.662442207336426), cam_dir=(-51.7166748046875,-141.62094116210938,2.3426902771461755e-05), length=3000)
    scenarios.append(s)

    # 24 
    s = Scenario(map='Town04', cam_pos=(-30.05730438232422,-73.35159301757812,12.965838432312012), cam_dir=(-48.75633239746094,19.82217788696289,4.791660830960609e-05), length=3000)
    scenarios.append(s)

    # 25
    s = Scenario(map='Town10HD', cam_pos=(-104.29922485351562,-23.038036346435547,20.315933227539062), cam_dir=(-45.07830047607422,-75.88121032714844,9.672575288277585e-06), length=3000)
    scenarios.append(s)

    # 26 
    s = Scenario(map='Town02', cam_pos=(57.11075210571289,293.70635986328125,9.862232208251953), cam_dir=(-44.1141242980957,146.2111358642578,2.3783497454132885e-05), length=3000)
    scenarios.append(s)

    # 27
    s = Scenario(map='Town10HD', cam_pos=(-89.47510528564453,135.5434112548828,17.469736099243164), cam_dir=(-39.64571762084961,-118.50187683105469,0.00021954052499495447), length=3000)
    scenarios.append(s)

    # 28
    s = Scenario(map='Town10HD', cam_pos=(98.21633911132812,137.0532989501953,13.589766502380371), cam_dir=(-48.36974334716797,-133.8146209716797,0.00010538495553191751), length=3000)
    scenarios.append(s)

    # 29
    s = Scenario(map='Town10HD', cam_pos=(-67.2658462524414,125.19412994384766,11.084273338317871), cam_dir=(-44.82524108886719,168.47000122070312,0.00012759777018800378), length=3000)
    scenarios.append(s)

    # 30
    s = Scenario(map='Town10HD', cam_pos=(-94.6222152709961,1.5529531240463257,12.659930229187012), cam_dir=(-50.385921478271484,125.7801513671875,0.00012318938388489187), length=3000)
    scenarios.append(s)

    # 31 
    s = Scenario(map='Town10HD', cam_pos=(-34.75411605834961,4.579659938812256,9.77426528930664), cam_dir=(-47.3773078918457,-141.1136932373047,0.00013364497863221914), length=3000)
    scenarios.append(s)

    # 32 
    s = Scenario(map='Town10HD', cam_pos=(29.76532554626465,39.30972671508789,14.098193168640137), cam_dir=(-54.682037353515625,-35.331851959228516,0.00011962300050072372), length=3000)
    scenarios.append(s)

    # 33 
    s = Scenario(map='Town10HD', cam_pos=(92.457763671875,83.59999084472656,11.70304012298584), cam_dir=(-46.222801208496094,-56.076786041259766,5.182942913961597e-05), length=3000)
    scenarios.append(s)

    # 34 
    s = Scenario(map='Town02', cam_pos=(198.55555725097656,107.4876708984375,10.433656692504883), cam_dir=(-47.84370040893555,152.6953125,3.689077129820362e-05), length=3000)
    scenarios.append(s)

    # 35 
    s = Scenario(map='Town02', cam_pos=(-1.4922281503677368,99.3492202758789,8.707717895507812), cam_dir=(-44.144351959228516,75.05903625488281,0.00014753315190318972), length=3000)
    scenarios.append(s)

    # 36 
    s = Scenario(map='Town05', cam_pos=(-65.62116241455078,-96.86927795410156,13.862954139709473), cam_dir=(-51.59771728515625,21.150402069091797,4.260765854269266e-05), length=3000)
    scenarios.append(s)

    # 37
    s = Scenario(map='Town05', cam_pos=(-111.85202026367188,-150.71258544921875,14.686253547668457), cam_dir=(-59.8665771484375,88.6191635131836,8.503461140207946e-05), length=3000)
    scenarios.append(s)

    # 38 
    s = Scenario(map='Town05', cam_pos=(-185.8836212158203,-140.8562469482422,17.03713607788086), cam_dir=(-63.49945831298828,48.140071868896484,8.036309009185061e-05), length=3000)
    scenarios.append(s)

    # 39
    s = Scenario(map='Town04', cam_pos=(209.04669189453125,-162.4080047607422,10.710044860839844), cam_dir=(-49.121089935302734,-130.51943969726562,1.8263584934175014e-05), length=3000)
    scenarios.append(s)

    return scenarios


def run_scenario(client, traffic_manager, cam_setup:list, scenario:Scenario, scenario_number:int, 
                 folder:Path, frame_skip:int=5):
    client.load_world(scenario.map)
    time.sleep(5.0)
    print(f"Loaded map {scenario.map}") 

    # Some parameters
    number_of_vehicles=80
    number_of_pedestrians=30
    im_size_x = 1280
    im_size_y = 720
    fov = 90.0

    # Random seed
    seed = 9001
    random.seed(seed)

    # Setup world and traffic manager
    world = client.get_world()
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.set_random_device_seed(seed)
    world.set_pedestrians_seed(seed)

    # Set timing/sync settings
    settings = world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.008 # timing for physics simulations, needs to be below 0.01
    world.apply_settings(settings)

    # Setup blueprints for road user spawning
    bps_vehicles = world.get_blueprint_library().filter('vehicle.*')
    bps_pedestrians = world.get_blueprint_library().filter('walker.pedestrian.*')
    bps_vehicles = sorted(bps_vehicles, key=lambda bp: bp.id)
    bps_pedestrians = sorted(bps_pedestrians, key=lambda bp: bp.id)

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Hack-import objects for spawning
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    time.sleep(1.0)

    # Spawn vehicles
    vehicles = list()
    batch = list()
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(bps_vehicles)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        light_state = vls.NONE
        
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, light_state)))
    
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            raise ValueError(response.error)
        else:
            vehicles.append(response.actor_id)
    
    print(f"Spawned {len(vehicles)} vehicles")

    # Spawn pedestrians
    pedestrians = list()
    walker_spawns = list()
    for _ in range(number_of_pedestrians):
        spawn_point = carla.Transform()
        loc = None
        while loc is None :
            loc = world.get_random_location_from_navigation()
            loc.z += 1.0 # To prevent collisions with the ground
            
            all_spawns = spawn_points + walker_spawns
            distances = [loc_dist(previous.location, loc) for previous in all_spawns]
            if min(distances) < 4.0:
                # To precent collisions with other road users 
                loc = None 
        spawn_point.location = loc
        
        walker_spawns.append(spawn_point) 
    
    batch = list()
    pedestrian_speeds = list()
    for spawn_point in walker_spawns:
        walker_bp = random.choice(bps_pedestrians)
        if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
        assert walker_bp.has_attribute('speed')
        if random.random() < 0.9:
            pedestrian_speeds.append(walker_bp.get_attribute('speed').recommended_values[1]) # walking speed
        else:
            pedestrian_speeds.append(walker_bp.get_attribute('speed').recommended_values[2]) # running speed
        
        batch.append(SpawnActor(walker_bp, spawn_point))
    
    results = client.apply_batch_sync(batch, True)
    for i, res in enumerate(results):
        if res.error:
            raise ValueError(res.error)
        else:
            pedestrian = Pedestrian(res.actor_id, float(pedestrian_speeds[i]), con=None)
            pedestrians.append(pedestrian)
            
    print("Actors spawned!")

    batch = list()
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for pedestrian in pedestrians:
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), pedestrian.id))
    results = client.apply_batch_sync(batch, True)
    for i, res in enumerate(results):
        if res.error:
            raise ValueError(res.error)
        else:
            pedestrians[i].con = res.actor_id
    
    world.tick()

    world.set_pedestrians_cross_factor(0.05)
    
    # Start moving pedestrians
    controller_actors = world.get_actors([p.con for p in pedestrians])
    for pedestrian, actor in zip(pedestrians, controller_actors):
        actor.start()
        actor.go_to_location(world.get_random_location_from_navigation())
        actor.set_max_speed(pedestrian.speed)

    print(f"Spawned {len(pedestrians)} pedestrians")

    traffic_manager.global_percentage_speed_difference(30.0)

    print("Stabilizing world...")
    for _ in range(150):
        world.tick()
        time.sleep(0.01)

    print("World stabilized")

    # Set up folders for data storage
    scenario_folder = folder / 'scenarios' / f"{long_str(scenario_number)}"
    scenario_folder.mkdir(parents=True, exist_ok=True)

    pos_folder = scenario_folder / 'positions'
    pos_folder.mkdir(exist_ok=True)

    ims_folder = scenario_folder / 'images'
    ims_folder.mkdir(exist_ok=True)

    seg_folder = scenario_folder / 'instance_segmentations'
    seg_folder.mkdir(exist_ok=True)

    # Spawn camera(s)
    sensor_queue = Queue()
    seg_sensor_queue = Queue()
    cameras = list()

    start_frame = world.get_snapshot().frame
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    
    # Instance segmentation camera:
    segcam_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    
    cam_bp.set_attribute('image_size_x', str(im_size_x))
    cam_bp.set_attribute('image_size_y', str(im_size_y))
    cam_bp.set_attribute('fov', str(fov))
    segcam_bp.set_attribute('image_size_x', str(im_size_x))
    segcam_bp.set_attribute('image_size_y', str(im_size_y))
    segcam_bp.set_attribute('fov', str(fov))

    base_location = carla.Location(*scenario.cam_pos)
    base_rotation = carla.Rotation(*scenario.cam_dir)
    base_transform = carla.Transform(base_location, base_rotation)
    
    up = base_transform.get_up_vector()
    right = base_transform.get_right_vector()
    forward = base_transform.get_forward_vector()

    for cam_no, cam_delta in enumerate(cam_setup):
        dx, dy, dz = cam_delta  
        cam_loc = base_location + dx*right + dy*up + dz*forward
        cam_loc = carla.Location(cam_loc.x, cam_loc.y, cam_loc.z)
        
        transform = carla.Transform(cam_loc, base_rotation)
        cam = world.spawn_actor(cam_bp, transform)
        cam_name = f"cam{cam_no}"

        def closure(sensor_queue, cam_name, start_frame, ims_folder, frame_skip):
            return lambda data: sensor_callback(data, sensor_queue, cam_name, start_frame, ims_folder, frame_skip, 'jpg')
        cam.listen(closure(sensor_queue, cam_name, start_frame, ims_folder, frame_skip))

        segcam = world.spawn_actor(segcam_bp, transform)
        def closure(seg_sensor_queue, cam_name, start_frame, seg_folder, frame_skip):
            return lambda data: sensor_callback(data, seg_sensor_queue, cam_name, start_frame, seg_folder, frame_skip, 'png')
        segcam.listen(closure(seg_sensor_queue, cam_name, start_frame, seg_folder, frame_skip))
        
        camera = Camera(cam, cam_no, cam_loc, base_rotation, cam_name, transform, segcam)
        cameras.append(camera)

    # Collect camera positions
    cams_obj = dict()
    cam_obj_list = list()
    for cam in cameras:
        cam_obj = {'id': cam.id, 'x':cam.loc.x, 'y':cam.loc.y, 'z':cam.loc.z,
                   'pitch': cam.dir.pitch, 'roll': cam.dir.roll, 'yaw': cam.dir.yaw}
        cam_obj_list.append(cam_obj)

    f  = im_size_x /(2.0 * tan(fov * pi / 360))
    Cx = im_size_x / 2.0
    Cy = im_size_y / 2.0
    cams_obj['instrinsics'] = {'f':f, 'Cx': Cx, 'Cy':Cy}
    cams_obj['cams'] = cam_obj_list 

    (scenario_folder / 'cameras.json').write_text(json.dumps(cams_obj, indent=2))

    # Build projection matrix
    
    cam = cameras[0]
    cam_values = {'id': cam.id, 'x':cam.loc.x, 'y':cam.loc.y, 'z':cam.loc.z,
                  'pitch': cam.dir.pitch, 'roll': cam.dir.roll, 
                  'yaw': cam.dir.yaw, 'f': f, 'Cx': Cx, 'Cy': Cy}
    camera_matrix, _ = build_cam(cam_values)

    bus_names = ('volkswagen.t2', 'volkswagen.t2_2021' )
    truck_names = ('carlamotors.firetruck', 'ford.ambulance', 'mercedes.sprinter', 'tesla.cybertruck', 'carlamotors.carlacola')

    first_id = min(vehicles)

    # Collect ground points
    camera = cameras[0]
    cam_t = camera.transform
    cam_forward = vector_normalize(cam_t.get_forward_vector())
    ground_dist = camera.loc.z / abs(cam_forward.z) # how far we should walk until we hit ground plane (z=0)
    ground_pos = camera.loc + ground_dist*cam_forward 
    n_points = 1000
    sqrt_n_points = int(round(sqrt(n_points)))
    ground_range = 50 # in metres
    ground_points = list()
    for ix in range(sqrt_n_points):
        dx = (ix-sqrt_n_points/2.0)*2*ground_range/sqrt_n_points
        for iy in range(sqrt_n_points):
            dy = (iy - sqrt_n_points/2.0)*2*ground_range/sqrt_n_points

            point = carla.Location(ground_pos.x + dx, ground_pos.y + dy, ground_pos.z + 10.0)
            point = world.ground_projection(point, 15.0)
            if point is not None:
                loc = point.location
                if abs(loc.z) < 2.0: # any further up and we assume they're not on the actual ground 

                    # Check if visible in camera
                    if is_visible(loc.x, loc.y, loc.z, camera_matrix, 
                                  im_h=im_size_y, im_w=im_size_x):
                        ground_points.append(loc)

    # Store ground points in file 
    ground_lines = [f"{point.x},{point.y},{point.z}" for point in ground_points]
    (scenario_folder / 'ground_points.txt').write_text('\n'.join(ground_lines))

    # Start recording video
    max_dist_to_include = 50 # in metres
    min_height_pixels = 15

    for frame_no in range(scenario.length*frame_skip):
        # Tick the server
        world.tick()

        actual_frame_no = frame_no // frame_skip
        is_actual_frame = frame_no%frame_skip == 0

        if is_actual_frame:
            print(f"Frame number {actual_frame_no} / {scenario.length}")

        # Poll the sensors
        for _ in range(len(cameras)):
            fname = sensor_queue.get(True, 1.0)
            if fname is not None:
                print(fname)

            fname = seg_sensor_queue.get(True, 1.0)
            if fname is not None:
                print(fname)
        
        if is_actual_frame:
            # Save ground-truth position of all nearby road users 
            objs = list()
            for vehicle_id in vehicles:
                vehicle = world.get_actor(vehicle_id)
                loc = vehicle.get_location()

                # Ignore those too far away
                if loc_dist(loc, cameras[0].loc) > max_dist_to_include:
                    continue

                bbox = vehicle.bounding_box
                l = bbox.extent.x*2
                w = bbox.extent.y*2
                h = bbox.extent.z*2
                    
                # Ignore if not visible in camera
                if not is_obj_visible(loc.x, loc.y, loc.z, h, camera_matrix,
                                      im_h=im_size_y, im_w=im_size_x, 
                                      min_height=min_height_pixels):
                    continue 

                vehicle_type = 'car'
                if any([bn in vehicle.type_id for bn in bus_names]):
                    vehicle_type = 'bus'
                elif any([tn in vehicle.type_id for tn in truck_names]):
                    vehicle_type = 'truck'
                elif vehicle.attributes['number_of_wheels'] == "2":
                    vehicle_type = 'bicyclist'
                
                if abs(w) < 0.001:
                    # Sometimes bicyclists have zero width due to a Carla bug
                    if vehicle_type == 'bicyclist':
                        w = 0.5
                    else:
                        raise ValueError(f"Road user with ID {vehicle_id-first_id} of type {vehicle_type} has no width! Not sure what to do")

                v = vehicle.get_velocity()
                rot = vehicle.get_transform().rotation
                forward = rot.get_forward_vector()

                objs.append({'type': vehicle_type, 'id': vehicle_id-first_id,
                             'x': loc.x, 'y': loc.y, 'z': loc.z, 'l': l, 'w': w,
                             'h': h, 'v_x': v.x, 'v_y': v.y, 'v_z': v.z,
                             'pitch': rot.pitch, 'roll': rot.roll, 'yaw': rot.yaw, 
                             'forward_x': forward.x, 'forward_y': forward.y, 
                             'forward_z': forward.z})
            
            for pedestrian in pedestrians:
                pedestrian_id = pedestrian.id
                actor = world.get_actor(pedestrian_id)
                loc = actor.get_location()
                
                # Ignore those too far away
                if loc_dist(loc, cameras[0].loc) > max_dist_to_include:
                    continue

                bbox = actor.bounding_box
                l = bbox.extent.x*2
                w = bbox.extent.y*2
                h = bbox.extent.z*2

                # Pedestrians' z value is at their middle height for some reason
                ped_z = loc.z - h/2.0

                # Ignore if not visible in camera
                if not is_obj_visible(loc.x, loc.y, ped_z, h, camera_matrix,
                                      im_h=im_size_y, im_w=im_size_x, 
                                      min_height=min_height_pixels):
                    continue 

                v = actor.get_velocity()
                rot = actor.get_transform().rotation
                forward = rot.get_forward_vector()

                objs.append({'type': 'pedestrian', 'id': pedestrian_id-first_id,
                             'x': loc.x, 'y': loc.y, 'z': ped_z, 'l': l, 'w': w,
                             'h': h, 'v_x': v.x, 'v_y': v.y, 'v_z': v.z,
                             'pitch': rot.pitch, 'roll': rot.roll, 'yaw': rot.yaw, 
                             'forward_x': forward.x, 'forward_y': forward.y, 
                             'forward_z': forward.z})
            
            pos_path = pos_folder / f"{long_str(actual_frame_no, 6)}.json"
            (pos_path).write_text(json.dumps(objs, indent=2))
            print(f"Written {pos_path}")
    
    # Cleanup before next scenario (if any)
    print("Cleaning up...")
    for actor in controller_actors:
        actor.stop()
    time.sleep(0.2)

    for camera in cameras:
        camera.obj.stop()
        camera.segobj.stop()
    time.sleep(0.2)
    
    client.apply_batch([carla.command.DestroyActor(v_id) for v_id in vehicles])
    time.sleep(0.2)
    
    client.apply_batch([carla.command.DestroyActor(pedestrian.con) for pedestrian in pedestrians])
    time.sleep(0.2)
    
    client.apply_batch([carla.command.DestroyActor(pedestrian.id) for pedestrian in pedestrians])
    time.sleep(0.2)
    
    print("Scenario finished!")


# What happens when an RGB sensor (camera) records an image
# Should be saved to file or not, depending on frame skips
def sensor_callback(data, sensor_queue, cam_name, start_frame, folder, frame_skip, suffix):
    frame_no = data.frame - start_frame - 1
    if frame_no % frame_skip == 0:    
        file_path = folder / cam_name / f"{long_str(frame_no//frame_skip, 6)}.{suffix}"
        file_path.parent.mkdir(exist_ok=True)
        
        if suffix == 'jpg':
            data.save_to_disk(str(file_path))
        elif suffix == 'png':
            array = np.frombuffer(data.raw_data, dtype=np.uint8) 
            array = np.reshape(array, (data.height, data.width, 4))
            iio.imwrite(file_path, array)

        sensor_queue.put(file_path)
    else:
        sensor_queue.put(None)

def main(host:str, port:int, tm_port:int, cam_setup:list, folder:Path, 
         scenario_number:int):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager(tm_port)

    scenarios = default_scenarios()
    scenario = scenarios[scenario_number]
    print(f"Starting scenario {scenario_number}")
    run_scenario(client, traffic_manager, cam_setup, scenario, scenario_number, folder)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--host", default='127.0.0.1', help="IP of the Carla server host")
    args.add_argument("--port", default=2000, help="Port to connect to the server", type=int)
    args.add_argument("--tm_port", default=8000, type=int, help="Traffic Manager communications port")
    args.add_argument("--cam_setup", default='0,0,0', help="List of camera offsets. To make a stereo camera setup with 0.5 metres distance, do '0,0,0;0.5,0,0'. Each tuple contains x, y and z distances from the default camera position. The camera is always facing in positive z direction. Separate by semicolons (note that you probably need quotes around this argument).")
    args.add_argument("--folder", default="./output", type=str, help="Folder to store output (default is './output')")
    args.add_argument("--scenario_number", default=0, type=int, help="Which scenario to run")
    args = args.parse_args()

    s_num = args.scenario_number 
    folder = Path(args.folder)

    try:
        cams = list()
        for cam_str in args.cam_setup.split(';'):
            x, y, z = [float(v) for v in cam_str.split(',')]
            cams.append((x,y,z))
        
        main(host=args.host, port=args.port, tm_port=args.tm_port, cam_setup=cams, 
             folder=folder, scenario_number=s_num)

    except KeyboardInterrupt:
        pass
    except:
        print("Unexpected error:", sys.exc_info()) 
        print(traceback.format_exc())
        print("You need to reset the CARLA server to ensure the resulting videos will be valid")
    finally:
        print('\nDone!')
