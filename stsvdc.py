"""
    Creates standard videos 
"""

import argparse
from pathlib import Path 
import sys
import random
import time 
from dataclasses import dataclass

import carla
from carla import VehicleLightState as vls

def loc_dist(a, b):
    return (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2

@dataclass
class Scenario:
    map:str
    cam_pos:tuple
    cam_dir:tuple
    length:int

def default_scenarios():
    scenrarios = list()
    s = Scenario(map='Town01', cam_pos=(101.38188171386719,183.1450958251953,7.839078903198242), cam_dir=(-24.54534149169922,130.83407592773438,4.599098247126676e-05), length=2000)
    scenrarios.append(s)

    s = Scenario(map='Town01', cam_pos=(323.6024475097656,185.66769409179688,10.069890975952148), cam_dir=(-32.938194274902344,55.12174987792969,3.051888779737055e-05), length=2000)
    scenrarios.append(s)

    s = Scenario(map='Town10HD', cam_pos=(-51.66246032714844,154.75738525390625,13.823832511901855), cam_dir=(-27.47414779663086,-86.13775634765625,0.00011643866309896111), length=2000)
    scenrarios.append(s)

    return scenrarios

def main(host:str, port:int, tm_port:int, cam_setup:list):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager(tm_port)

    scenarios = default_scenarios()
    for s_num, scenario in enumerate(scenarios):
        print(f"Starting scenario {s_num+1} / {len(scenarios)}")
        run_scenario(client, traffic_manager, scenario)

def run_scenario(client, traffic_manager, scenario:Scenario):
    # Some parameters
    number_of_vehicles=80
    number_of_pedestrians=42

    # Random seed
    seed = 420
    random.seed(seed)

    # Setup world and traffic manager
    world = client.get_world()
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.set_random_device_seed(seed)

    # Set timing/sync settings
    settings = world.get_settings()
    traffic_manager.set_synchronous_mode(True)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.0075 # timing for physics simulations, needs to be below 0.01
    world.apply_settings(settings)

    # Setup blueprints for road user spawning
    blueprints_vehicles = world.get_blueprint_library().filter('vehicle.*')
    blueprints_pedestrians = world.get_blueprint_library().filter('walker.pedestrian.*')
    blueprints_vehicles = sorted(blueprints_vehicles, key=lambda bp: bp.id)
    blueprints_pedestrians = sorted(blueprints_pedestrians, key=lambda bp: bp.id)

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Hack-import objects for spawning
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # Spawn vehicles
    vehicles = list()
    batch = list()
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(blueprints_vehicles)
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
            loc.z += 0.5 # To prevent collisions with the ground
            
            all_spawns = spawn_points + walker_spawns
            distances = [loc_dist(previous.location, loc) for previous in all_spawns]
            if min(distances) < 10.0:
                # To precent collisions with other road users 
                loc = None 
        spawn_point.location = loc
        
        walker_spawns.append(spawn_point) 
    
    batch = list()
    pedestrian_speeds = list()
    for spawn_point in walker_spawns:
        walker_bp = random.choice(blueprints_pedestrians)
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
            pedestrians.append({'id': res.actor_id, 'speed': float(pedestrian_speeds[i])})

    print("Actors spawned!")

    batch = list()
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for pedestrian in pedestrians:
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), pedestrian["id"]))
    results = client.apply_batch_sync(batch, True)
    for i, res in enumerate(results):
        if res.error:
            raise ValueError(res.error)
        else:
            pedestrians[i]['con'] = res.actor_id
    
    world.tick()

    world.set_pedestrians_cross_factor(0.05)
    
    # Start moving pedestrians
    controller_actors = world.get_actors([p['con'] for p in pedestrians])
    for pedestrian, actor in zip(pedestrians, controller_actors):
        actor.start()
        actor.go_to_location(world.get_random_location_from_navigation())
        actor.set_max_speed(pedestrian['speed'])

    print(f"Spawned {len(pedestrians)} pedestrians")

    traffic_manager.global_percentage_speed_difference(30.0)

    print("Stabilizing world...")
    for _ in range(100):
        world.tick()
        time.sleep(0.005)

    print("World stabilized")

    # Spawn camera(s)
    




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--host", default='127.0.0.1', help="IP of the Carla server host")
    args.add_argument("--port", default=2000, help="Port to connect to the server", type=int)
    args.add_argument("--tm_port", default=8000, type=int, help="Traffic Manager communications port")
    args.add_argument("--cam_setup", default=[(0,0,0)], help="List of camera offsets. To make a stereo camera setup with 0.5 metres distance, do [(0,0,0), (0.5,0,0)]. Each tuple contains x, y and z distances from the default camera position. The camera is always facing in positive z direction. ")
    args = args.parse_args()

    try:
        main(host=args.host, port=args.port, tm_port=args.tm_port, cam_setup=args.cam_setup)
    except KeyboardInterrupt:
        pass
    except:
        print("Unexpected error:", sys.exc_info()) 
        print("You need to reset the CARLA server to ensure the resulting videos will be valid")
    finally:
        print('\nDone!')