"""
    Creates standard videos 
"""

import argparse 
from pathlib import Path 
import sys
import random
import time 

import carla
from carla import VehicleLightState as vls

def main(host:str, port:int, tm_port:int):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    
    # Some parameters
    number_of_vehicles=100
    number_of_pedestrians=10

    # Random seed
    seed = 420
    random.seed(seed)

    # Setup world and traffic manager
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(tm_port)
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
        
    batch = []
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

    while True:
        world.tick()
        time.sleep(0.005)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--host", default='127.0.0.1', help="IP of the Carla server host")
    args.add_argument("--port", default=2000, help="Port to connect to the server", type=int)
    args.add_argument("--tm_port", default=8000, type=int, help="Traffic Manager communications port")
    args = args.parse_args()

    try:
        main(host=args.host, port=args.port, tm_port=args.tm_port)
    except KeyboardInterrupt:
        pass
    except:
        print("Unexpected error:", sys.exc_info()) 
        print("You need to reset the CARLA server to ensure the resulting videos will be valid")
    finally:
        print('\nDone!')