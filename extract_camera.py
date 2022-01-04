"""
    Simple script to get the position and orientation of camera written to the terminal. 
    This script was used to create the default scenarios.
"""

import carla
from carla import VehicleLightState as vls
import argparse
import time 
import random 

from utocs import loc_dist, Pedestrian

def main():
    host = '127.0.0.1'
    port = 2000
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    args = argparse.ArgumentParser()
    args.add_argument("--map", type=str, default="Town10HD")
    args = args.parse_args()
    map_name = args.map

    client.load_world(map_name)
    time.sleep(1.0)

    world = client.get_world()
    spawn_road_users(client, world)

    old_pos = None 
    def_length = 3000
    while True:
        transform = world.get_spectator().get_transform()
        pos = transform.location
        rot = transform.rotation
        if old_pos and loc_dist(pos, old_pos) > 0.001:
            print(f"Scenario(map='{map_name}', cam_pos=({pos.x},{pos.y},{pos.z}), cam_dir=({rot.pitch},{rot.yaw},{rot.roll}), length={def_length})")
        old_pos = pos

def spawn_road_users(client, world, tm_port=8000, number_of_vehicles=80, 
                     number_of_pedestrians=30):
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    bps_vehicles = world.get_blueprint_library().filter('vehicle.*')
    bps_pedestrians = world.get_blueprint_library().filter('walker.pedestrian.*')
    bps_vehicles = sorted(bps_vehicles, key=lambda bp: bp.id)
    bps_pedestrians = sorted(bps_pedestrians, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

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

if __name__=="__main__":
    main()