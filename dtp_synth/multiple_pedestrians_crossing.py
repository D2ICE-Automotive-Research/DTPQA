import carla
import random
import time
import os
import glob
import json
import argparse

from PIL import Image
from utils import move_walker_forward


# Global variables to store image name, data directory, and image dimensions
image_name = ""
save_path = ""
image_width = 0
image_height = 0


# Custom exception for respawning vehicles
class RespawnVehicleException(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, help="Map to use for the simulation")
    parser.add_argument("--save_path", type=str, help="Directory to save the data")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--spectate", type=bool, default=False, help="True if you want to spectate the simulation while data is created.")
    parser.add_argument("--vehicle_id", type=int, default=29, help="ID of the vehicle to spawn")
    parser.add_argument("--cam_init_loc", type=str, default="1.5, 0, 1.2", help="Should be adjusted depending on the vehicle used.")
    parser.add_argument("--pedestrian_ids", type=list, default=[1, 5, 6, 16], help="ID of the pedestrians to spawn")
    parser.add_argument("--image_width", type=int, default=1920, help="Width of the saved image")
    parser.add_argument("--image_height", type=int, default=1080, help="Height of the saved image")
    parser.add_argument("--wait_time", type=float, default=0.5, help="Time to wait between actions")
    parser.add_argument("--port_number", type=int, default=2000, help="Port number for CARLA server connection")
    parser.add_argument("--question", type=str, default="How many pedestrians are crossing the road?")
    return parser.parse_args()

def save_image(image):
    """
    Save the captured image to disk and resize it.
    
    Args:
        image: carla.Image - The image captured by the camera.
    """
    global image_name
    global save_path
    global image_width
    global image_height

    current_time = int(time.time() * 1000)
    image_path = f"{save_path}/images/{current_time}.png"
    image_name = f"{current_time}.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save_to_disk(image_path)

    with Image.open(image_path) as img:
        img = img.resize((image_width, image_height), Image.LANCZOS)
        img.save(image_path)


# Function to destroy all actors in the world
def destroy_all_actors(world, wait_time, exception_raise=True):
    """
    Destroy all actors in the CARLA world.
    
    Args:
        world: carla.World - The CARLA world object.
    """
    actors = world.get_actors()

    cameras = actors.filter("sensor.camera.rgb")
    for camera in cameras:
        camera.destroy()
        time.sleep(wait_time)

    vehicles = actors.filter("vehicle.*")
    for vehicle in vehicles:
        vehicle.destroy()
        time.sleep(wait_time)

    walkers = actors.filter("walker.*")
    for walker in walkers:
        walker.destroy()
        time.sleep(wait_time)

    if exception_raise:
        raise RespawnVehicleException


def main(args):
    global image_name
    # Connect to the server and retrieve the world object
    client = carla.Client("localhost", args.port_number)
    client.set_timeout(args.wait_time * 120)  # Some maps need a lot of time to load so it's better to increase the timeout
    world = client.get_world()

    town = args.map

    # Load the specified map
    client.load_world(town)
    carla_map = world.get_map()

    # Get blueprints for vehicles and walkers
    vehicle_blueprints = world.get_blueprint_library().filter("*vehicle*")
    walker_blueprints = world.get_blueprint_library().filter("*walker*")

    # Generate waypoints on the map
    waypoints = carla_map.generate_waypoints(1.0)

    # Get the spectator actor
    spectator = world.get_spectator()

    # Define different weather conditions
    weather_conditions = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainSunset,
        carla.WeatherParameters.ClearNight,
        carla.WeatherParameters.CloudyNight,
        carla.WeatherParameters.WetNight,
        carla.WeatherParameters.WetCloudyNight,
        carla.WeatherParameters.MidRainyNight,
        carla.WeatherParameters.HardRainNight,
        carla.WeatherParameters.SoftRainNight,
        carla.WeatherParameters.DustStorm
    ]

    # Map weather conditions to descriptions so it can be saved in a json file
    weather_description = {
        carla.WeatherParameters.ClearNoon: "Clear Noon",
        carla.WeatherParameters.CloudyNoon: "Cloudy Noon",
        carla.WeatherParameters.WetNoon: "Wet Noon",
        carla.WeatherParameters.WetCloudyNoon: "Wet Cloudy Noon",
        carla.WeatherParameters.MidRainyNoon: "Mid Rainy Noon",
        carla.WeatherParameters.HardRainNoon: "Hard Rain Noon",
        carla.WeatherParameters.SoftRainNoon: "Soft Rain Noon",
        carla.WeatherParameters.ClearSunset: "Clear Sunset",
        carla.WeatherParameters.CloudySunset: "Cloudy Sunset",
        carla.WeatherParameters.WetSunset: "Wet Sunset",
        carla.WeatherParameters.WetCloudySunset: "Wet Cloudy Sunset",
        carla.WeatherParameters.MidRainSunset: "Mid Rain Sunset",
        carla.WeatherParameters.HardRainSunset: "Hard Rain Sunset",
        carla.WeatherParameters.SoftRainSunset: "Soft Rain Sunset",
        carla.WeatherParameters.ClearNight: "Clear Night",
        carla.WeatherParameters.CloudyNight: "Cloudy Night",
        carla.WeatherParameters.WetNight: "Wet Night",
        carla.WeatherParameters.WetCloudyNight: "Wet Cloudy Night",
        carla.WeatherParameters.MidRainyNight: "Mid Rainy Night",
        carla.WeatherParameters.HardRainNight: "Hard Rain Night",
        carla.WeatherParameters.SoftRainNight: "Soft Rain Night",
        carla.WeatherParameters.DustStorm: "Dust Storm"
    }

    cnt = 0

    # Main loop to generate samples
    while cnt < args.num_samples:
        print(f"Generating sample {cnt + 1} / {args.num_samples}", end="\r")
        try:
            # Set a random weather condition
            random_weather = random.choice(weather_conditions)
            world.set_weather(random_weather)
            time.sleep(args.wait_time)

            # Choose a random waypoint to spawn the vehicle
            random_waypoint = random.choice(waypoints)
            spawn_transform = random_waypoint.transform
            spawn_transform.location.z += 0.5
            spawn_location = spawn_transform.location

            # Create a copy of spawn_transform
            spawn_transform_copy = carla.Transform(spawn_transform.location, spawn_transform.rotation)

            try:
                if args.spectate:
                    # Set the spectator a bit higher than the spawn transform looking downwards
                    spectator_location = carla.Location(spawn_location.x, spawn_location.y, spawn_location.z + 20)
                    spectator_rotation = carla.Rotation(pitch=-90)
                    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
                    time.sleep(args.wait_time * 4)

                # Spawn the vehicle
                vehicle = world.spawn_actor(vehicle_blueprints[args.vehicle_id], spawn_transform)
                time.sleep(args.wait_time)
                vehicle.set_simulate_physics(False)

                # Turn on the high beam lights if it's night
                if random_weather.sun_altitude_angle < 0:
                    light_state = carla.VehicleLightState.HighBeam
                    vehicle.set_light_state(light_state)
                    time.sleep(args.wait_time / 5)
            except:
                # Destroy all actors if spawning fails
                destroy_all_actors(world, args.wait_time)
                continue

            forward_vector = spawn_transform.get_forward_vector()

            # Create a transform to place the camera on top of the vehicle
            init_loc = args.cam_init_loc.split(",")
            init_loc = [float(loc) for loc in init_loc]
            camera_init_trans = carla.Transform(carla.Location(init_loc[0], init_loc[1], init_loc[2]))

            # Get the camera blueprint and set its resolution
            camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "3840")
            camera_bp.set_attribute("image_size_y", "2160")

            # Spawn the camera and attach it to the vehicle
            camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
            time.sleep(args.wait_time)
            camera.listen(save_image)
            camera.stop()  # Running once camera.listen() followed directly by camera.stop() doesn't have enough time to capture an image, but if not used 
                           # the first image captured came blurry sometimes.
            
            # Calculate a perpendicular vector to the forward vector
            perpendicular_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)

            # Define the two different directions for the pedestrians to spawn
            pedestrian_rotation_left = spawn_transform.rotation
            pedestrian_rotation_left.yaw -= 90
            pedestrian_rotation_right = spawn_transform_copy.rotation
            pedestrian_rotation_right.yaw += 90
            pedestrian_rotations = [pedestrian_rotation_left, pedestrian_rotation_right]

            # Define distances for pedestrian spawning
            distances = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
            pedestrian_indices = args.pedestrian_ids

            # Initialize list to store filenames and counter for images to delete
            filenames = []
            images_to_delete = 0

            # Iterate over defined distances
            for distance in distances:
                final_distance = distance + init_loc[0]

                # Iterate over number of pedestrians to spawn
                for num in range(1, 1+len(pedestrian_indices)):
                    pedestrians = []

                    # Spawn pedestrians
                    for i in range(num):
                        attempts = 0
                        max_attempts = 5 if images_to_delete > 0 else 1  # If some images are already generated try a few more times before giving up
                        spawned = False

                        # Attempt to spawn pedestrian within max_attempts
                        while not spawned and attempts < max_attempts:
                            attempts += 1
                            ped_distance = random.gauss(final_distance, 1.2)
                            perturbation = random.uniform(-3, 3)
                            pedestrian_location = spawn_location + carla.Location(
                                x=forward_vector.x * ped_distance + perpendicular_vector.x * perturbation,
                                y=forward_vector.y * ped_distance + perpendicular_vector.y * perturbation,
                                z=0.2
                            )

                            # Check if the location is on the road or shoulder
                            waypoint_driving = carla_map.get_waypoint(pedestrian_location, project_to_road=False, lane_type=carla.LaneType.Driving)
                            waypoint_shoulder = carla_map.get_waypoint(pedestrian_location, project_to_road=False, lane_type=carla.LaneType.Shoulder)
                            
                            if waypoint_driving is None and waypoint_shoulder is None and attempts == max_attempts:
                                destroy_all_actors(world, args.wait_time)
                            elif waypoint_driving is None and waypoint_shoulder is None:
                                continue
                            
                            # If we have a valid location, spawn the pedestrian
                            pedestrian_rotation = random.choice(pedestrian_rotations)
                            pedestrian_transform = carla.Transform(pedestrian_location, pedestrian_rotation)
                            try:
                                pedestrian = world.spawn_actor(walker_blueprints[pedestrian_indices[i]], pedestrian_transform)
                                time.sleep(args.wait_time)
                                pedestrians.append(pedestrian)
                                spawned = True
                            except:
                                if attempts == max_attempts:
                                    destroy_all_actors(world, args.wait_time)
                                else:
                                    continue
                            
                    # Move each pedestrian forward with a random speed
                    for pedestrian in pedestrians:
                        speed = random.uniform(1.0, 5.0)
                        move_walker_forward(pedestrian, speed)
                    time.sleep(args.wait_time)

                    # Check for occlusions
                    results = []
                    for pedestrian in pedestrians:
                        hit_results = world.cast_ray(camera.get_transform().location, pedestrian.get_transform().location)
                        results.append(hit_results)

                    allowed_labels = ["Pedestrians", "Car", "NONE"]
                    for result in results:
                        occluded = False
                        for hit in result:
                            if str(hit.label) not in allowed_labels:
                                occluded = True
                                break

                        if occluded:
                            destroy_all_actors(world, args.wait_time)
                            

                    # Capture image and update filenames list
                    camera.listen(save_image)
                    images_to_delete += 1
                    time.sleep(args.wait_time / 2)
                    camera.stop()
                    filenames.append(image_name)

                    # Destroy pedestrians after capturing
                    for pedestrian in pedestrians:
                        pedestrian.destroy()
                    time.sleep(args.wait_time)

            
            # Start listening to the camera and save the image
            camera.listen(save_image)
            time.sleep(args.wait_time / 2)
            camera.stop()
            filenames.append(image_name)
            cnt += 1

            # Save a json annotation file
            if not os.path.exists(f"{args.save_path}/annotations.json"):
                dataset = {
                    "question": args.question,
                    "answers": ["One", "Two", "Three", "Four", "One", "Two", "Three", "Four", "One", "Two", "Three", "Four", "One", "Two", "Three", "Four", "One", "Two", "Three", "Four", "One", "Two", "Three", "Four", "Zero"],  # One per distance
                    "distances": [50, 50, 50, 50, 40, 40, 40, 40, 30, 30, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10, 5, 5, 5, 5, None],
                    "samples": []
                }
            else:
                with open(f"{args.save_path}/annotations.json") as f:
                    dataset = json.load(f)

            # Create annotations for the current sample
            annotations = {
                "town": town,
                "file_names": filenames,
                "weather": weather_description.get(random_weather, "Unknown")
            }

            # Append the annotations to the dataset
            dataset["samples"].append(annotations)

            # Save the updated dataset to the annotations file
            with open(f"{args.save_path}/annotations.json", "w") as f:
                json.dump(dataset, f)

            # Destroy all actors in the world without raising an exception
            destroy_all_actors(world, args.wait_time, exception_raise=False)
        except RespawnVehicleException:
            if images_to_delete > 0:
                time.sleep(args.wait_time * 4)
                # Get the list of image files in the output directory
                image_files = sorted(glob.glob(f"{args.save_path}/images/*.png"), key=os.path.getmtime)

                if filenames[-1] in image_files[-1]:
                    # Delete the last "images_to_delete" number of images
                    for image_file in image_files[-images_to_delete:]:
                        os.remove(image_file)

                filenames.clear()
                continue


if __name__ == "__main__":
    args = parse_arguments()
    save_path = args.save_path
    image_width = args.image_width
    image_height = args.image_height
    main(args)
    print("\nData generation completed.")
