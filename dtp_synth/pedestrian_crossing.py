import carla
import random
import time
import os
import glob
import json
import argparse

from PIL import Image
from utils import destroy_all_actors, move_walker_forward

# Global variables to store image name, data directory, and image dimensions
image_name = ""
save_path = ""
image_width = 0
image_height = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, help="Map to use for the simulation")
    parser.add_argument("--save_path", type=str, help="Directory to save the data")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--spectate", type=bool, default=False, help="True if you want to spectate the simulation while data is created.")
    parser.add_argument("--vehicle_id", type=int, default=29, help="ID of the vehicle to spawn")
    parser.add_argument("--cam_init_loc", type=str, default="1.5, 0, 1.2", help="Should be adjusted depending on the vehicle used.")
    parser.add_argument("--pedestrian_id", type=int, default=1, help="ID of the pedestrian to spawn")
    parser.add_argument("--image_width", type=int, default=1920, help="Width of the saved image")
    parser.add_argument("--image_height", type=int, default=1080, help="Height of the saved image")
    parser.add_argument("--wait_time", type=float, default=0.5, help="Time to wait between actions")
    parser.add_argument("--port_number", type=int, default=2000, help="Port number for CARLA server connection")
    parser.add_argument("--question", type=str, default="Are there any pedestrians crossing the road?")
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

    # Get the CARLA map and blueprints for vehicles and pedestrians
    vehicle_blueprints = world.get_blueprint_library().filter("*vehicle*")
    walker_blueprints = world.get_blueprint_library().filter("*walker*")

    # Generate waypoints on the map
    waypoints = carla_map.generate_waypoints(1.0)

    # Get the spectator actor
    if args.spectate:
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

    # Keep a count of the number of samples generated
    cnt = 0

    # Main loop to generate samples
    while cnt < args.num_samples:
        print(f"Generating sample {cnt + 1} / {args.num_samples}", end="\r")
        # Set a random weather condition
        random_weather = random.choice(weather_conditions)
        world.set_weather(random_weather)
        time.sleep(args.wait_time)

        # Choose a random waypoint to spawn the vehicle
        random_waypoint = random.choice(waypoints)
        spawn_transform = random_waypoint.transform
        spawn_transform.location.z += 0.5
        spawn_location = spawn_transform.location

        try:
            if args.spectate:
                # Set the spectator a bit higher than the spawn transform looking downwards
                spectator_location = carla.Location(spawn_location.x, spawn_location.y, spawn_location.z + 20)
                spectator_rotation = carla.Rotation(pitch=-90)
                spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
                time.sleep(args.wait_time * 4)

            # Spawn the vehicle and disable physics
            vehicle = world.spawn_actor(vehicle_blueprints[args.vehicle_id], spawn_transform)
            time.sleep(args.wait_time)
            vehicle.set_simulate_physics(False)  # We don't want the vehicle to start moving if it happens to be spawned on a downhill

            # Turn on the high beam lights if it's night
            if random_weather.sun_altitude_angle < 0:
                light_state = carla.VehicleLightState.HighBeam
                vehicle.set_light_state(light_state)
                time.sleep(args.wait_time / 5)
        except:
            # If spawning fails, destroy all actors and continue
            destroy_all_actors(world, args.wait_time)
            continue

        # Get the forward vector of the vehicle
        forward_vector = spawn_transform.get_forward_vector()

        # Create a transform to place the camera on top of the vehicle
        init_loc = args.cam_init_loc.split(",")
        init_loc = [float(loc) for loc in init_loc]
        camera_init_trans = carla.Transform(carla.Location(init_loc[0], init_loc[1], init_loc[2]))

        # Get the camera blueprint and set its resolution
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "3840")  # Setting very high resolution and then resizing within the save_image function gave better quality images.
        camera_bp.set_attribute("image_size_y", "2160")

        # Spawn the camera and attach it to the vehicle
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        time.sleep(args.wait_time)
        camera.listen(save_image)
        camera.stop()  # Running once camera.listen() followed directly by camera.stop() doesn't have enough time to capture an image, but if not used 
                       # the first image captured came blurry sometimes.
        
        # Calculate a perpendicular vector to the forward vector
        perpendicular_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)
        perturbation = random.uniform(-3, 3)
        speed = random.uniform(1.0, 5.0)

        # Randomly choose a direction for the pedestrian to cross
        pedestrian_rotation = spawn_transform.rotation
        if random.random() > 0.5:
            pedestrian_rotation.yaw += 90
            direction = "right"
        else:
            pedestrian_rotation.yaw -= 90
            direction = "left"

        # Define distances for pedestrian spawning
        distances = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0]

        filenames = []
        images_to_delete = 0
        for distance in distances:
            image_taken = False
            final_distance = distance + init_loc[0]

            # Calculate the pedestrian's location
            pedestrian_location = spawn_location + carla.Location(
                x=forward_vector.x * final_distance + perpendicular_vector.x * perturbation,
                y=forward_vector.y * final_distance + perpendicular_vector.y * perturbation,
                z=0.2
            )

            # Check if the pedestrian location is valid (road or shoulder)
            waypoint_driving = carla_map.get_waypoint(pedestrian_location, project_to_road=False, lane_type=carla.LaneType.Driving)
            waypoint_shoulder = carla_map.get_waypoint(pedestrian_location, project_to_road=False, lane_type=carla.LaneType.Shoulder)
            
            if waypoint_driving is None and waypoint_shoulder is None:
                destroy_all_actors(world, args.wait_time)
                break

            # Spawn the pedestrian
            pedestrian_transform = carla.Transform(pedestrian_location, pedestrian_rotation)
            try:
                pedestrian = world.spawn_actor(walker_blueprints[args.pedestrian_id], pedestrian_transform)
                time.sleep(args.wait_time)

                # Move the pedestrian forward
                move_walker_forward(pedestrian, speed=speed)
                time.sleep(args.wait_time)
            except:
                destroy_all_actors(world, args.wait_time)
                break

            # Check if the pedestrian is occluded
            hit_results = world.cast_ray(camera.get_transform().location, pedestrian.get_transform().location)
            allowed_labels = ["Pedestrians", "Car", "NONE"]
            occluded = False
            for hit in hit_results:
                if str(hit.label) not in allowed_labels:
                    occluded = True
                    break

            if occluded:
                destroy_all_actors(world, args.wait_time)
                break

            # Capture the image
            camera.listen(save_image)
            image_taken = True
            images_to_delete += 1
            time.sleep(args.wait_time / 2)
            camera.stop()
            filenames.append(image_name)
            pedestrian.destroy()
            time.sleep(args.wait_time)
        else:
            images_to_delete = 0

        # Delete images if necessary
        if images_to_delete > 0:
            time.sleep(args.wait_time * 4)
            # Get the list of image files in the output directory
            image_files = sorted(glob.glob(f"{args.save_path}/images/*.png"), key=os.path.getmtime)

            # Delete the last "images_to_delete" number of images
            for image_file in image_files[-images_to_delete:]:
                os.remove(image_file)

            filenames.clear()

            continue
        
        # Save the final empty image and annotations
        if image_taken:
            camera.listen(save_image)
            time.sleep(args.wait_time / 2)
            camera.stop()
            filenames.append(image_name)
            cnt += 1

            # Save a json annotation file
            if not os.path.exists(f"{args.save_path}/annotations.json"):
                dataset = {
                    "question": args.question,
                    "answers": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],  # One per distance
                    "distances": [50, 40, 30, 20, 10, 5, None],
                    "samples": []
                }
            else:
                with open(f"{args.save_path}/annotations.json") as f:
                    dataset = json.load(f)

            # Create annotations for the current sample
            annotations = {
                "town": town,
                "file_names": filenames,
                "weather": weather_description.get(random_weather, "Unknown"),
                "direction": direction
            }

            # Append the annotations to the dataset
            dataset["samples"].append(annotations)

            # Save the updated dataset to the JSON file
            with open(f"{args.save_path}/annotations.json", "w") as f:
                json.dump(dataset, f)

        # Destroy all actors after each sample
        destroy_all_actors(world, args.wait_time)


if __name__ == "__main__":
    args = parse_arguments()
    save_path = args.save_path
    image_width = args.image_width
    image_height = args.image_height
    main(args)
    print("\nData generation completed.")
