import carla
import random
import time
import os
import glob
import json
import pickle
import argparse
import numpy as np

from PIL import Image
from utils import destroy_all_actors


draft_dir = ""
image_width = 0
image_height = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, help="Map to use for the simulation")
    parser.add_argument("--data_dir", type=str, help="Directory to save the data")
    parser.add_argument("--blinker_annotations", type=str, help="Path to the blinker annotations file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--spectate", type=bool, default=False, help="True if you want to spectate the simulation while data is created.")
    parser.add_argument("--vehicle_id", type=int, default=29, help="ID of the vehicle to spawn")
    parser.add_argument("--cam_init_loc", type=str, default="1.5, 0, 1.2", help="Should be adjusted depending on the vehicle used.")
    parser.add_argument("--truck_id", type=int, default=18, help="ID of the truck to spawn")
    parser.add_argument("--image_width", type=int, default=1920, help="Width of the saved image")
    parser.add_argument("--image_height", type=int, default=1080, help="Height of the saved image")
    parser.add_argument("--wait_time", type=float, default=0.5, help="Time to wait between actions")
    parser.add_argument("--port_number", type=int, default=2000, help="Port number for CARLA server connection")
    parser.add_argument("--question", type=str, default="Which of the truck's blinkers, if any, is on?")
    return parser.parse_args()


def save_image(image):
    """
    Save the captured image to disk and resize it.
    
    Args:
        image: carla.Image - The image captured by the camera.
    """
    global draft_dir
    global image_width
    global image_height

    current_time = int(time.time() * 1000)
    image_path = f'{draft_dir}/{current_time}.png'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save_to_disk(image_path)

    with Image.open(image_path) as img:
        img = img.resize((image_width, image_height), Image.LANCZOS)
        img.save(image_path)


def get_cropped_rectangle(image_path, left, right, top, bottom):
    """
    Opens an image, crops a rectangular region based on specified horizontal and vertical limits,
    and calculates the average pixel value of that region.

    Parameters:
        image_path (str): The file path to the image.
        left (int): The x-coordinate of the left boundary of the rectangle.
        right (int): The x-coordinate of the right boundary of the rectangle.
        top (int): The y-coordinate of the top boundary of the rectangle.
        bottom (int): The y-coordinate of the bottom boundary of the rectangle.
    """
    # Open the image using Pillow (PIL)
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Verify that the specified crop rectangle is within the image bounds
    if left < 0 or top < 0 or right > img_width or bottom > img_height:
        raise ValueError("The defined rectangle is out of the image boundaries.")
    if left >= right or top >= bottom:
        raise ValueError("Invalid rectangle dimensions: 'left' must be less than 'right' and 'top' must be less than 'bottom'.")

    # Define the crop box (left, top, right, bottom)
    crop_box = (left, top, right, bottom)
    
    # Crop the image to the defined rectangular region
    cropped_image = image.crop(crop_box)

    # Convert the cropped image to a NumPy array (ignoring the alpha channel if present)
    cropped_array = np.array(cropped_image)[..., :3]

    # Calculate the average pixel value
    avg_pixel_value = np.mean(cropped_array)

    return avg_pixel_value


def check_for_good_images(data_dir, draft_dir, blinker_annotations, distance, blinker_on):
    """
    Filters and selects the best image based on blinker brightness criteria.

    Args:
        data_dir (str): Directory where the selected image will be saved.
        draft_dir (str): Directory containing draft images to be evaluated.
        blinker_annotations (dict): Dictionary containing blinker position and brightness threshold information.
        distance (int): Distance parameter used to fetch specific blinker annotations.
        blinker_on (str): Indicates which blinker is on ("left" or "right").

    Returns:
        str or None: The name of the selected image if one is found, otherwise None.

    Raises:
        OSError: If there is an issue reading an image file.
        SyntaxError: If there is an issue with the image file format.
    """
    draft_images = glob.glob(f"{draft_dir}/*.png")

    qualified_images = []

    blinker_off = "right" if blinker_on == "left" else "left"

    for path in draft_images:
        try:
            avg_pixel_value_on = get_cropped_rectangle(path, *blinker_annotations["blinker_position"][str(distance)][blinker_on])
            avg_pixel_value_off = get_cropped_rectangle(path, *blinker_annotations["blinker_position"][str(distance)][blinker_off])
            if (avg_pixel_value_on >= blinker_annotations["brightness_threshold"][str(distance)]["minimum"]) \
            and (avg_pixel_value_on - avg_pixel_value_off >= blinker_annotations["brightness_threshold"][str(distance)]["minimum_difference"]):
                qualified_images.append((path, avg_pixel_value_on))
        except (OSError, SyntaxError):
            continue

    if len(qualified_images) == 0:
        return None
    elif len(qualified_images) == 1:
        image_path = qualified_images[0][0]
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(f"{data_dir}/images", image_name)
        os.makedirs(data_dir, exist_ok=True)
        os.rename(image_path, new_image_path)
        return image_name
    else:
        # Choose the image with the highest avg_pixel_value_on
        best_image = max(qualified_images, key=lambda x: x[1])
        image_path = best_image[0]
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(f"{data_dir}/images", image_name)
        os.makedirs(f"{data_dir}/images", exist_ok=True)
        os.rename(image_path, new_image_path)
        return image_name
        

def main(args):
    with open(args.blinker_annotations, "rb") as f:
        blinker_annotations = pickle.load(f)

    global image_name
    # Connect to the server and retrieve the world object
    client = carla.Client("localhost", args.port_number)
    client.set_timeout(args.wait_time * 120)  # Some maps need a lot of time to load so it's better to increase the timeout
    world = client.get_world()

    town = args.map

    # Load the specified map
    client.load_world(town)
    carla_map = world.get_map()

    # Get blueprints for vehicles
    vehicle_blueprints = world.get_blueprint_library().filter("*vehicle*")

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
        camera.stop()

        # Define distances for pedestrian spawning
        distances = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
        truck_length_x = 3.967855215072632

        # Choose a random blinker
        if random.random() > 0.5:
            blinker = "left"
        else:
            blinker = "right"

        filenames = []
        images_to_delete = 0
        for distance in distances:
            image_taken = False
            final_distance = distance + init_loc[0] + truck_length_x

            # Calculate the location of the truck
            truck_location = spawn_location + carla.Location(
                x=forward_vector.x * final_distance,
                y=forward_vector.y * final_distance,
                z=0.2
            )

            # Get the closest waypoint to the truck location and check if it's a driving lane
            waypoint_driving = carla_map.get_waypoint(truck_location, project_to_road=False, lane_type=carla.LaneType.Driving)
            if waypoint_driving is None:
                destroy_all_actors(world, args.wait_time)
                break

            truck_transform = carla.Transform(truck_location, spawn_transform.rotation)
            try:
                # Spawn the truck
                truck = world.spawn_actor(vehicle_blueprints[args.truck_id], truck_transform)
                time.sleep(args.wait_time * 4)
                truck.set_simulate_physics(False)
                default_light_state = vehicle.get_light_state()
                truck.set_light_state(default_light_state)
                
                # Turn on the blinker
                if blinker == "left":
                    truck.set_light_state(carla.VehicleLightState.LeftBlinker)
                elif blinker == "right":
                    truck.set_light_state(carla.VehicleLightState.RightBlinker)
            except RuntimeError:
                destroy_all_actors(world, args.wait_time)
                break

            # Check if the truck is occluded
            hit_results = world.cast_ray(camera.get_transform().location, truck.get_transform().location)
            allowed_labels = ["Car", "NONE", "Roads", "Truck"]
            occluded = False
            for hit in hit_results:
                if str(hit.label) not in allowed_labels:
                    occluded = True
                    break

            if occluded:
                destroy_all_actors(world, args.wait_time)
                break
            
            filename = None
            count = 0
            while filename is None:
                count += 1
                # Set the blinker state for the truck
                if blinker == "left":
                    truck.set_light_state(carla.VehicleLightState.LeftBlinker)
                elif blinker == "right":
                    truck.set_light_state(carla.VehicleLightState.RightBlinker)
                
                # Delete all files in the draft directory
                for file in glob.glob(f"{draft_dir}/*.png"):
                    os.remove(file)
                
                # Break the loop if no good image is found after 5 attempts
                if count > 5:
                    break
                
                # Capture and save the image
                camera.listen(save_image)
                time.sleep(args.wait_time * 24)  # Wait for the camera to capture multiple images
                camera.stop()
                
                # Check for good images based on blinker brightness criteria
                filename = check_for_good_images(args.data_dir, draft_dir, blinker_annotations, distance, blinker)
            
            # If no good image is found, print a message and break the loop
            if filename is None:
                print(f"No good images found. Weather: {weather_description.get(random_weather)}, Town: {args.map}")
                break
            
            # Append the filename to the list of filenames
            filenames.append(filename)
            images_to_delete += 1
            image_taken = True
            
            # Destroy the truck actor
            truck.destroy()
            time.sleep(args.wait_time)
        else:
            images_to_delete = 0

        # Delete the last "images_to_delete" number of images
        if images_to_delete > 0:
            time.sleep(args.wait_time * 4)
            # Get the list of image files in the output directory
            image_files = sorted(glob.glob(f"{args.data_dir}/images/*.png"), key=os.path.getmtime)

            # Delete the last "images_to_delete" number of images
            for image_file in image_files[-images_to_delete:]:
                os.remove(image_file)

            filenames.clear()

            continue

        if image_taken:
            cnt += 1

            # Save a json annotation file
            if not os.path.exists(f"{args.data_dir}/annotations.json"):
                dataset = {
                    "question": args.question,
                    "distances": [50, 40, 30, 20, 10, 5],
                    "samples": []
                }
            else:
                with open(f"{args.data_dir}/annotations.json") as f:
                    dataset = json.load(f)

            annotations = {
                "town": town,
                "file_names": filenames,
                "weather": weather_description.get(random_weather, "Unknown"),
                "blinker_on": blinker
            }

            dataset["samples"].append(annotations)

            with open(f"{args.data_dir}/annotations.json", "w") as f:
                json.dump(dataset, f)

        destroy_all_actors(world, args.wait_time)


if __name__ == "__main__":
    args = parse_arguments()
    draft_dir = args.data_dir + "/draft_images"
    image_width = args.image_width
    image_height = args.image_height
    main(args)
