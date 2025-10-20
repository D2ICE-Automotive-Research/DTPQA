import carla
import random
import time
import os
import glob
import json
import argparse

from PIL import Image
from utils import destroy_all_actors


image_name = ""
data_dir = ""
image_width = 0
image_height = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, help="Map to use for the simulation")
    parser.add_argument("--data_dir", type=str, help="Directory to save the data")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--spectate", type=bool, default=False, help="True if you want to spectate the simulation while data is created.")
    parser.add_argument("--vehicle_id", type=int, default=29, help="ID of the vehicle to spawn")
    parser.add_argument("--cam_init_loc", type=str, default="1.5, 0, 1.2", help="Should be adjusted depending on the vehicle used.")
    parser.add_argument("--pedestrian_id", type=int, default=1, help="ID of the pedestrian to spawn")
    parser.add_argument("--image_width", type=int, default=1920, help="Width of the saved image")
    parser.add_argument("--image_height", type=int, default=1080, help="Height of the saved image")
    parser.add_argument("--wait_time", type=float, default=0.5, help="Time to wait between actions")
    parser.add_argument("--port_number", type=int, default=2000, help="Port number for CARLA server connection")
    parser.add_argument("--question", type=str, default="What's the colour of the traffic light?")
    return parser.parse_args()

def save_image(image):
    """
    Save the captured image to disk and resize it.
    
    Args:
        image: carla.Image - The image captured by the camera.
    """
    global image_name
    global data_dir
    global image_width
    global image_height

    current_time = int(time.time() * 1000)
    image_path = f"{data_dir}/images/{current_time}.png"
    image_name = f"{current_time}.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save_to_disk(image_path)

    with Image.open(image_path) as img:
        img = img.resize((image_width, image_height), Image.LANCZOS)
        img.save(image_path)


def get_waypoints_within_radius(all_waypoints, location, radius):
    """
    Returns a list of waypoints within the specified radius from the given location.
    
    Args:
        location (carla.Location): The center location.
        radius (float): The radius (in meters) within which to search for waypoints.
        world (carla.World): The CARLA world object.
        
    Returns:
        list: List of carla.Waypoint objects within the given radius.
    """
    
    # Filter waypoints based on distance from the given location.
    nearby_waypoints = [wp for wp in all_waypoints if location.distance(wp.transform.location) <= radius]
    
    return nearby_waypoints


def angle_difference(angle1, angle2):
    """
    Computes the smallest difference between two angles in degrees.
    """
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def select_waypoint_by_distance_and_yaw(waypoints, reference_location, min_distance, max_distance, target_yaw, min_yaw_diff=25):
    """
    Selects a waypoint from a list of waypoints that are within a specified distance range from a reference location,
    and then returns the one whose yaw has the smallest divergence from a target yaw.

    Args:
        waypoints (list): List of carla.Waypoint objects.
        reference_location (carla.Location): The reference location from which to measure the distance.
        min_distance (float): The minimum allowed distance from the reference location.
        max_distance (float): The maximum allowed distance from the reference location.
        target_yaw (float): The target yaw (in degrees) to compare against.
        
    Returns:
        carla.Waypoint or None: The waypoint that meets the criteria or None if no waypoint is found.
    """
    # Filter waypoints based on the distance from the reference location.
    filtered_waypoints = [
        wp for wp in waypoints
        if min_distance <= reference_location.distance(wp.transform.location) <= max_distance
    ]
    
    if not filtered_waypoints:
        print("No filtered waypoints found")
        return None

    # Select the waypoint with the smallest yaw difference relative to the target yaw.
    best_wp = None
    
    for wp in filtered_waypoints:
        wp_yaw = wp.transform.rotation.yaw
        diff = angle_difference(wp_yaw, target_yaw)
        if diff < min_yaw_diff:
            min_yaw_diff = diff
            best_wp = wp

    return best_wp


def main(args):
    global image_name

    # Town-specific configuration parameters
    # [camera adjustment distance, whether to align with waypoints, list of distances to capture from]
    town_info = {"Town01": [2.5, False, [50.0, 40.0, 30.0, 20.0, 10.0]],
                 "Town02": [2.5, False, [50.0, 40.0, 30.0, 20.0, 10.0]],
                 "Town03": [5.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town04": [5.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town05": [5.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town06": [5.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town07": [5.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town10HD": [6.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town12": [4.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town13": [4.0, True, [50.0, 40.0, 30.0, 20.0]],
                 "Town15": [4.0, True, [50.0, 40.0, 30.0, 20.0]],}
    
    # Connect to the CARLA server
    client = carla.Client("localhost", args.port_number)
    client.set_timeout(args.wait_time * 120)  # Some maps need a lot of time to load so it's better to increase the timeout
    world = client.get_world()

    town = args.map

    # Get town-specific parameters
    adjustment = town_info[town][0]
    align = town_info[town][1]
    distances = town_info[town][2]

    spectator = world.get_spectator()

    # Load the specified map
    client.load_world(town)

    # Get all road waypoints for vehicle placement
    waypoints = world.get_map().generate_waypoints(1.0)
    driving_waypoints = [wp for wp in waypoints if wp.lane_type == carla.LaneType.Driving]

    # Get vehicle blueprints for spawning
    vehicle_blueprints = world.get_blueprint_library().filter("*vehicle*")

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

    # Get all traffic lights in the world
    actors = world.get_actors()
    traffic_lights = actors.filter("traffic.traffic_light")
    traffic_lights = list(traffic_lights)
    traffic_lights_states = ["Green", "Yellow", "Red"]

    # Adjust number of samples if fewer traffic lights are available
    if len(traffic_lights) < args.num_samples:
            num_samples = len(traffic_lights)
    else:
        num_samples = args.num_samples

    # Main simulation loop
    while cnt < num_samples:
        # Set a random weather condition
        random_weather = random.choice(weather_conditions)
        world.set_weather(random_weather)
        time.sleep(args.wait_time)

        if not traffic_lights:
            print("No traffic lights left")
            break

        # Select a random traffic light and remove it from the list to avoid duplicates
        traffic_light = random.choice(traffic_lights)
        traffic_lights.remove(traffic_light)
        traffic_light_transform = traffic_light.get_transform()

        # Adjust traffic light position for better camera view
        traffic_light_transform.location.z += 0.5
        # Get the forward vector of the traffic light
        forward_vector = traffic_light_transform.get_forward_vector()

        # Position camera at a fixed distance from the traffic light in the opposite direction
        traffic_light_transform.location.x -= forward_vector.x * adjustment
        traffic_light_transform.location.y -= forward_vector.y * adjustment
        traffic_light_transform.rotation.yaw -= 90
        traffic_light_transform_tuple = (traffic_light_transform.location.x, traffic_light_transform.location.y, traffic_light_transform.location.z,
                                         traffic_light_transform.rotation.pitch, traffic_light_transform.rotation.yaw, traffic_light_transform.rotation.roll)
        # Get the new forward vector after rotation
        forward_vector = traffic_light_transform.get_forward_vector()

        # Choose a random traffic light state (Green, Yellow, Red)
        random_state = random.choice(traffic_lights_states)

        filenames = []
        precise_distances = []
        images_to_delete = 0
        # Loop through multiple distances to capture images
        for distance in distances:
            image_taken = False

            # Create spawn transform for the vehicle at the current distance
            spawn_transform = carla.Transform(carla.Location(traffic_light_transform_tuple[0], traffic_light_transform_tuple[1], traffic_light_transform_tuple[2]),
                                              carla.Rotation(traffic_light_transform_tuple[3], traffic_light_transform_tuple[4], traffic_light_transform_tuple[5]))
            spawn_transform.location.x -= forward_vector.x * distance
            spawn_transform.location.y -= forward_vector.y * distance

            # If alignment is enabled, find the nearest valid road waypoint
            if align:
                nearby_waypoints = get_waypoints_within_radius(driving_waypoints, spawn_transform.location, 5)

                wp = select_waypoint_by_distance_and_yaw(nearby_waypoints, traffic_light.get_transform().location, distance-2, distance+2, spawn_transform.rotation.yaw)
                if wp is None:
                    destroy_all_actors(world, args.wait_time)
                    break

                # Find the rightmost lane
                right_lane = wp.get_right_lane()
                if right_lane is not None and right_lane.lane_type == carla.LaneType.Driving:
                    while right_lane is not None and right_lane.lane_type == carla.LaneType.Driving:
                        wp = right_lane
                        right_lane = wp.get_right_lane()
                        
                spawn_transform = wp.transform
                spawn_transform.location.z += 0.5

            try:
                vehicle = world.spawn_actor(vehicle_blueprints[args.vehicle_id], spawn_transform)
                time.sleep(args.wait_time)
                vehicle.set_simulate_physics(False)  # We don't want the vehicle to start moving if it happens to be spawned on a downhill

                # Turn on the high beam lights if it's night
                if random_weather.sun_altitude_angle < 0:
                    light_state = carla.VehicleLightState.HighBeam
                    vehicle.set_light_state(light_state)
                    time.sleep(args.wait_time / 5)
            except:
                destroy_all_actors(world, args.wait_time)
                break

            # Create camera transform relative to the vehicle
            init_loc = args.cam_init_loc.split(",")
            init_loc = [float(loc) for loc in init_loc]
            camera_init_trans = carla.Transform(carla.Location(init_loc[0], init_loc[1], init_loc[2]))

            # Create camera blueprint with high resolution
            camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "3840")
            camera_bp.set_attribute("image_size_y", "2160")

            # Spawn camera attached to the vehicle
            camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
            time.sleep(args.wait_time)
            camera.listen(save_image)   
            camera.stop()

            # Calculate actual distance from camera to traffic light
            precise_distance = camera.get_location().distance(traffic_light.get_transform().location)

            # Ensure the distance is within acceptable range
            if not (distance - 5) < precise_distance < (distance + 5):
                destroy_all_actors(world, args.wait_time)
                break

            precise_distances.append(precise_distance)

            if args.spectate:
                # Move spectator to camera position for visualization
                spectator = world.get_spectator()
                spectator.set_transform(camera.get_transform())

            # Check if traffic light is occluded by disallowed objects
            hit_results = world.cast_ray(camera.get_transform().location, traffic_light.get_transform().location)
            allowed_labels = ["Car", "Fences", "Sidewalks", "NONE", "Roads", "Poles", "Ground"]
            occluded = False
            for hit in hit_results:
                if str(hit.label) not in allowed_labels:
                    occluded = True
                    break
            if occluded:
                destroy_all_actors(world, args.wait_time)
                break

            # Set the traffic light to the randomly chosen state
            if random_state == "Green":
                traffic_light.set_state(carla.TrafficLightState.Green)
            elif random_state == "Yellow":
                traffic_light.set_state(carla.TrafficLightState.Yellow)
            elif random_state == "Red":
                traffic_light.set_state(carla.TrafficLightState.Red)
            time.sleep(args.wait_time / 5)
            
            # Capture the image
            camera.listen(save_image)
            image_taken = True
            images_to_delete += 1
            time.sleep(args.wait_time / 2)
            camera.stop()

            filenames.append(image_name)
            time.sleep(args.wait_time)
            destroy_all_actors(world, args.wait_time)
        else:
            # This executes if the for loop completes without breaking
            images_to_delete = 0

        # If any images need to be deleted (due to failure in the process)
        if images_to_delete > 0:
            time.sleep(2)
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
                    "distances": [50, 40, 30, 20, 10],
                    "samples": []
                }
            else:
                with open(f"{args.data_dir}/annotations.json") as f:
                    dataset = json.load(f)

            annotations = {
                "town": town,
                "file_names": filenames,
                "precise_distances": precise_distances,
                "weather": weather_description.get(random_weather, "Unknown"),
                "answer": random_state
            }

            dataset["samples"].append(annotations)

            with open(f"{args.data_dir}/annotations.json", "w") as f:
                json.dump(dataset, f)

        # Clean up any remaining actors before next iteration
        destroy_all_actors(world, args.wait_time)


if __name__ == "__main__":
    args = parse_arguments()
    data_dir = args.data_dir
    image_width = args.image_width
    image_height = args.image_height
    main(args)
