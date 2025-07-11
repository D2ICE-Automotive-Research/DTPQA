import os
import time
import carla

from PIL import Image

def destroy_all_actors(world, wait_time):
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

def move_walker_forward(walker, speed):
    """
    Move a walker forward in their current facing direction.
    
    Args:
        walker: carla.Walker - The walker actor.
        speed: float - Walking speed in m/s.
    """
    # Get the walker's current transform
    transform = walker.get_transform()
    
    # Create a control object for the walker
    control = carla.WalkerControl()
    
    # Set the direction to the walker's forward vector
    control.direction = transform.get_forward_vector()
    
    # Set the walking speed
    control.speed = speed
    
    # Apply the control
    walker.apply_control(control)
