import numpy as np

from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap

def calculate_distance(car_translation, car_rotation, pedestrian_translation, camera_translation):
    car_translation = np.array(car_translation)
    pedestrian_translation = np.array(pedestrian_translation)
    car_rotation = np.array(car_rotation)
    camera_translation = np.array(camera_translation)
    ego_rotation = Quaternion(car_rotation)
    final_translation = ego_rotation.rotate(camera_translation) + car_translation

    distance = np.linalg.norm(pedestrian_translation - final_translation)

    return distance

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def calculate_angle_between_camera_and_pedestrian(camera_rotation, car_rotation, pedestrian_rotation):
    w, x, y, z = pedestrian_rotation
    pedestrian_yaw = quaternion_yaw(Quaternion(w, x, y, z))

    w, x, y, z = car_rotation
    car_yaw = quaternion_yaw(Quaternion(w, x, y, z))

    w, x, y, z = camera_rotation
    camera_yaw = quaternion_yaw(Quaternion(w, x, y, z))

    adjusment = np.pi/2 + camera_yaw

    final_yaw = car_yaw + adjusment

    diff = (final_yaw - pedestrian_yaw)  # Compute difference
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def get_surface_type(nusc, annotation):
    """
    Determines the surface type (road, sidewalk, etc.) of a nuScenes annotation.
    
    Args:
        nusc: The NuScenes API instance.
        annotation: The annotation dictionary containing 'sample_token' and 'translation'.
    
    Returns:
        The surface type as a string (e.g., 'road', 'sidewalk', 'vegetation', 'building', 'other').
    """
    # Extract sample token from the annotation
    sample_token = annotation['sample_token']
    sample = nusc.get('sample', sample_token)
    
    # Retrieve the scene and log to get the map name
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']
    
    # Load the corresponding map
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
    
    # Get the annotation's x, y coordinates
    x, y, _ = annotation['translation']
    
    # Query the map layers at the annotation's coordinates
    layers = nusc_map.layers_on_point(x, y)
    
    return layers
