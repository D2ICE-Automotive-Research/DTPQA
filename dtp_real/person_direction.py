import numpy as np
import argparse
import json
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from tqdm import tqdm
from utils import calculate_distance, calculate_angle_between_camera_and_pedestrian


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the NuScenes dataset root directory.')
    parser.add_argument("--save_path", type=str, help="Path to save the annotations.")
    parser.add_argument("--cameras", type=list, default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], help="List of cameras to use for the dataset.")
    parser.add_argument("--bins", type=list, default=[5, 10, 20, 30, 40, 50], help="Distance bins for the dataset.")
    parser.add_argument("--question", type=str, default="In which direction is the person walking?")
    return parser.parse_args()


def main(args):
    # Initialize NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    # Filter out the data that don't have any humans in the scene
    pedestrian_samples = []
    for sample in nusc.sample:
        for ann in sample['anns']:
            sample_annotation = nusc.get('sample_annotation', ann)
            if sample_annotation['category_name'] == "human.pedestrian.adult":
                pedestrian_samples.append(sample)
                break

    single_pedestrians_data = []

    # Keep only the samples with a single human in the scene across all cameras
    for sample in tqdm(pedestrian_samples):
        for camera in args.cameras:
            datum = {}
            human_ann = None
            cnt = 0
            cnt2 = 0
            for ann in sample['anns']:
                sample_annotation = nusc.get('sample_annotation', ann)
                if sample_annotation['category_name'] == "human.pedestrian.stroller":
                    continue
                if "human" in sample_annotation["category_name"] or "bicycle" in sample_annotation["category_name"] or "motorcycle" in sample_annotation["category_name"]:
                    _, boxes, _ = nusc.get_sample_data(sample['data'][camera], box_vis_level=BoxVisibility.ANY, selected_anntokens=[ann])
                    if boxes:
                        cnt2 += 1
                if "human" in sample_annotation["category_name"]:
                    _, boxes, _ = nusc.get_sample_data(sample['data'][camera], box_vis_level=BoxVisibility.ALL, selected_anntokens=[ann])
                    if boxes and sample_annotation['visibility_token'] == '4':
                        human_ann = sample_annotation
                        cnt += 1
            if cnt == cnt2 == 1:
                sample_data = nusc.get("sample_data", sample['data'][camera])
                ego_pose_token = nusc.field2token("ego_pose", "timestamp", sample["timestamp"])[0]
                ego_pose = nusc.get("ego_pose", ego_pose_token)
                calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

                datum["annotation"] = human_ann
                datum["filename"] = sample_data["filename"]
                datum["camera_translation"] = calibrated_sensor["translation"]
                datum["camera_rotation"] = calibrated_sensor["rotation"]
                datum["car_translation"] = ego_pose["translation"]
                datum["car_rotation"] = ego_pose["rotation"]
                single_pedestrians_data.append(datum)

    # Create the final dataset with the distance between the pedestrian and the camera, the distance bin and the filename
    data_up_to_max_distance = []
    bins = np.array(args.bins)

    for datum in single_pedestrians_data:
        new_datum = {}
        distance = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["annotation"]["translation"], datum["camera_translation"])
        if distance < args.bins[-1] + 5:  # Keep only the samples that have a distance less than max_bin + 5
            new_datum["annotation"] = datum["annotation"]
            new_datum["filename"] = datum["filename"]
            new_datum["car_rotation"] = datum["car_rotation"]
            new_datum["camera_rotation"] = datum["camera_rotation"]
            new_datum["distance"] = distance
            new_datum["distance_bin"] = int(bins[np.abs(bins - distance).argmin()])
            data_up_to_max_distance.append(new_datum)

    # Filter out the humans that aren't walking
    walking_data = []
    for datum in data_up_to_max_distance:
        vel_coord = nusc.box_velocity(datum["annotation"]["token"])
        if vel_coord[0] is not None:
            velocity = np.linalg.norm(vel_coord[:2])
            if velocity > 0.5:
                walking_data.append(datum)

    # Keep only the data where the pedestrian is clearly moving to the left or right from the perspective of the camera
    dataset = []

    for datum in walking_data:
        new_datum = {}
        angle = calculate_angle_between_camera_and_pedestrian(datum["camera_rotation"], datum["car_rotation"], datum["annotation"]["rotation"])
        if 70 < np.degrees(angle) < 110:
            direction = "Right"
        elif -110 < np.degrees(angle) < -70:
            direction = "Left"
        else:
            continue

        new_datum["filename"] = datum["filename"]
        new_datum["question"] = args.question
        new_datum["answer"] = direction
        new_datum["distance"] = datum["distance"]
        new_datum["distance_bin"] = datum["distance_bin"]
        dataset.append(new_datum)

    # Save the dataset to the specified path
    save_path = f"{args.save_path}/annotations.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
