import numpy as np
import argparse
import random
import json
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from tqdm import tqdm
from utils import calculate_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the NuScenes dataset root directory.')
    parser.add_argument("--save_path", type=str, help="Path to save the annotations.")
    parser.add_argument("--cameras", type=list, default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], help="Cameras to use for the dataset.")
    parser.add_argument("--bins", type=list, default=[5, 10, 20, 30, 40, 50], help="Distance bins for the dataset.")
    parser.add_argument("--num_negative_samples", type=int, default=200, help="Number of negative samples")
    parser.add_argument("--question", type=str, default="How many people are there in the image?")
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

    # Keep only the samples with exactly two or three pedestrians in the scene
    two_pedestrians_data = []
    three_pedestrians_data = []

    for sample in tqdm(pedestrian_samples):
        for camera in args.cameras:
            datum = {}
            human_anns = []
            cnt = 0
            cnt2 = 0
            for ann in sample['anns']:
                sample_annotation = nusc.get('sample_annotation', ann)
                if sample_annotation['category_name'] == "human.pedestrian.stroller":
                    continue
                if "human" in sample_annotation["category_name"]  or "bicycle" in sample_annotation["category_name"] or "motorcycle" in sample_annotation["category_name"]:
                    _, boxes, _ = nusc.get_sample_data(sample['data'][camera], box_vis_level=BoxVisibility.ANY, selected_anntokens=[ann])
                    if boxes:
                        cnt2 += 1
                if "human" in sample_annotation["category_name"]:
                    _, boxes, _ = nusc.get_sample_data(sample['data'][camera], box_vis_level=BoxVisibility.ALL, selected_anntokens=[ann])
                    if boxes and sample_annotation['visibility_token'] == '4':
                        human_anns.append(ann)
                        cnt += 1
            if 2 <= cnt == cnt2 <= 3:
                ego_pose_token = nusc.field2token("ego_pose", "timestamp", sample["timestamp"])[0]
                ego_pose = nusc.get("ego_pose", ego_pose_token)
                pedestrian_anns = [nusc.get("sample_annotation", ann) for ann in human_anns]
                sample_data = nusc.get("sample_data", sample['data'][camera])
                camera = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                
                datum["filename"] = sample_data["filename"]
                datum["pedestrians_translation"] = [ann["translation"] for ann in pedestrian_anns]
                datum["camera_translation"] = camera["translation"]
                datum["car_translation"] = ego_pose["translation"]
                datum["car_rotation"] = ego_pose["rotation"]

                if cnt == 2:
                    two_pedestrians_data.append(datum)
                elif cnt == 3:
                    three_pedestrians_data.append(datum)

    # Create the final form of the dataset with two pedestrians
    two_pedestrians_final_data = []
    bins = np.array(args.bins)

    for datum in two_pedestrians_data:
        new_datum = {}
        distance_1 = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["pedestrians_translation"][0], datum["camera_translation"])
        distance_2 = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["pedestrians_translation"][1], datum["camera_translation"])
        avg_distance = (distance_1 + distance_2) / 2
        variance = np.var([distance_1, distance_2])
        if distance_1 < args.bins[-1] + 5 and distance_2 < args.bins[-1] + 5:  # Keep only the samples that have a distance less than max_bin + 5 meters
            new_datum["filename"] = datum["filename"]
            new_datum["question"] = args.question
            new_datum["answer"] = "Two"
            new_datum["distance_1"] = distance_1
            new_datum["distance_2"] = distance_2
            new_datum["avg_distance"] = avg_distance
            new_datum["variance"] = variance
            new_datum["distance_bin"] = int(bins[np.abs(bins - avg_distance).argmin()])
            two_pedestrians_final_data.append(new_datum)

    # Create the final form of the dataset with three pedestrians
    three_pedestrians_final_data = []

    for datum in three_pedestrians_data:
        new_datum = {}
        distance_1 = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["pedestrians_translation"][0], datum["camera_translation"])
        distance_2 = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["pedestrians_translation"][1], datum["camera_translation"])
        distance_3 = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["pedestrians_translation"][2], datum["camera_translation"])
        avg_distance = (distance_1 + distance_2 + distance_3) / 3
        variance = np.var([distance_1, distance_2, distance_3])
        if distance_1 < args.bins[-1] + 5 and distance_2 < args.bins[-1] + 5 and distance_3 < args.bins[-1] + 5:  # Keep only the samples that have a distance less than max_bin + 5 meters
            new_datum["filename"] = datum["filename"]
            new_datum["question"] = args.question
            new_datum["answer"] = "Three"
            new_datum["distance_1"] = distance_1
            new_datum["distance_2"] = distance_2
            new_datum["distance_3"] = distance_3
            new_datum["avg_distance"] = avg_distance
            new_datum["variance"] = variance
            new_datum["distance_bin"] = int(bins[np.abs(bins - avg_distance).argmin()])
            three_pedestrians_final_data.append(new_datum)

    dataset = two_pedestrians_final_data + three_pedestrians_final_data

    # Save the dataset to the specified path
    save_path = f"{args.save_path}/annotations.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
