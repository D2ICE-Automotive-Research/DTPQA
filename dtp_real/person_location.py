import numpy as np
import argparse
import json
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from tqdm import tqdm
from utils import calculate_distance, get_surface_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, help='Path to the NuScenes dataset root directory.')
    parser.add_argument("--save_path", type=str, help="Path to save the annotations.")
    parser.add_argument("--cameras", type=list, default=["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"], help="Cameras to use for the dataset.")
    parser.add_argument("--bins", type=list, default=[5, 10, 20, 30, 40, 50], help="Distance bins for the dataset.")
    parser.add_argument("--num_negative_samples", type=int, default=200, help="Number of negative samples")
    parser.add_argument("--question", type=str, default="Where in the image is the person located?")
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

    # Keep only the samples with a single human in the scene
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
    dataset = []
    bins = np.array(args.bins)

    for datum in single_pedestrians_data:
        new_datum = {}
        distance = calculate_distance(datum["car_translation"], datum["car_rotation"], datum["annotation"]["translation"], datum["camera_translation"])
        if distance < args.bins[-1] + 5:  # Keep only the samples that have a distance less max_bin + 5
            new_datum["filename"] = datum["filename"]
            new_datum["question"] = args.question

            layers = get_surface_type(nusc, datum["annotation"])
            if layers["ped_crossing"]:
                new_datum["answer"] = "crossing"
            elif layers["walkway"]:
                new_datum["answer"] = "sidewalk"
            elif layers["drivable_area"]:
                new_datum["answer"] = "road"
            else:
                continue 

            new_datum["distance"] = distance
            new_datum["distance_bin"] = int(bins[np.abs(bins - distance).argmin()])
                
            dataset.append(new_datum)

    

    # Save the dataset to the specified path
    save_path = f"{args.save_path}/annotations.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
