# DTP-Real

This directory contains the code for generating **real-world data** for DTPQA using the **nuScenes** dataset. It provides four scripts, each corresponding to a different type of data in DTP-Real. These scripts extract annotations from nuScenes and organize them according to DTPQA's specifications.

## Command-Line Arguments

**Required arguments for all scripts:**
- `--dataroot`: Path to the root directory of the nuScenes dataset.  
- `--save_path`: Directory where the generated annotations will be saved. 

Additional script-specific arguments are documented in the respective source files.

## Data Generation Steps

1. **Download the nuScenes dataset**: [https://nuscenes.org/nuscenes#download](https://nuscenes.org/nuscenes#download)  

2. **Clone this repository**:
```bash
  git clone https://github.com/D2ICE-Automotive-Research/DTPQA
  cd DTPQA/dtp_real
```

3. **Set up the Python environment**:
```bash
   conda create -n dtpqa-real python=3.11 -y
   conda activate dtpqa-real
   pip install numpy==1.26.4
   pip install nuscenes-devkit==1.2.0

```

4. **Create a directory for saving annotations**:
```bash
   mkdir /path/to/your/directory
```

5. **Run a script to generate annotations** (example for person in the scene):
```bash
   python person_in_the_scene.py --dataroot /path/to/nuScenes/root/directory --save_path /path/to/your/directory
```
Repeat step 5 for other scripts to generate different types of DTP-Real data.
