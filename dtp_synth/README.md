# DTP-Synth

This directory contains the code for generating synthetic data used in DTPQA via CARLA simulator. It provides 5 scripts that generate 6 different data categories in DTP-Synth. Note that `pedestrian_crossing.py` generates both Cat.1-Synth and Cat.2-Synth data.

## Command-Line Arguments

**Required arguments for all scripts:**
- `--map`: CARLA map to use for simulation
- `--save_path`: Directory where the generated data will be saved.

**Other important arguments for all scripts:**
- `--num_samples`: Number of samples to generate (default: 20)
- `--wait_time`: Time interval between certain actions in seconds (default: 0.5)
  - **Note:** This parameter may require adjustment based on hardware capabilities. Faster hardware can support lower wait times.

Additional script-specific arguments are documented in each source file.

## Data Generation Steps

### Method 1: Docker (Recommended)

Using the pre-configured Docker image is the simplest approach for generating more data similar to those in DTP-Synth.

1. **Pull the Docker image**:
```bash
docker pull niktheod/dtp-synth-gen:1.0
```

2. **Create and run the container**:
```bash
docker run --gpus all -p 2000-2002:2000-2002 --name dtp-synth-gen-cont niktheod/dtp-synth-gen:1.0 ./CarlaUE4.sh -RenderOffScreen
```

3. **Open a new terminal and access the container**:
```bash
docker exec -it dtp-synth-gen-cont /bin/bash
```

4. **Activate the Conda environment and navigate to the workspace**:
```bash
conda activate dtp-synth
cd /workspace/dtp_synth
```

5. **Execute the desired data generation script** (example for pedestrian crossing):
```bash
python pedestrian_crossing.py --map Town01 --save_path /workspace/your_directory
```

Repeat step 5 for other scripts to generate different types of DTP-Synth data.

**Note:** If the simulator crashes, simply restart the container and repeat steps 3–5.

**Important:** The Docker image includes only the base CARLA maps (Town01-05, Town10HD). To use additional maps (Town06, Town07, Town11, Town12, Town13, Town15), you must manually import them into the container.

### Method 2: Local Installation

Alternatively, you can install CARLA locally and run the scripts directly.

#### Installation Steps

1. **Install CARLA 0.9.15** following the [official installation guide](https://carla.readthedocs.io/en/0.9.15/)

2. **Clone this repository**:
```bash
git clone https://github.com/D2ICE-Automotive-Research/DTPQA
cd DTPQA/dtp_synth
```

3. **Set up the Python environment**:
```bash
conda create -n carla python=3.8.16 -y
conda activate carla
conda install pillow==10.4.0 -y
pip install carla==0.9.15
```

4. **Create a directory for saving data**:
```bash
mkdir /path/to/your/directory
```

5. **Run CARLA**:
```bash
/path/to/CARLA/root/directory/CarlaUE4.sh
```

6. **Run a data generation script**:
```bash
python pedestrian_crossing.py --map Town01 --save_path /path/to/your/directory
```
Repeat step 6 for other scripts to generate different types of DTP-Synth data.

**Important:** If the simulator crashes just restart it and keep generating data.
