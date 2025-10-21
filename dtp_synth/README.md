# DTP-Synth

This directory contains the code for generating synthetic data used in DTPQA via CARLA simulator.

## Overview

The repository provides 5 scripts that generate 6 different data categories in DTP-Synth. Note that `pedestrian_crossing.py` generates both Cat.1-Synth and Cat.2-Synth data.

### Command-Line Arguments

**Required arguments:**
- `--map`: CARLA map to use for simulation
- `--data_dir`: Output directory for generated data

**Other important arguments:**
- `--num_samples`: Number of samples to generate (default: 20)
- `--wait_time`: Time interval between actions in seconds (default: 0.5)
  - **Note:** This parameter may require adjustment based on hardware capabilities. Faster hardware can support lower wait times.

Additional script-specific arguments are documented in each source file.

## Data Generation

### Method 1: Docker (Recommended)

Using the pre-configured Docker image is the simplest approach for generating more data similar to those in DTP-Synth.

1. Pull the Docker image:
```bash
   docker pull niktheod/dtp-synth-gen
```

2. Create and run the container:
```bash
   docker run -it --gpus all -p 2000-2002:2000-2002 niktheod/dtp-synth-gen
```

3. Create an output directory inside the container:
```bash
   mkdir /path/to/your/directory
```

4. Navigate to the dtp_synth directory:
```bash
   cd dtp_synth
```

5. Execute the desired data generation script:
```bash
   python pedestrian_crossing.py --map Town01 --data_dir /path/to/your/directory
```

**Important:** The Docker image includes only the base CARLA maps (Town01-07, Town10HD). To use additional maps (Town11, Town12, Town13, Town15), you must manually import them into the container.

### Method 2: Local Installation

Alternatively, you can install CARLA locally and run the scripts directly.

#### Prerequisites
- CARLA 0.9.15
- Python 3.8.16
- Conda (recommended for environment management)

#### Installation Steps

1. Install CARLA 0.9.15 following the [official installation guide](https://carla.readthedocs.io/en/0.9.15/)

2. Clone the repository:
```bash
   git clone https://github.com/D2ICE-Automotive-Research/DTPQA
   cd DTPQA/dtp_synth
```

3. Set up the Python environment:
```bash
   conda create -n carla python=3.8.16 -y
   conda activate carla
   conda install pillow==10.4.0 -y
   pip install carla==0.9.15
```

4. Create an output directory:
```bash
   mkdir /path/to/your/directory
```

5. Run a data generation script:
```bash
   python pedestrian_crossing.py --map Town01 --data_dir /path/to/your/directory
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `pedestrian_crossing.py` | Generates Cat.1-Synth and Cat.2-Synth data |
| `multiple_pedestrians_crossing.py` | Generates Cat.3-Synth data |
| `blinker.py` | Generates Cat.4-Synth data |
| `traffic_lights.py` | Generates Cat.5-Synth data |
| `traffic_signs.py` | Generates Cat.6-Synth data |

## Troubleshooting

- If you encounter any issues (e.g. not all images are saved in the directory), try to adjust the `--wait_time` parameter based on your hardware and see if that solves the problem
- Ensure CARLA server is running before executing generation scripts if running without docker
- Verify GPU drivers are properly configured when using the Docker image with `--gpus all`
