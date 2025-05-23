# on\_track\_sysid

**This is a ROS 2 package for end-to-end vehicle system identification on a scaled autonomous racecar (simulation *or* Jetson-based car).**

To incorporate your own on-board computer, simply enter the model name when prompted for `self.racecar_version`

It automates

1. **Data Collection** of longitudinal/lateral speed, yaw-rate and steering input.
2. **Neural-network aided identification** of the lateral tyre model (Pacejka C-α coefficients).
3. **Lookup-table generation** for downstream controllers.

> 📚 This work is inspired by:
>
> Dikici, O., Ghignone, E., Hu, C., Baumann, N., Xie, L., Carron, A., Magno, M., & Corno, M. (2024). *Learning-Based On-Track System Identification for Scaled Autonomous Racing in Under a Minute*. arXiv:2411.17508. [https://arxiv.org/abs/2411.17508](https://arxiv.org/abs/2411.17508)

---

## Quick start

```bash
# In your ROS 2 workspace
cd ~/ros2_ws/src

git clone https://github.com/akumar1807/on_track_sysid.git

# Install Python dependencies (inside the same Python env as ROS 2)
python3 -m pip install -r on_track_sysid/requirements.txt

# Build the workspace
cd ..
colcon build --symlink-install
source install/setup.bash
```

---

## Console scripts (nodes / helpers)

| Script (call with `ros2 run on_track_sysid <name>`) | ROS 2 node name      | Purpose                                                                                            |
| --------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| **`collect_data`**                                  | `data_collector`     | Log data **in simulation** (F1TENTH / Autodrive). Writes a CSV in *src/on\_track\_sysid/data/*.    |
| **`jetson_collect`**                                | `jetson_data_logger` | Log data on the physical Jetson-based car.                                                         |
| **`with_data_sys_id`**                              | *(no node)*          | Offline training **after** data have been collected. Reads the CSV and trains the NN + Pacejka ID. |
| **`jetson_sys_id`**                                 | *(no node)*          | Same as above but loads the Jetson dataset by default.                                             |
| **`ontrack`**                                       | `ontrack`            | One-shot pipeline **on the Jetson**: collects, trains, generates LUT, then exits.                  |

> ℹ️  All scripts prompt for confirmation before overwriting datasets/models.

### Example

```bash
# 1️⃣ Collect 30 s of data in simulation (Total data collection time can be changed in `nn_params.yaml`)
ros2 run on_track_sysid collect_data

#   (follow the interactive prompts, drive the car, save the CSV)

# 2️⃣ Train the model offline
ros2 run on_track_sysid with_data_sys_id
```

---

## Topics used

| Data source                | Simulation topic                                     | Jetson topic                                                             |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Longitudinal speed (m s⁻¹) | `/autodrive/f1tenth_1/speed` (`std_msgs/Float32`)    | `/odom` → `twist.twist.linear.x`                                         |
| Lateral speed (m s⁻¹)      | *assumed 0*                                          | `/odom` → `twist.twist.linear.y`                                         |
| Yaw-rate (rad s⁻¹)         | `/autodrive/f1tenth_1/imu` (`sensor_msgs/Imu`)       | `/odom` → `twist.twist.angular.z`                                        |
| Steering input (rad)       | `/autodrive/f1tenth_1/steering` (`std_msgs/Float32`) | `/commands/servo/position` (`std_msgs/Float64`) → linear mapping in code |

If your setup uses different topic names, remap them at launch time, e.g.

```bash
ros2 run on_track_sysid collect_data --ros-args -r __ns:=/mycar -r /speed:=/mycar/speed_mps
```

---

## Parameters

All high-level knobs live in **`params/nn_params.yaml`**:

```yaml
data_collection_duration: 30      # seconds logged per run
num_of_iterations: 6              # outer refinement loops
num_of_epochs: 5000               # training epochs per loop
lr: 0.0005                        # learning rate for Adam
weight_decay: 0.001               # L2 regularisation
```

Tune these as needed (e.g. shorter collection for quick tests).

---

## File layout

```
on_track_sysid/
├── on_track_sysid/              # Python package
│   ├── collect_data_for_sys_id.py  # → collect_data
│   ├── collect_data_jetson.py     # → jetson_collect
│   ├── on_track_jetson.py         # → ontrack
│   ├── with_data_sys_id.py        # → with_data_sys_id
│   ├── jetson_sys_id.py           # → jetson_sys_id
│   ├── train_model.py             # NN + tyre ID core
│   └── …                          # helpers, NN, plotting, LUT gen
├── params/
│   └── nn_params.yaml
├── models/                       # saved *.pth and LUTs appear here
├── data/                         # generated CSV logs
├── setup.py
└── package.xml
```

---

## Dependencies

### ROS 2 packages

* `rclpy`
* `std_msgs`, `nav_msgs`, `geometry_msgs`

### Python libraries (pip / requirements.txt)

* **NumPy**
* **PyTorch** (≥ 2.0)
* **SciPy**
* **tqdm**

> On Jetson you may want to install the NVIDIA-optimised PyTorch wheel.

---

## Development & contribution

1. Make sure `pre-commit` passes `ament_flake8` and `ament_pep257`.
2. Submit a PR against `main` with a descriptive title.

Bug reports / feature requests → [GitHub Issues](https://github.com/akumar1807/on_track_sysid/issues).

---

## License

See `LICENSE` (TBD).

---

### Acknowledgements

Based on the ETH Zurich dynamic modelling workflow; adapted and extended for F1TENTH and Jetson-Nano hardware.
