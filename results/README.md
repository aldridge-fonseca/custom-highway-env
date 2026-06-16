# Results

This folder contains the generated results for the custom highway environment.

## Included plots

| ID | Result | File |
|---|---|---|
| 13 | Learning curve | `plots/ID13_Custom_Env_Learning_Curve.png` |
| 14 | 500-episode performance test | `plots/ID14_Custom_Env_Performance_Test.png` |

## Training setup

The included PPO run used:

- `MlpPolicy`
- 300,000 training timesteps
- LiDAR observation with 128 cells
- deterministic evaluation over 500 episodes

The trained checkpoint is stored in `../models/ID13_Custom_Env_Trained_Model.zip`.
