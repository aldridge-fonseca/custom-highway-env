# Environment design

`CustomHighwayEnv` is a custom `highway-env` task focused on hazard navigation. It keeps the standard high-level driving actions from `highway-env`, but changes the road layout, traffic behavior, hazards, and reward terms.

## Road layout

The environment uses a straight five-lane highway with a 625-meter route. Episodes end when the ego vehicle crashes, leaves the road, reaches the end, or exceeds the time limit.

## Hazards

Each episode can include several hazards:

- A lane closure, represented by a 40-meter obstacle in one of the middle lanes.
- A stalled vehicle, placed in a lane different from the closure.
- An emergency vehicle, which drives faster than normal traffic and is marked in the observation dictionary.

The randomized placement makes the task less tied to one memorized path.

## Traffic behavior

Traffic vehicles use an IDM-style behavior model. `YieldingIDMVehicle` extends the standard behavior by checking for approaching emergency vehicles behind it. When an emergency vehicle is nearby, traffic can make lane changes or slow down to avoid blocking it.

## Observation space

The default observation is LiDAR-based:

```python
"observation": {
    "type": "LidarObservation",
    "cells": 128,
    "maximum_range": 120,
    "normalize": True,
}
```

The custom environment also monkey-patches vehicle and obstacle dictionaries to include an `is_emergency` flag. This lets observation-based agents distinguish regular traffic from emergency vehicles.

## Action space

The environment uses `DiscreteMetaAction`, the standard high-level action abstraction from `highway-env`. The action space is discrete with five actions.

## Reward function

The reward is a weighted sum of safety, speed, progress, completion, lane-change, and yielding terms.

| Term | Weight | Description |
|---|---:|---|
| `collision_reward` | -1.5 | Penalizes crashes. |
| `high_speed_reward` | 0.3 | Rewards speeds in the target range of 20-28 m/s. |
| `progress_reward` | 0.4 | Rewards forward progress along the road. |
| `success_reward` | 1.0 | Rewards reaching the end without crashing. |
| `lane_change_reward` | -0.01 | Adds a small cost for lane changes. |
| `yielding_reward` | 0.5 | Rewards the ego vehicle for not blocking nearby emergency vehicles. |

The final reward is multiplied by the on-road indicator, so going off-road removes positive reward contributions.

## Termination

An episode terminates when:

- the ego vehicle crashes,
- the ego vehicle leaves the road,
- the ego vehicle reaches the end of the road.

An episode is truncated when the configured time limit is reached.
