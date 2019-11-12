# animals
Reinforcement learning project for improving animal tracking prediction and behavioral analysis.

## data

Place Turkey Vulture data in the *data* directory, with the filename `turkey_vultures.csv`

## model
 https://github.com/MatthewJA/Inverse-Reinforcement-Learning
- f: Feature Matrix (n by d)
- a: actions (int)
- gamma: discount factor (float)
- A: transition probabilities (n by a by n)
- zeta: trajectories (t by l by 2)
- e: epochs
- r: learning rate

**TODO (as of 11/4):**
- Define buckets for each feature
    - Zach's notes: Why not define bucket count for each feature, which gives us d for free and is more flexible for future users? This has been done but it is changeable
- Create feature matrix (size n by d where n: # of states, d: dimensions of states)
    - This has been done now
- **NEW ITEM**: Find min and max values for every feature
- Decide on ranges for hyper-parameters (discount, epochs, learning rate)
- Linearly interpolate turkey data
    - This should be doable in the process of creating trajectories
- Iterate over each bird:
    - Iterate over each location:
        - Define state action pair (state i, action j)
        - keep track of transitions: Every time bird went from state i to state
          k using action j (probability determined as number to state k with
          action j / total number that left i)
**TODO (as of 11/11):**
- Fix distance from coast (Seth)
- Population density/city distance (Seth)
- Have all features output results to npy file for pxs (Alex)
- Fix linear interpolation (Zach + Leah)
- Finish feature class (Zach)
- Try bringing in IRL model to our work (Leah)
     - Passing in our data to the guy's IRL library
- Learn/Understand IRL model and bring questions/needed clarifications/etc (link above; All)

## Features

- Land vs. Water (2 buckets)
- Distance from Coast (10 buckets)
- Elevation (10 buckets)
