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
- <s> Create feature matrix (size n by d where n: # of states, d: dimensions of states) </s>
- <s> Find min and max values for every feature </s>
- Decide on ranges for hyper-parameters (discount, epochs, learning rate)
- <s> Linearly interpolate turkey data
    - This should be doable in the process of creating trajectories </s>
- <s> Iterate over each bird:
    - Iterate over each location:
        - Define state action pair (state i, action j)
        - keep track of transitions: Every time bird went from state i to state
          k using action j (probability determined as number to state k with
          action j / total number that left i) </s>


**TODO (as of 11/11):**
- <s> Fix distance from coast (Seth) </s>
- <s> Population density/city distance (Seth) </s>
- <s> Have all features output results to npy file for pxs (Alex) </s>
- <s> Fix linear interpolation (Zach + Leah) </s>
- <s> Finish feature class (Zach) </s>
- <s> Try bringing in IRL model to our work (Leah)
     - Passing in our data to the guy's IRL library </s>
- Learn/Understand IRL model and bring questions/needed clarifications/etc (link above; All)

**TODO (as of 11/18):**
- <s> Allow for selection of image area to use (Seth/Zach) </s>
- Get started on project deliverables
    - Slides (Alex)
- Start on poster for Computing Innovation Fair (if we can do it... checking with Saad) (Seth, Leah)
- Create the CLI (Zach)
- <s> See if different hyperparameters can help speed up the model (Leah) </s>

**TODO (as of 11/23):**
- Create and Finish CLI
- Finish Slides
- Get reward for 50x50 map
- Test out larger on iMac

## Features

- Land vs. Water (2 buckets)
- Distance from Coast (10 buckets)
- Elevation (10 buckets)
