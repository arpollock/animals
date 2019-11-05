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
- Create feature matrix (size n by d where n: # of states, d: dimensions of states)
- Decide on ranges for hyper-parameters (discount, epochs, learning rate)
- Linearly interpolate turkey data
- Iterate over each bird:
    - Iterate over each location:
        - Define state action pair (state i, action j)
        - keep track of transitions: Every time bird went from state i to state
          k using action j (probability determined as number to state k with
          action j / total number that left i)
