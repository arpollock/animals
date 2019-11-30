import random
import numpy as np


class environment:
    def __init__(self, reward_map, starting_loc, ending_loc):
        self.reward_map = reward_map
        self.height, self.width = reward_map.shape
        self.n_states = self.width * self.height
        self.n_actions = 4
        self.starting_state = self.get_state(starting_loc)
        self.current_state = self.starting_state
        self.terminal_state = self.get_state(ending_loc)

    def step(self, action):
        '''action 0: up one (state - 1)
           action 1: right one (state + h)
           action 2: down one (state + 1)
           action 3: left one (state - h)'''

        change = max(self.height * (action % 2), 1)
        if action % 3 == 0:
            change = -change
        x, y = self.get_coord(self.current_state)
        reward = 0
        if ((x == 0 and action == 3) or (x == self.width - 1 and action == 1)
            or (y == 0 and action == 0) or (y == self.height - 1
                                            and action == 2)):
            reward = -10
        else:
            self.current_state += change
            x, y = self.get_coord(self.current_state)
            reward = self.reward_map[y][x] - 1
        if self.current_state == self.terminal_state:
            reward = 1000

        return [self.current_state, reward,
                self.current_state == self.terminal_state]

    def get_state(self, loc):
        return loc[0] * self.height + loc[1]

    def get_coord(self, state):
        x = state // self.height
        y = state - (x * self.height)
        return (x, y)

    def reset(self):
        self.current_state = self.starting_state
        return self.current_state


def calculate_new_q_val(q_table, state, action, reward, next_state, alpha, gamma):
    """Calculate the updated Q table value for a particular state and action given the necessary parameters

    Args:
        q_table (np.array): The Q table
        state (int): The current state of the simulation's index in the Q table
        action (int): The current action's index in the Q table
        reward (float): The returned reward value from the environment
        next_state (int): The next state of the simulation's index in the Q table (Based on the environment)
        alpha (float): The learning rate
        gamma (float): The discount rate

    Returns:
        float: The updated action-value expectation for the state and action
    """
    return (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]))


def initialize_q_table(env):
    return np.zeros((env.n_states, env.n_actions))


def select_action(q_row, method, epsilon=0.5):
    """Select the appropriate action given a Q table row for the state and a
    chosen method

    Args:
        q_row (np.array): The row from the Q table to utilize
        method (str): The method to use, either "random" or "epsilon"
        epsilon (float, optional): Defaults to 0.5. The epsilon value to use
        for epislon-greed action selection

    Raises:
        NameError: If method specified is not supported

    Returns:
        int: The index of the action to apply
    """
    if method not in ["random", "epsilon"]:
        raise NameError("Undefined method.")
    e = np.random.rand()

    if method == "random" or e < epsilon:
        action_index = np.random.randint(q_row.shape[0])
    else:
        action_index = np.argmax(q_row)
    return action_index


def train_sim(env, params, n=100):
    """Train a simulation on an environment and return its Q table

    Args:
        env (gym.envs): The environment to train in
        params (dict): The parameters needed to train the simulation: method (for action selection), epsilon, alpha, gamma
        n (int, optional): Defaults to 100. The number of simulations to run for training

    Returns:
        np.array: The trained Q table from the simulation
    """
    my_q = initialize_q_table(env)

    for i in range(n):
        current_state = env.reset()
        done = False
        step = 0

        while not done:
            # Get the next action based on current state
            # Step through the environment with the selected action
            # Update the qtable

            next_action = select_action(my_q[current_state], params["method"], params["epsilon"])
            next_state, reward, done = env.step(next_action)
            my_q[current_state][next_action] = calculate_new_q_val(my_q, current_state, next_action, reward, next_state, params["alpha"], params["gamma"])
            # print(current_state, next_state)
            # Prep for next iteration
            current_state = next_state
            step += 1
            # print(step)

        if (i+1) % 10 == 0:
            print(f"Simulation #{i+1:,} complete.")

    return my_q


def test_sim(env, q_table, n=100, render=False):
    """Test an environment using a pre-trained Q table

    Args:
        env (gym.envs): The environment to test
        q_table (np.array): The pretrained Q table
        n (int, optional): Defaults to 100. The number of test iterations to run
        render (bool, optional): Defaults to False. Whether to display a rendering of the environment

    Returns:
        np.array: Array of length n with each value being the cumulative reward achieved in the simulation
    """
    rewards = []

    for i in range(n):
        current_state = env.reset()

        tot_reward = 0
        done = False
        step = 0

        while not done:

            # Determine the best action
            # Step through the environment

            action = select_action(q_table[current_state], "epsilon", 0)
            current_state, reward, done = env.step(action)

            tot_reward += reward
            step += 1
            if render:
                print(f"Simulation: {i + 1}")
                print(f"Step: {step}")
                print(f"Current State: {current_state}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Total rewards: {tot_reward}")
            if step == 200:
                print("Agent got stuck. Quitting...")
                break

        rewards.append(tot_reward)

    return np.array(rewards)


epsilon1_params = {
    "method": "epsilon",
    "epsilon": 0.4,
    "alpha": 0.1,
    "gamma": 0.5
}

epsilon2_params = {
    "method": "epsilon",
    "epsilon": 0.5,
    "alpha": 0.1,
    "gamma": 0.5
}

# filename = input("Enter filename: ")
filename = 'rewards_1_50.npy'

reward_map = np.load(filename, allow_pickle=True)
size = int((filename.split('_')[-1]).split('.')[0])
reward_map = np.reshape(reward_map, (size, size), order='F')
env = environment(reward_map, (0, 0), (49, 49))

n = 10000
epsilon1_q = train_sim(env, epsilon1_params, n)
# epsilon2_q = train_sim(env, epsilon2_params, n)
np.save(epsilon1_q, 'q_table.npy')

epsilon1_rewards = test_sim(env, epsilon1_q, 10, render=True)
