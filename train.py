import numpy as np
from tqdm import tqdm
from env import environment
import copy
    
class QLearning:
    def __init__(self, env):
        self.env = env
        self.init_hyperparams()
        self.writer = SummaryWriter()
        self.Q = np.zeros((self.env.lines*self.env.columns, self.env.num_actions))
        self.old = np.zeros((1, 4))
        self.state = []
        self.actions = ['r', 'u', 'l', 'd']
        self.path = []
        self.total_reward = 0

    def init_hyperparams(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.05

    def get_state_index(self, state):
        return state[0] * (self.env.lines-1) + state[1]
    
    def learn(self, epoch=1000):
        for i in tqdm(range(epoch)):
            state = copy.deepcopy(self.env.reset())
            episode_reward = 0

            while True:
                self.env.is_carrot()
                self.env.is_barrier()
                state_index = self.get_state_index(state)
            
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(low=0, high=self.env.num_actions)
                else:
                    action = np.argmax(self.Q[state_index])
               
                pre_current = copy.deepcopy(self.env.current)
                next_state, reward, done = self.env.step(action, pre_current)
                next_state_index = self.get_state_index(next_state)

                # if current can't move, reset position
                if self.env.check_moved(self.env.current):
                    self.Q[state_index, action] -= 100
                    self.env.graph = copy.deepcopy(self.env.graph_backup)
                    self.env.current = copy.deepcopy(self.env.reset())
                    continue

                # if position not change, increase the punishment
                if state_index == next_state_index:
                    reward -= 100

                episode_reward += reward
                
                #  Q value will be zero when it arrive goal
                # minus 1 to eval
                self.Q[state_index, action] = self.Q[state_index, action] + self.alpha * (
                    reward + (self.gamma*np.max(self.Q[next_state_index])) - self.Q[state_index, action]) - 0.1
                state = copy.deepcopy(next_state)

                if done: 
                    break

    def eval_policy(self):
        self.env.graph = copy.deepcopy(self.env.graph_backup)
        self.env.current = copy.deepcopy(self.env.start_position)
        total_reward = 0

        while True:
            current_index = self.get_state_index(self.env.current)
            action = np.argmax(self.Q[current_index])
            if self.Q[current_index, action] == 0:
                self.Q[current_index, action] = float('-inf')
                continue
            pre_current = copy.deepcopy(self.env.current)
            state, reward, done = self.env.step(action, pre_current)
            
            self.path.append(self.actions[action])
            total_reward += reward
            
            if done:
                break
        return total_reward
    
if __name__ == '__main__':
    G = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 3, 1, 1, 1],
         [1, 1, 1, 0, 0, 1, 1, 1],
         [1, 1, 0, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 1, 4, 1],
         [1, 1, 1, 1, 2, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],])

    env = Environment(G)
    model = QLearning(env)
    model.learn(1)
    _ = model.eval_policy()
    print(model.env.graph)
    print(f"reward: {_}")
    print(f"shortest path: {model.path}")
