import numpy as np
from torch.utils.tensorboard import SummaryWriter
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
            self.state = copy.deepcopy(self.env.reset())
            episode_reward = 0

            while True:
                self.env.is_carrot()
                self.env.is_barrier()
                state_index = self.get_state_index(self.state)
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(low=0, high=self.env.num_actions)
                else:
                    action = np.argmax(self.Q[state_index])
                
                self.env.pre_current = copy.deepcopy(self.env.current)
                next_state, reward, done = self.env.step(action, self.env.pre_current)
                next_state_index = self.get_state_index(next_state)
                episode_reward += reward
                self.Q[state_index, action] = self.Q[state_index, action] + self.alpha * (
                    reward + np.max(self.Q[next_state_index]) - self.Q[state_index, action]
                )

                self.state = copy.deepcopy(next_state)
                if done or self.env.check_moved(self.env.current):
                    self.total_reward = copy.deepcopy(episode_reward)
                    episode_reward = 0
                    self.env.graph = copy.deepcopy(self.env.graph_backup)
                    self.env.reset()
                    break
            self.writer.add_scalar("total reward", self.total_reward, i)
        self.writer.close()

    def eval_policy(self):
        self.env.graph = copy.deepcopy(self.env.graph_backup)
        self.env.current = self.env.get_index(self.env.start, self.env.graph)[0]
        total_reward = 0
        while True:
            self.env.pre_current = copy.deepcopy(self.env.current) 
            action = np.argmax(self.Q[self.get_state_index(self.env.current)])
            state, reward, done = self.env.step(action, self.env.pre_current)
            self.path.append(self.actions[action])
            total_reward += reward

            if done:
                break
        return total_reward
    
if __name__ == '__main__':
    # pdb.set_trace()
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
    model.learn(5000)
    _ = model.eval_policy()
    print(model.env.graph)
    print(f"reward: {_}")
    print(f"shortest path: {model.path}")