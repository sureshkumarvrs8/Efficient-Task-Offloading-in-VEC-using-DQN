import random
import numpy as np
from scipy.integrate import quad
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
m = 10**6
k = 10**3
g = 10**9

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0
    def store(self, state, action, next_state, reward, done):
        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done
        self.idx = (self.idx+1)%self.capacity
    def sample(self, batch_size, device):
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)
        states = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(device)
        return states, actions, next_states, rewards, dones
    def __len__(self):
        return len(self.buffer_state)

class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 32)
        self.dense5 = nn.Linear(32, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = (self.dense5(x))
        return x
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    def load_model(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))

class DQNAgent:
    def __init__(self, device, state_size, action_size,discount=0.99,eps_max=1.0,eps_min=0.01,eps_decay=0.995,memory_capacity=25000,lr=1e-3,train_mode=True):
        self.device = device
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.discount = discount
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval()
        if not train_mode:
            self.policy_net.eval()
        self.memory = ReplayMemory(capacity=memory_capacity)
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item()
    def learn(self, batchsize):
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)
        actions=actions.type(torch.int64)
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1))
        q_target = self.target_net.forward(next_states).max(dim=1).values
        q_target[dones] = 0.0
        y_j = rewards + (self.discount * q_target)
        y_j = y_j.view(-1, 1)
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
    def save_model(self, filename):
        self.policy_net.save_model(filename)
    def load_model(self, filename):
        self.policy_net.load_model(filename=filename, device=self.device)

class RSU:
    def __init__(self):
        self.freq = 1*g # in Hz or cycles/s
        self.height = 5 #in meters
        self.radius = 300 #in meters
        self.stay_dist = 2*math.sqrt(self.radius**2-self.height**2) #in meters
        self.power = 2 #in watts
        self.loadfactor = 0
    def compDelay(self,task_size):
        return task_size/(self.freq/297.62)
    def energy(self,task_size):
        self.energy = self.power*self.compDelay(task_size)
        return self.energy

class Vehicle:
    def __init__(self):
        self.freq = 500*m # in Hz or cycles/s
        self.speed = 60 #km/hr
        self.speed = (self.speed*5)/18 #m/s
        self.power = 1.5 #in watts
        self.loadfactor = 0
    def stayTime(self):
        rsu = RSU()
        self.stay_time = rsu.stay_dist/self.speed
        return int(self.stay_time)
    def compDelay(self,task_size):
        self.comp_delay = task_size/(self.freq/297.62)
        return self.comp_delay
    def energy(self, task_size):
        self.energy = self.power*self.compDelay(task_size)
        return self.energy

bw = 10*m #in Hz
loss = -4
fading = 0.5
noise = -70 #in dB
noise = 10**(noise/10)
rsu = RSU()
v = Vehicle()
def rate(t):
    d = math.sqrt(rsu.height**2 + (rsu.stay_dist/2 - v.speed*t)**2)
    r = bw*math.log2(1+((v.power*(d**loss)*(fading**2))/noise))
    return r
def commDelay(task_size):
    avg_rate = quad(rate,1,v.stayTime())
    avg_rate = (abs(avg_rate[0])/v.stayTime())
    return task_size/avg_rate

states = []
states.append([0,0,0,0])
for t_size in [5,10]:
  if t_size==5:
    tdeadlim=3
  else:
    tdeadlim=5
  for t_dead in range(tdeadlim,11):
    for v_lf in range(0,11):
      for r_lf in range(0,11):
        states.append([t_size, t_dead, v_lf, r_lf])
print(len(states),states)

class Environment:
    def __init__(self, v, rsu):
        self.v = v
        self.rsu = rsu
        self.cnt = 0
        self.states = states
        self.action_space = [0,1] #0-local, 1-offload
    def reset(self):
        self.delay = 0
        v.loadfactor = 0
        rsu.loadfactor = 0
        self.loadfactor = 0
        self.cnt = 0
        return self.states[0]
    def step(self,state,action):
        task_size = state[0]
        task_deadline = state[1]
        v.loadfactor = state[2]
        rsu.loadfactor = state[3]
        self.done = False
        loc_delay = v.compDelay(task_size*m)
        rsu_delay = rsu.compDelay(task_size*m) + commDelay(task_size*m)
        self.cnt+=1
        self.next_state = self.states[self.cnt]
        self.reward = 0
        w1 = 2
        w2 = 1
        self.delay = (1-action)*loc_delay + action*rsu_delay
        self.loadfactor = (1-action)*v.loadfactor + action*rsu.loadfactor
        mean_lf = (v.loadfactor + rsu.loadfactor)/2
        self.loadfactor_var = ((self.loadfactor - mean_lf)**2)/2
        self.reward = -w1*(self.delay) - w2*(self.loadfactor_var)
        if(task_deadline < self.delay):
          self.reward -= 10
        if(v.loadfactor < 2 and action == 0):
          self.reward += 10
        if(v.loadfactor > 8 and action == 1):
          self.reward += 10
        if(v.loadfactor < 2 and action == 1):
          self.reward -= 10
        if(v.loadfactor > 8 and action == 0):
          self.reward -= 10

        if(self.cnt == 1690):
            self.done = True
        return self.next_state, self.reward, self.done

def fill_memory(env, dqn_agent, num_memory_fill_eps):
    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()
        while not done:
            action = random.choice([0,1])
            next_state, reward, done = env.step(state,action)
            dqn_agent.memory.store(state=state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                done=done)
            state = next_state

def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, model_filename):
    fill_memory(env, dqn_agent, num_memory_fill_eps)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    reward_history = []
    step_cnt = 0
    best_score = -np.inf
    for ep_cnt in range(num_train_eps):
        done = False
        state = env.reset()
        ep_score = 0
        while not done:
            action = dqn_agent.select_action(state)
            next_state, reward, done = env.step(state,action)
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=done)
            dqn_agent.learn(batchsize=batchsize)
            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()
            state = next_state
            ep_score += reward
            step_cnt += 1
        dqn_agent.update_epsilon()
        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-10:])
        if current_avg_score >= best_score:
            dqn_agent.save_model(model_filename)
            best_score = current_avg_score
        print('Ep: {}, Score: {}'.format(ep_cnt, ep_score))

def test(env, dqn_agent, num_test_eps):
    step_cnt = 0
    reward_history = []
    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:
            action = dqn_agent.select_action(state)
            print(state, action)
            next_state, reward, done = env.step(state,action)
            score += reward
            state = next_state
            step_cnt += 1
        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

if __name__ ==  '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = True
    v = Vehicle()
    rsu = RSU()
    env = Environment(v,rsu)
    model_filename = 'offload_online_net'
    if train_mode:
        dqn_agent = DQNAgent(device, state_size = 4, action_size = 2,
                    discount=0.99,
                    eps_max=1.0,
                    eps_min=0.01,
                    eps_decay=0.995,
                    memory_capacity=5000,
                    lr=1e-3,
                    train_mode=True)
        train(env=env,
                dqn_agent=dqn_agent,
                num_train_eps=1000,
                num_memory_fill_eps=3,
                update_frequency=100,
                batchsize=64,
                model_filename=model_filename)

    else:
        dqn_agent = DQNAgent(device,
                    state_size = 4, action_size = 2,
                    discount=0.99,
                    eps_max=0.0,
                    eps_min=0.0,
                    eps_decay=0.0,
                    train_mode=False)
        dqn_agent.load_model(model_filename)
        test(env=env, dqn_agent=dqn_agent, num_test_eps=10)

action = dqn_agent.select_action([5,7,8.7,5])
print(action)





