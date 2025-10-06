# Ref : https://github.com/gouxiangchen/soft-Q-learning/blob/master/sql.py

import loa_game, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import torch.optim as optim # for using optimization method
from torch.optim import RMSprop, Adam
import torch.nn.init as init # for weight initialization
from torchsummary import summary
import torchvision.transforms as transforms # for data transformation to use pytorch model
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm

from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import datetime
import argparse

import numpy as np
import random
import gym

# EPISODE_NUM = 2500 # for training, set this with other number
EPISODE_NUM = 5
# TRAIN_START = 1000
TRAIN_START = 20000

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
    
    def forward(self, x):
        return self.layer(x)

class DQN_Agent:
    def __init__(self):
        self.render = True # set this True for rendering game
        self.load_model = True # set this True for bring trained h5 file

        # state & action
        self.state_size = (4, 84, 84)
        self.action_size = 4

        # DQN hyperparameters
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 10
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 64
        self.train_start = TRAIN_START
        self.update_target_rate = 400
        self.discount_factor = 0.99

        # replay memory
        self.memory = deque(maxlen=100000)
        self.no_op_steps = 30

        # model & target model
        self.model = self.build_model().to(device)
        self.target_model = self.build_model().to(device)
        self.update_target_model()

        self.optimizer = RMSprop(params=self.model.parameters(), lr=0.00025, eps=0.01)
        
        self.avg_q_max, self.avg_loss = 0, 0
        
        if self.load_model:
            self.model.load_state_dict(torch.load("../save_model/DQN/DQN_agent.pth", map_location=torch.device('cpu')))

    def build_model(self):
        model = QNetwork(self.action_size)

        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, history):
        # For training
        # history = np.float32(history / 255.0)
        # if np.random.rand() <= self.epsilon:
        #     return random.randrange(self.action_size)
        # else:
        #     with torch.no_grad():
        #         q_value = self.model.forward(torch.FloatTensor(history).to(device))

        # return np.argmax(q_value[0].cpu().detach().numpy())
        
        # For inference
        with torch.no_grad():
            q_value = self.model.forward(torch.FloatTensor(history).to(device))
        
        return np.argmax(q_value[0].cpu().detach().numpy())

    # save sample <s, a, r, s'> to replay memory
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # random batch training
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        pred = self.model.forward(torch.FloatTensor(history).to(device)).to(device)
        target_value = self.target_model.forward(torch.FloatTensor(next_history).to(device)).to(device) # 64 X 4

        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                            torch.amax(target_value[i])

        self.optimizer.zero_grad()

        loss = F.huber_loss(pred.gather(1, torch.unsqueeze(torch.LongTensor(action), 1).to(device)).to(device), torch.unsqueeze(torch.FloatTensor(target), 1).to(device))
        
        loss.backward()

        self.optimizer.step()

        self.avg_loss += loss.item()

class SQL_Agent:
    def __init__(self):
        self.render = True # set this True for rendering game
        self.load_model = True # set this True for bring trained h5 file

        # state & action
        self.state_size = (4, 84, 84)
        self.action_size = 4

        # SQL hyperparameters
        self.alpha = 0.6 # for managing entropy term (= exploration) (+ it is also hyperparameter)
        self.batch_size = 64 # BATCH
        self.train_start = TRAIN_START # 128
        self.update_target_rate = 400 # UPDATE_STEPS
        self.discount_factor = 0.99 # GAMMA

        # replay memory
        self.memory = deque(maxlen=100000) # REPLAY_MEMORY
        self.no_op_steps = 30

        # model & target model
        self.model = self.build_model().to(device)
        self.target_model = self.build_model().to(device)
        self.update_target_model()

        self.optimizer = RMSprop(params=self.model.parameters(), lr=0.00025, eps=0.01)
        
        self.avg_q_max, self.avg_loss = 0, 0
        
        if self.load_model:
            self.model.load_state_dict(torch.load("../save_model/SQL/SQL_agent.pth", map_location=torch.device('cpu')))

    def build_model(self):
        model = QNetwork(self.action_size)
        
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, history):
        history = np.float32(history / 255.0)

        with torch.no_grad():
            q = self.model.forward(torch.FloatTensor(history).to(device)).to(device)
            v = self.alpha * torch.logsumexp((q/self.alpha).to(device), dim=1).to(device)
            # v = self.alpha * torch.log(torch.sum(torch.exp(q/self.alpha), dim=1, keepdim=True))

            # other reasonable solution
            # ratio = (current_log_probs - old_log_probs).sum(1, keepdim=True)
            # ratio = ratio.clamp_(max=88).exp()
 
            action_dist = torch.exp((q-v).to(device)/self.alpha).to(device)

            action_dist = action_dist / torch.sum(action_dist).to(device)

            return Categorical(action_dist).sample().item()

    # save sample <s, a, r, s'> to replay memory
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # random batch training
    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        # batch size = 64, state_size = (4, 84, 84)
        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        dead = torch.FloatTensor(dead).to(device)

        next_q = self.target_model.forward(torch.FloatTensor(next_history).to(device)).to(device)
        next_v = self.alpha * torch.logsumexp((next_q/self.alpha).to(device), dim=1).to(device)
        # next_v = self.alpha * torch.log(torch.sum(torch.exp(next_q/self.alpha), dim=1, keepdim=True))

        y = torch.unsqueeze(reward, 1).to(device) + (1 - torch.unsqueeze(dead, 1)).to(device) * self.discount_factor * torch.unsqueeze(next_v, 1).to(device)
        
        self.optimizer.zero_grad()

        loss = F.huber_loss(self.model.forward(torch.FloatTensor(history).to(device)).gather(1, torch.unsqueeze(action, 1).to(device)), y.to(device))
        
        loss.backward()

        self.optimizer.step()

        self.avg_loss += loss.item()

# gray scaling
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    env = loa_game.Env()

    parser = argparse.ArgumentParser(description='Custom-made Game RL')
    parser.add_argument('--agent', default='DQN', help='DQN or SQL')

    args = parser.parse_args()

    # save_dir = '../save_model/' + str(args.agent) + '/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # writer_dir = '../tensorboard/' + str(args.agent) + '/'
    # if not os.path.exists(writer_dir):
    #     os.makedirs(writer_dir)
    
    # writer = SummaryWriter(writer_dir + str(args.agent) + "_agent")

    if(args.agent == "DQN"):
        if(torch.cuda.is_available()):
            device = torch.device('cuda:0')
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            device = torch.device('cpu')
            print("Device set to : cpu")
        
        agent = DQN_Agent()
    else:
        if(torch.cuda.is_available()):
            device = torch.device('cuda:1')
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            device = torch.device('cpu')
            print("Device set to : cpu")
        
        agent = SQL_Agent()

    scores, episodes, global_step = [], [], 0

    for e in tqdm(range(EPISODE_NUM)):
        done = False
        dead = False

        step = 0
        score = 0
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=0)
        history = history[np.newaxis, :]

        while not done:
            if agent.render:
                env.render()

            global_step += 1
            step += 1

            action = agent.get_action(history)

            # proceed one step
            observe, reward, done, info = env.step(action)

            # preprocess state for each time step
            next_state = pre_processing(observe)
            next_state = next_state[np.newaxis, :]
            next_history = np.append(next_state, history[0][:3, :, :], axis=0)
            next_history = next_history[np.newaxis, :]

            agent.avg_q_max += torch.amax(
                agent.model.forward(torch.FloatTensor(np.float32(history / 255.)).to(device))[0])

            reward = np.clip(reward, -1., 1.)

            # save sample <s, a, r, s'> to replay memory & training
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # update target model at schedule intervals
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                # record training information per episode
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    
                    # writer.add_scalar('episode_score', stats[0], e)
                    # writer.add_scalar('episode_avg_max_q', stats[1], e)
                    # writer.add_scalar('step', stats[2], e)
                    # writer.add_scalar('episode_avg_loss', stats[3], e)

                print()
                print("episode:", e, " | score:", score, ", memory length:",
                      len(agent.memory),", global_step:", global_step, ", average_q:",
                      (agent.avg_q_max / float(step)).item(), ", average loss:",
                      agent.avg_loss / float(step))
                print()

                agent.avg_q_max, agent.avg_loss = 0, 0

        # save model every 100 episodes
        # if e % 100 == 0:
        #     torch.save(agent.model.state_dict(), save_dir + str(args.agent) + "_agent.pth")

#writer.flush()
#writer.close()
env.recorder.end_recording()
env.close()
