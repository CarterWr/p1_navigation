#Needed imports
import numpy as np
import torch#import stuff such as functional later when needed just doing the base imports
import random
import torch.optim as optim
import torch.nn.functional as F

#import model.py
from model import QNetwork

#Getting device for Q-Network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """learns the enviroment and interactis with it"""
    
    def __init__(self, action_size, state_size, seed, eps=1.0, lr=0.05, gamma=0.8, TAU=0.001, batch_size=64, buffer_size=int(1e5)):
        """
        Initalizes a agent object
        
        Params
        - action_size(int): the size of the action space
        - state_size(int): the amount of states there are in a given enviroment
        - seed(int): random seed(random number)
        - eps(float): the epsilon value
        - lr(float): the learning rate used for the optimzer
        - gamma(float): the discount rate for the agent to limit its view for future reward or limit neersight
        - TAU(float): the target network update parameter
        - batch_size(int): the amount of batches
        """
        #initalizations
        self.action_size = action_size
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.eps = eps
        self.gamma = gamma
        self.TAU = TAU
        self.batch_size = batch_size
        
        
        #Creating a Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.time_step = 0
        
    def step(self, state, action, reward, next_state, done, update_every=4):
        """
        Perform a step in the environment
        
        Params
        - state: The current state in the enviroment
        - action: the action the agent wants to take in its current enviroment
        - reward(float): A reward obtained by the action
        - next_state: The next state after taking the action
        - done(bool): if the episode is done or not
        - update_every(int): this number is how many time steps it takes to update the fixed q values
        """
        #add experince tuple to memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.time_step = (self.time_step + 1) % update_every
        
        if self.time_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
                
        
        return None
    
    
    def epsilon_update(self, e_decay=.005, e_min=.01):
        """
        Updates epsilon, call this function in the training loop
        
        Params
        - e_decay(float): is the decay rate of the epsilon value
        - e_min(float): the minimum value that epsilon is allowed to go to
           
        Returns
        None: This function updates the class epsilon value
        """
        
        #Loop until epsilon value is equal to the min value
        if self.eps >= e_min:
            self.eps *= (1 - e_decay)
            
        return None
    
    
    def action(self, state):
        """
        given the current state select an action
        
        params
        - state(array-like): the current state of the enviromnet 
        
        returns: the action of given state for the current policy
        
        """
    
        # Convert the state to a PyTorch tensor and move it to the device (GPU if available)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Pass the state through the Q-network to get Q-values
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        
        # Calculate action probs for each action
        probs = np.full(self.action_size, self.eps / self.action_size)
        
        # Move probs and q_values to the device
        probs_tensor = torch.FloatTensor(probs).to(device)
        q_values = q_values.to(device)

        # Update action probabilities for greedy action
        probs_tensor[q_values.argmax().item()] += 1 - self.eps
        
        return np.random.choice(self.action_size, p=probs_tensor.cpu().numpy())
        

    def learn(self, exp_tup):
        """
        Update paramaters using the experince tuple
        
        Params
        - exp_tup(tuple): experince tuple (state, action, reward, next_state, done)
        """
        
        #Unpack experince tuple
        state, action, reward, next_state, done = exp_tup
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state).gather(1, action)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        return None
    
    def soft_update(self, local_model, target_model):
        """
        Update model paramaters
        
        Params
        - local_model(pytorch model): weights will be copied from the model
        - target_model(pytorch model): weights will be copied from the model
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
        
        return None 

    
# ==================
# ReplayBuffer Class
# ==================

# ReplayBuffer Imports
from collections import namedtuple, deque

class ReplayBuffer:
    """Stores a finite amount of experince tuples"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initalize a replay buffer object
        
        Params
        - action_size(int): the size of the action space
        - buffer_size(int): the size of the buffer
        - batch_size(int): the size of each training batch
        - seed(int): random seed(random number)
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        
    def add(self, state, action, reward, next_state, done):
        """adds new experience to buffer"""
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
        return None
        
    def sample(self):
        """
        sampels randomly from the stored memory
        
        Returns:
        - exp_tup(tuple) a experince tuple 
        
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        exp_tup = (states, actions, rewards, next_states, dones)
        
        return  exp_tup
    
    def __len__(self):
        """Returns the size of the memory"""
        return len(self.memory)