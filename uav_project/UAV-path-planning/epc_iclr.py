import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# ========== 定义Actor网络（适配离散动作空间） ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)  # action_dim=动作总数
        self.max_action = max_action

    def forward(self, state):
        # state: [batch_size, state_dim] 或 [state_dim]
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 单样本转为batch维度
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, state):
        logits = self.forward(state)
        # 采样动作（离散）
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # 移除batch维度（适配单样本）
        if len(action.shape) > 0 and action.shape[0] == 1:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)

        return action, log_prob, entropy


# ========== 定义Critic网络（修复维度拼接） ==========
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 1
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # Critic 2（双Critic，减少过估计）
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

        self.action_dim = action_dim

    def forward(self, state, action):
        # ========== 核心修复：保证state和action都是2维张量 ==========
        # state: [batch_size, state_dim]
        # action: [batch_size, action_dim]（独热编码）
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # 拼接状态和动作（维度匹配）
        sa = torch.cat([state, action], 1)  # [batch_size, state_dim+action_dim]

        # Critic 1输出
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Critic 2输出
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def q1_forward(self, state, action):
        # 仅用于目标网络更新
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


# ========== 定义EPCTrainer（添加entropy_coeff参数） ==========
class EPCTrainer:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, entropy_coeff=0.001):
        # 1. 初始化网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # 2. 复制目标网络参数
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 3. 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 4. 定义超参数（新增entropy_coeff）
        self.gamma = gamma
        self.max_action = max_action
        self.lr = lr
        self.entropy_coeff = entropy_coeff  # 熵系数参数
        self.tau = 0.005  # 目标网络更新系数
        self.action_dim = action_dim

    def update(self, batch):
        # ========== 1. 转换为tensor并保证维度正确 ==========
        # 确保所有张量都是2维：[batch_size, dim]
        states = torch.FloatTensor(batch['states'])  # [64, state_dim]
        actions = torch.FloatTensor(batch['actions'])  # [64, action_dim]
        next_states = torch.FloatTensor(batch['next_states'])  # [64, state_dim]
        rewards = torch.FloatTensor(batch['rewards'])  # [64, 1]
        dones = torch.FloatTensor(batch['dones'])  # [64, 1]

        # ========== 2. 更新Critic ==========
        with torch.no_grad():
            # 采样下一个动作（返回logits：[64, action_dim]）
            next_action_logits = self.actor(next_states)
            next_action_dist = torch.distributions.Categorical(logits=next_action_logits)
            next_actions = next_action_dist.sample()  # [64]

            # 将整数动作转为独热编码：[64, action_dim]
            next_actions_oh = F.one_hot(next_actions, num_classes=self.action_dim).float()

            # 目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions_oh)
            target_q = torch.min(target_q1, target_q2) - self.entropy_coeff * next_action_dist.log_prob(
                next_actions).unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== 3. 更新Actor ==========
        action_logits = self.actor(states)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions_pred = action_dist.sample()  # [64]
        actions_pred_oh = F.one_hot(actions_pred, num_classes=self.action_dim).float()  # [64, action_dim]

        q1, q2 = self.critic(states, actions_pred_oh)
        q = torch.min(q1, q2)
        # Actor损失（使用传入的熵系数）
        actor_loss = (self.entropy_coeff * action_dist.log_prob(actions_pred).unsqueeze(1) - q).mean()

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 4. 更新目标Critic网络 ==========
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ========== 5. 返回损失值 ==========
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }