# 第一步：添加multiagent路径
import sys
import os
import random

sys.path.append("C:\\Users\\22895\\Desktop\\uav_project\\multiagent-particle-envs")

# Win11兼容设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['SUPPRESS_MA_PROMPT'] = '1'
os.chdir('C:\\Users\\22895\\Desktop\\uav_project\\UAV-path-planning')

# 导入模块
import argparse
import numpy as np
import torch
from collections import deque
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from epc_iclr import EPCTrainer

# 日志保存配置
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "uav_epc_results")
os.makedirs(save_dir, exist_ok=True)

train_logs = {
    'episodes': [],
    'total_rewards': [],
    'actor_losses': [],
    'critic_losses': [],
    'actor_grad_norms': [],
    'smooth_rewards': []
}

# 测试路径可写
test_save_dir = os.path.join(current_dir, "train_results")
os.makedirs(test_save_dir, exist_ok=True)
with open(os.path.join(test_save_dir, "test.txt"), "w") as f:
    f.write("测试文件：路径可写")
print(f"测试文件已生成：{os.path.join(test_save_dir, 'test.txt')}")

# 解析参数（最终微调：平衡探索与利用）
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=2000)
parser.add_argument('--ep_length', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=3e-5)      # 5e-5 → 3e-5
parser.add_argument('--ent_coef', type=float, default=0.0004)  # 0.0003 → 0.0004
parser.add_argument('--save_dir', type=str, default='./uav_epc_results')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 加载环境并重载奖励函数（强化连续收集引导）
scenario = scenarios.load("food_collection.py").Scenario()

def enhanced_reward(agent, world):
    reward = 0.0
    food_collected = 0

    # 1. 碰撞惩罚（保持）
    if agent.collide:
        for landmark in world.landmarks:
            if 'obstacle' in landmark.name and scenario.is_collision(agent, landmark):
                reward -= 0.05

    # 2. 收集食物奖励 + 更强的累积奖励
    for landmark in world.landmarks:
        if 'food' in landmark.name and scenario.is_collision(agent, landmark):
            food_collected += 1
            reward += 4.0
            reward += food_collected * 1.0  # 累积奖励+1.0/个
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

    # 3. 向最近食物移动奖励（进一步放大引导）
    active_foods = [l for l in world.landmarks if 'food' in l.name]
    if active_foods:
        closest_food = min(active_foods, key=lambda f: np.linalg.norm(agent.state.p_pos - f.state.p_pos))
        dist = np.linalg.norm(agent.state.p_pos - closest_food.state.p_pos)
        reward += max(0, 0.1 - 0.02 * dist)

    # 4. 靠近障碍物惩罚（保持）
    for landmark in world.landmarks:
        if 'obstacle' in landmark.name:
            obs_dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            if obs_dist < 1.0:
                reward -= (1.0 - obs_dist) * 0.005

    # 5. 保持移动奖励（保持）
    agent_speed = np.linalg.norm(agent.state.p_vel)
    if agent_speed > 0.01:
        reward += 0.001

    return reward

scenario.reward = enhanced_reward
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

# 初始化维度与训练器
state_dim = env.observation_space[0].shape[0]
from multiagent.multi_discrete import MultiDiscrete
if hasattr(env.action_space[0], 'n'):
    num_actions = env.action_space[0].n
    max_action = num_actions - 1
elif hasattr(env.action_space[0], 'high'):
    num_actions = env.action_space[0].shape[0]
    max_action = env.action_space[0].high[0]
elif isinstance(env.action_space[0], MultiDiscrete):
    num_actions = env.action_space[0].num_discrete_space
    max_action = env.action_space[0].high.max()
else:
    num_actions = 1
    max_action = 1.0

trainer = EPCTrainer(
    state_dim=state_dim,
    action_dim=num_actions,
    max_action=max_action,
    lr=args.lr,
    gamma=args.gamma,
    entropy_coeff=args.ent_coef
)
trainer.tau = 0.01  # 目标网络更新系数

# 训练循环配置
replay_buffer = deque(maxlen=30000)
clean_interval = 1000

def is_valid(value):
    if isinstance(value, np.ndarray):
        return not (np.isinf(value).any() or np.isnan(value).any())
    elif isinstance(value, torch.Tensor):
        return not (torch.isinf(value).any() or torch.isnan(value).any())
    else:
        return not (np.isinf(value) or np.isnan(value))

def one_hot_action(action, num_actions):
    oh = np.zeros(num_actions, dtype=np.float32)
    oh[int(action)] = 1.0
    return oh

def calculate_smooth_reward(rewards, window=10):
    if len(rewards) < window:
        return np.mean(rewards) if rewards else 0.0
    return np.mean(rewards[-window:])

print("===== 无人机Food Collection训练开始（最终优化版，2000回合） =====")
print(f"配置：lr={args.lr}, 熵系数={args.ent_coef}, 梯度裁剪max_norm=0.2")

for episode in range(args.num_episodes):
    if episode % clean_interval == 0 and episode > 0:
        current_len = len(replay_buffer)
        if current_len > 0:
            replay_buffer = deque(list(replay_buffer)[current_len//2:], maxlen=30000)
            print(f"📦 第{episode}回合：清理经验池，剩余{len(replay_buffer)}条")

    obs = env.reset()
    total_reward = 0.0
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    actor_grad_sum = 0.0
    step_count = 0

    for step in range(args.ep_length):
        actions = []
        action_oh_list = []
        for ob in obs:
            action_tensor, _, _ = trainer.actor.sample(torch.FloatTensor(ob.astype(np.float32)))
            action = int(torch.clamp(action_tensor.float().round(), 0, max_action).item())
            actions.append(action)
            action_oh_list.append(one_hot_action(action, num_actions))

        next_obs, rewards, dones, _ = env.step(actions)
        total_reward += sum(rewards)
        step_count += 1

        replay_buffer.append({
            'states': obs[0].astype(np.float32),
            'actions': action_oh_list[0],
            'next_states': next_obs[0].astype(np.float32),
            'rewards': np.float32(rewards[0]),
            'dones': np.float32(dones[0])
        })

        if len(replay_buffer) >= args.batch_size:
            try:
                batch_samples = random.sample(replay_buffer, args.batch_size)
                batch = {
                    'states': np.stack([s['states'] for s in batch_samples]),
                    'actions': np.stack([s['actions'] for s in batch_samples]),
                    'rewards': np.array([s['rewards'] for s in batch_samples], dtype=np.float32)[:, np.newaxis],
                    'next_states': np.stack([s['next_states'] for s in batch_samples]),
                    'dones': np.array([s['dones'] for s in batch_samples], dtype=np.float32)[:, np.newaxis]
                }
                if not is_valid(batch['rewards']):
                    print(f"⚠️  第{episode}回合第{step}步：奖励异常，跳过")
                    continue
                loss = trainer.update(batch)
                if is_valid(loss['actor_loss']) and is_valid(loss['critic_loss']):
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(trainer.actor.parameters(), max_norm=0.2)
                    torch.nn.utils.clip_grad_norm_(trainer.critic.parameters(), max_norm=0.2)
                    actor_loss_sum += loss['actor_loss']
                    critic_loss_sum += loss['critic_loss']
                    actor_grad_sum += actor_grad_norm.item()
            except Exception as e:
                print(f"⚠️  第{episode}回合第{step}步：更新失败 - {str(e)}")
                continue
        obs = next_obs
        if all(dones):
            break

    avg_actor_loss = actor_loss_sum / step_count if step_count > 0 else 0
    avg_critic_loss = critic_loss_sum / step_count if step_count > 0 else 0
    avg_actor_grad = actor_grad_sum / step_count if step_count > 0 else 0
    smooth_reward = calculate_smooth_reward(train_logs['total_rewards'] + [total_reward])

    train_logs['episodes'].append(episode)
    train_logs['total_rewards'].append(total_reward)
    train_logs['actor_losses'].append(avg_actor_loss)
    train_logs['critic_losses'].append(avg_critic_loss)
    train_logs['actor_grad_norms'].append(avg_actor_grad)
    train_logs['smooth_rewards'].append(smooth_reward)

    if episode % 50 == 0:
        if len(train_logs['smooth_rewards']) >= 1000:
            recent_smooth = train_logs['smooth_rewards'][-1000:]
            recent_avg = np.mean(recent_smooth)
            best_avg = np.mean(train_logs['smooth_rewards'][-2000:-1000]) if len(train_logs['smooth_rewards'])>=2000 else 0
            if best_avg > 0 and recent_avg < best_avg * 0.85:
                print("⚠️  平滑奖励连续1000回合下降超15%，触发早停！")
                torch.save({
                    'actor': trainer.actor.state_dict(),
                    'critic': trainer.critic.state_dict(),
                    'episode': episode,
                    'train_logs': train_logs
                }, os.path.join(args.save_dir, 'uav_epc_best_earlystop.pth'))
                np.save(os.path.join(save_dir, 'train_logs_earlystop.npy'), train_logs)
                env.close()
                sys.exit(0)
        print(f"回合 {episode:4d} | 总奖励: {total_reward:5.1f} | 平滑奖励: {smooth_reward:5.1f} | Actor损失: {avg_actor_loss:.4f} | 梯度范数: {avg_actor_grad:.4f}")
        np.save(os.path.join(save_dir, "train_logs_final_2000ep_ultimate.npy"), train_logs)
        torch.save({
            'actor': trainer.actor.state_dict(),
            'critic': trainer.critic.state_dict(),
            'episode': episode,
            'train_config': args
        }, os.path.join(args.save_dir, f'uav_epc_ep{episode}.pth'))

env.close()
final_episode = args.num_episodes - 1
np.save(os.path.join(save_dir, "train_logs_final_2000ep_ultimate.npy"), train_logs)
best_episode = np.argmax(train_logs['smooth_rewards'])
torch.save({
    'actor': trainer.actor.state_dict(),
    'critic': trainer.critic.state_dict(),
    'best_episode': best_episode,
    'best_smooth_reward': train_logs['smooth_rewards'][best_episode],
    'train_logs': train_logs,
    'train_config': args
}, os.path.join(args.save_dir, 'uav_epc_best_final_ultimate.pth'))

print(f"\n===== 2000回合最终优化版训练完成！ =====")
print(f"📊 训练结果：")
print(f"   - 最优平滑奖励回合：{best_episode}，值：{train_logs['smooth_rewards'][best_episode]:.1f}")
print(f"   - 最终100回合平均奖励：{np.mean(train_logs['total_rewards'][-100:]):.1f}")