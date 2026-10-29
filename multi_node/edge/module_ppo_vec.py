import config
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random
from towhee_embedding import towhee_embedding
from mv_weight_file import mv_weight_file
from copy_weight_file import copy_weight_file
from diffusion import Diffusion
# Do not modify this program

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states_for_ppo = []
        self.logprobs = []
        self.rewards = []
        self.states_for_common = []

    def clear(self):
        del self.actions[:]
        del self.states_for_ppo[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.states_for_common[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.actor = Diffusion(state_dim=state_dim_actor, action_dim=action_dim, max_action=action_dim)
        self.transen = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=6
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 30)
        )
        # 通过价值输出动作
        self.critic = nn.Sequential(
            nn.Linear(state_dim_critic, critic_node),
            nn.Tanh(),
            nn.Linear(critic_node, critic_node),
            nn.Tanh(),
            nn.Linear(critic_node, 1)
        )
        # 通过策略评估state+action
    def act_limitation(self, state_for_ppo, task_input):
        att = self.linear_layer(self.transen(task_input.view(1,1,768)).view(1, 768))
        new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
        state_add1 = torch.tensor(new_state[np.newaxis, :])
        action_probs = self.actor(state_add1)
        dist = Categorical(action_probs)
        action = dist.sample()

        action_logprob = dist.log_prob(action)


        # print(action_logprob,'action_logprob')
        return action.detach(), action_logprob.detach(), new_state
    def get_new_state(self, state_for_ppo, task_input):
        att = self.linear_layer(self.transen(task_input.view(1, 1, 768)).view(1, 768))
        new_state = torch.cat([state_for_ppo[0:30], att.squeeze()], 0)

        return new_state

    def act(self, state_for_ppo, task_input):

        att = self.linear_layer(self.transen(task_input.view(1,1,768)).view(1, 768))

        new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
        action_probs = self.actor(new_state)

        # print(action_probs,'select_action_probs')
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # print(action_logprob,'action_logprob')
        return action.detach(), action_logprob.detach(), new_state

    def evaluate(self, state_for_ppo, state_for_common, action, action_oldlogprobs):
        # 最初是一样的，更新后可以输出学习后的动作
        action_probs = self.actor(state_for_ppo)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        # print(action_logprobs[0],'action_logprobs')
        dist_entropy = dist.entropy()
        state_for_common_values = self.critic(state_for_common)
        return action_logprobs, state_for_common_values, dist_entropy


class PPO:
    def __init__(self, state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node).to(
            self.device)
        # 有初始权重就加载，没有则从随机初始化开始，之后由训练端推送的权重覆盖
        init_weight = os.path.exists(config.INIT_WEIGHT_PATH)
        if init_weight:
            self.policy.load_state_dict(torch.load(config.INIT_WEIGHT_PATH, map_location=self.device))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.transen.parameters(), 'lr': lr_actor},
            {'params': self.policy.linear_layer.parameters(), 'lr': lr_actor},
        ])
        self.policy_old = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node).to(
            self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        print('载入初始模型' if init_weight else '未找到初始权重，使用随机初始化')
        self.MseLoss = nn.MSELoss()

    def select_action(self, timestep_counter, state_for_ppo, task_array, send_nodes, index_for_large):

        if task_array == []:
            task_embedding = towhee_embedding(inputt=' ')
            # print(task_embedding)
            print('任务为空')
        else:
            task_embedding = towhee_embedding(task_array[0])

        with torch.no_grad():
            state_for_ppo = torch.FloatTensor(state_for_ppo.astype(np.float32)).to(self.device)
            task_input = torch.FloatTensor(np.float32(task_embedding)).to(self.device)
            action, action_logprob, new_state = self.policy_old.act_limitation(state_for_ppo, task_input)
                # print(timestep_counter, new_state, 'state_for_ppo')

        # new_state_common = new_state = self.policy_old.get_new_state(state_for_ppo, task_input)
        action = action.to(self.device)
        action_logprob = action_logprob.to(self.device)
        self.buffer.states_for_ppo.append(new_state)
        # print(self.buffer.states_for_ppo,'self.buffer.states_for_ppo')
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        # print(action_logprob,'action_logprob')
        return action.item(), action_logprob, new_state

    def generate_unique_random_numbers(self, start, end, count):
        numbers = set()

        while len(numbers) < count:
            number = random.randint(start, end)
            numbers.add(number)

        return list(numbers)

    def clear_buffer(self, index_for_large, update_times):

        # clear buffer
        # 清空记录?
        self.buffer.clear()
        print('buffer已经清空')

    def update(self, index_for_large, update_times, timestep_counter):
        learn_rewards = self.buffer.rewards.copy()
        learn_states_for_ppo = self.buffer.states_for_ppo.copy()
        learn_states_for_common = self.buffer.states_for_common.copy()
        learn_actions = self.buffer.actions.copy()
        learn_logprobs = self.buffer.logprobs.copy()

        try:
            time.sleep(60)
            self.policy.load_state_dict(torch.load(config.EDGE_WEIGHT_PATH, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            update_times = update_times + 1
        except:
            try:
                time.sleep(60)
                self.policy.load_state_dict(torch.load(config.EDGE_WEIGHT_PATH, map_location=self.device))
                self.policy_old.load_state_dict(self.policy.state_dict())
                update_times = update_times + 1
            except:
                update_times = update_times + 1
                print('model has error')
                file = open('error model.txt', 'a')
                # 写入内容
                file.write(str(update_times))
                file.write('\n')
                # 关闭文件
                file.close()
        if update_times > 30:
            if timestep_counter >= 13500:
                copy_weight_file(config.EDGE_WEIGHT_PATH, "./weightsave_log", update_times)
            mv_weight_file(config.EDGE_WEIGHT_PATH, "./weight_log", update_times)

        model_policy = self.policy_old

        print('模型更新成功')
        return update_times




    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)

    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
