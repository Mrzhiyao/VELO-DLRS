import config
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import random
from towhee_embedding import towhee_embedding
# Do not modify this program
from show_file_name import show_file_name
from show_file_len import show_file_len
from diffusion import Diffusion
import time
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
        # print(state_dim_actor,actor_node,action_dim)
        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim_actor, actor_node),
        #     nn.Tanh(),
        #     nn.Linear(actor_node, actor_node),
        #     nn.Tanh(),
        #     nn.Linear(actor_node, action_dim),
        #     nn.Softmax(dim=-1)
        # )
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
        # new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
        # print(new_state.shape, type(new_state),'new_state')
        # action_probs = self.actor(new_state)
        # dist = Categorical(action_probs)
        # action = dist.sample()
        # action_logprob = dist.log_prob(action)

        new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
        # print(state_for_ppo.shape)
        # print(new_state.shape, type(new_state),'new_state')
        state_add1 = torch.tensor(new_state[np.newaxis, :])
        action_probs = self.actor(state_add1)
        print('action_probs:', action_probs)
        dist = Categorical(action_probs)
        # print(dist,'dist')
        action = dist.sample()

        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), new_state

    def get_new_state(self, state_for_ppo, task_input):
        att = self.linear_layer(self.transen(task_input.view(1, 1, 768)).view(1, 768))
        new_state = torch.cat([state_for_ppo[0:30], att.squeeze()], 0)

        return new_state

    def act(self, state_for_ppo, task_input):
        # 这个部分输出act时是输出的是动作，以及概率，而不是直接根据value输出最大值。
        # print(state_for_ppo)
        att = self.linear_layer(self.transen(task_input.view(1,1,768)).view(1, 768))
        # print(att.shape, 'att')
        # print(state_for_ppo.shape, 'state_for_ppo')
        # 0 1 2 3 4 5 6 7 / 40 41 42 43
        # for i in range(state_for_ppo.shape[0]):
        #     if i <= 39:
        #         if i%4 == 0:
        #             state_for_ppo[i] = state_for_ppo[i] * att[0][0]
        #         elif i%4 == 2:
        #             state_for_ppo[i] = state_for_ppo[i] * att[0][1]
        #         elif i%4 == 3:
        #             state_for_ppo[i] = state_for_ppo[i] * att[0][2]
        #     else:
        #         if (i-40) % 2 == 0:
        #             state_for_ppo[i] = state_for_ppo[i] * att[0][0]
        #         elif (i-40) % 2 == 1:
        #             state_for_ppo[i] = state_for_ppo[i] * att[0][1]
        # print(state_for_ppo.shape)
        # print(att.shape)
        # print(torch.cat([state_for_ppo, att.squeeze()], 0).shape)
        new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
        action_probs = self.actor(new_state)

        # print(action_probs,'select_action_probs')
        dist = Categorical(action_probs)
        # print(dist,'dist')
        action = dist.sample()
        # print(action,'actionces')
        # print(action,type(action),'action')
        # print(action,'actiondist')
        # print(action_probs, dist.log_prob(action),'dist.log_prob(action)')
        action_logprob = dist.log_prob(action)
        # print(action_logprob,'action_logprob')
        return action.detach(), action_logprob.detach(), new_state

    def evaluate(self, state_for_ppo, state_for_common, action, action_oldlogprobs):
        # 最初是一样的，更新后可以输出学习后的动作
        print(state_for_ppo,'state_for_ppo')
        action_probs = self.actor(state_for_ppo)
        # print(action_probs.shape)
        # print(action,'action')
        # print(action_probs,'action_probs')
        # print(action_oldlogprobs[0],'action_oldlogprobs')
        # print(action_probs,'action_probs')
        dist = Categorical(action_probs)
        # print(dist,'dist')
        # 这个地方为什么action不一样？
        print(action)
        print(action.shape,'shape')
        action_logprobs = dist.log_prob(action)
        # print(action_logprobs[0],'action_logprobs')
        dist_entropy = dist.entropy()
        # print(state_for_common.to(dtype=torch.double).to(self.device), type(state_for_common.to(dtype=torch.double).to(self.device)),'ddd')

        # x = torch.tensor(state_for_common.detach(), dtype=torch.double).to(self.device)
        # print(x.to(self.device), type(x.to(self.device)),'ddd')
        # print(state_for_common,state_for_common.shape,type(state_for_common))
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
        # self.policy.load_state_dict(torch.load("model_network.pth"))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.transen.parameters(), 'lr': lr_actor},
            {'params': self.policy.linear_layer.parameters(), 'lr': lr_actor},
        ])
        self.policy_old = ActorCritic(state_dim_critic, state_dim_actor, action_dim, actor_node, critic_node).to(
            self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.policy_old.load_state_dict(torch.load("model_network.pth"))
        # print('载入初始化模型')
        self.MseLoss = nn.MSELoss()

    def select_action(self, timestep_counter, state_for_ppo, task_array, send_nodes, index_for_large):

        if task_array == []:
            task_embedding = towhee_embedding(inputt=' ')
            # print(task_embedding)
            print('任务为空')
        else:
            task_embedding = towhee_embedding(task_array[0])
            # print(task_embedding)
        # print(torch.FloatTensor(np.float32(task_embedding)).to(self.device),'task_embedding')

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

    def update(self, index_for_large, update_times):
        update_times_greedy = update_times + 3
        node1_file, node2_file, node3_file = show_file_name(update_times)
# 分割线
        learn_rewards = []
        learn_states_for_ppo = []
        learn_states_for_common = []
        learn_actions = []
        learn_logprobs = []
        if 1500 / update_times_greedy > 200:
            for i in range(5):

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_reward_list' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_reward = list1*10
                else:
                    list_reward = [item for sublist in [list_reward, list1] for item in sublist]
                # print(len(list_reward))

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_current_action_buffer' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_action = list1
                else:
                    list_action = [item for sublist in [list_action, list1] for item in sublist]
                # print(len(list_action))

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_current_action_logprob' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_action_prob = list1
                else:
                    list_action_prob = [item for sublist in [list_action_prob, list1] for item in sublist]
                # print(len(list_action_prob))

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_current_stateppo' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_current_stateppo = list1
                else:
                    list_current_stateppo = [item for sublist in [list_current_stateppo, list1] for item in sublist]
                # print(len(list_current_stateppo))

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_state_for_common' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_current_state_for_common = list1
                else:
                    list_current_state_for_common = [item for sublist in [list_current_state_for_common, list1] for item in
                                                     sublist]

                file1_name = os.path.join(config.EXPERT_DIR, 'data_greedy_sim_task' + str(300 * (i + 1)) + '.npz')
                data = np.load(file1_name)
                list1 = data['list1_name']  # 数据格式依旧是numpy.array
                if i == 0:
                    list_task = list1
                else:
                    list_task = [item for sublist in [list_task, list1] for item in
                                                     sublist]
                # print(len(list_current_state_for_common))
            # print(len(list_current_stateppo),len(list_current_stateppo[0]),'xd')
            # list_reward = list_reward[0:100]
            # list_action = list_action[0:100]
            # list_action_prob = list_action_prob[0:100]
            # list_current_stateppo = list_current_stateppo[0:100]
            # list_current_state_for_common = list_current_state_for_common[0:100]

            # print(len(list_current_stateppo), len(list_current_stateppo[0]),'ddd')
            # print('??')
            # print(list_current_stateppo,'dadad')
            # for k in range(10):
            # print(len(learn_rewards), len(learn_states_for_ppo),len(learn_states_for_common),len(learn_actions))
            # print(len(list_current_stateppo),'list_current_stateppo')
            for k in range(len(list_current_stateppo)):
                # print('更新的k',k)
                # print(len(list_current_state_for_common[k]), list_current_state_for_common[k])
                # print(k,'kkk')
                # print(list_task[k])
                # print(list_task[k][0])
                vc_local = [[], [], []]
                vc_neigh = [[], [], []]
                ques = []
                new_ppo = np.zeros((3, 60))
                new_common = np.zeros((180))
                for i in range(3):
                    if i == 0:
                        ques.append(list_task[k][0])
                        # answer.append(ans_list1[k])
                    elif i == 1:
                        ques.append(list_task[k][1])
                        # answer.append(ans_list2[k])
                    elif i == 2:
                        ques.append(list_task[k][2])
                        # answer.append(ans_list3[k])
                    # print(type(list_current_stateppo[0]), len(list_current_stateppo[0]))
                    # print(len(list_current_state_for_common[0]))
                for m in range(3):
                    for n in range(30):
                        if n<30:
                            # print(list_current_stateppo[k][m][n],'ddd')
                            vc_local[m].append(list_current_stateppo[k][m][n])
                    for ii in range(30):
                        new_ppo[m][ii] = vc_local[m][ii]
                    # print(new_ppo,'???2')
                    att = self.policy.linear_layer(self.policy.transen(torch.tensor(towhee_embedding(ques[m][0])).view(1,1,768).to(self.device)).view(1, 768))
                    # print(att.to(device="cpu").detach().numpy()[0])
                    for ii in range(30):
                        new_ppo[m][ii+30] = att.to(device="cpu").detach().numpy()[0][ii]
                    # print(new_ppo,'ddd')
                #     print(list_current_stateppo[k][m])
                #     print(att)
                #     print(new_ppo[m])
                #     print('???')
                #
                # print(list_current_stateppo[k])
                # print(new_ppo)
                # print(list_current_stateppo[k],'lsd')
                list_current_stateppo[k] = new_ppo
                # print(new_ppo)
                # print(list_current_stateppo[k])
                # print('mid_output')
                # print(list_current_stateppo[k])
                # print(list_current_stateppo[k],'wdet')
                for ii in range(180):
                    if ii<60:
                        new_common[ii] = new_ppo[0][ii]
                    elif 60<=ii<120:
                        new_common[ii] = new_ppo[1][ii-60]
                    else:
                        new_common[ii] = new_ppo[2][ii-120]

                list_current_state_for_common[k] = new_common
                        # list_current_state_for_common[k] =
            # print(list_current_stateppo)
                    # print(list_action,'accc')
            print(1500/update_times_greedy,'更新所用知识数')
        # 421.0526315
        # print(list_current_stateppo)
        device = self.device
        # print(torch.from_numpy(list_current_stateppo[0][1]).to(device))

        if 1500/update_times_greedy > 200:
            # random_numbers = self.generate_unique_random_numbers(0, len(list_reward) - 1, 2000)
            if update_times_greedy == 1:
                random_numbers = []
                for i in range(1500):
                    random_numbers.append(i)
            else:
                random_numbers = self.generate_unique_random_numbers(0, len(list_reward) - 1, 1500/update_times_greedy)
            # random_numbers = self.generate_unique_random_numbers(0, 10, 5)
            # print(random_numbers,'random_numbers')
            device = self.device
            for i in random_numbers:
                learn_rewards.append(list_reward[i][index_for_large])
                # inputt = list_current_stateppo[i][index_for_large]
                midd = list_current_stateppo[i][index_for_large]
                # print(torch.tensor(midd, dtype=torch.double),'dxdx')
                # print(torch.from_numpy(midd),'xxxxxx')
                learn_states_for_ppo.append(torch.from_numpy(list_current_stateppo[i][index_for_large]).to(torch.float32).to(device))
                learn_states_for_common.append(torch.from_numpy(list_current_state_for_common[i]).to(torch.float32).to(device))
                # 专家动作直接使用原始动作（0: 缓存, 1: LLM）
                learn_actions.append(torch.tensor(int(list_action[i][index_for_large])).to(device))
                dist = Categorical(torch.tensor([1,0,0,0]).to(device))
                action = np.where(np.array([1,0,0,0]) == 1)[0][0]


                # print(list_action_prob[i][index_for_large],'dist:list_action_prob[i][index_for_large]')
                # print(action,'update_action')
                # print(action,'action_where')
                action_logprob = dist.log_prob(torch.tensor(action).to(device))
                # print(action_logprob,'action_logprob_get')
                learn_logprobs.append(action_logprob)
        print(len(learn_rewards),'lens')
        print(len(learn_actions), learn_actions,'wudi')
        # print(len(learn_states_for_common),len(learn_states_for_common[0]))
        # print(len(learn_states_for_ppo))
        # print(learn_states_for_ppo,'ddd')


        len1, len2, len3 = show_file_len(update_times)
        if len1 == 8 and len2 == 8 and len3 == 8:
            if index_for_large == 0:
                node_file = node1_file
            elif index_for_large == 1:
                node_file = node2_file
            else:
                node_file = node3_file
            print(len(node_file))
            for j in range(8):
                if 'reward' in node_file[j]:
                    file1_name = node_file[j]
                    while 1:
                        try:
                            data = np.load(file1_name)
                            break
                        except:
                            print('数据传输未完成，reward')

                    list1 = data['list1_name']  # 数据格式依旧是numpy.array
                    list_rewards_mem = list1

                if 'action_buffer' in node_file[j]:
                    file1_name = node_file[j]
                    while 1:
                        try:
                            data = np.load(file1_name)
                            break
                        except:
                            print('数据传输未完成，action_buffer')
                    list1 = data['list1_name']  # 数据格式依旧是numpy.array
                    list_action_mem = list1

                if 'action_logprob' in node_file[j]:
                    file1_name = node_file[j]
                    while 1:
                        try:
                            data = np.load(file1_name)
                            break
                        except:
                            print('数据传输未完成，action_logprob')
                    list1 = data['list1_name']  # 数据格式依旧是numpy.array
                    list_action_pro_mem = list1

                if 'stateppo' in node_file[j]:
                    file1_name = node_file[j]
                    while 1:
                        try:
                            data = np.load(file1_name)
                            break
                        except:
                            print('数据传输未完成，stateppo')
                    list1 = data['list1_name']  # 数据格式依旧是numpy.array
                    list_current_stateppo_mem = np.random.randn(300, 1, 60)

                if 'stateppo' in node_file[j]:
                    file1_name = node_file[j]
                    while 1:
                        try:
                            data = np.load(file1_name)
                            break
                        except:
                            print('数据传输未完成，stateppo')
                    list1 = data['list1_name']  # 数据格式依旧是numpy.array
                    list_current_state_for_common_mem = np.random.randn(300, 1, 180)



            # 取出来公共部分
            for x in range(3):
                if x == 0:
                    node_file = node1_file
                elif x == 1:
                    node_file = node2_file
                else:
                    node_file = node3_file
                for j in range(8):
                    if 'stateppo' in node_file[j]:
                        file1_name = node_file[j]
                        print(file1_name, 'stateppo')
                        while 1:
                            try:
                                data = np.load(file1_name)
                                break
                            except:
                                print('数据传输未完成，common_stateppo')
                        list1 = data['list1_name']  # 数据格式依旧是numpy.array
                        if x == 0:
                            list_states_mem = list1
                        else:
                            list_states_mem = np.concatenate((list_states_mem, list1), axis=1)
                    if 'task' in node_file[j]:
                        file1_name = node_file[j]
                        print(file1_name, 'task')
                        while 1:
                            try:
                                data = np.load(file1_name)
                                break
                            except:
                                print('数据传输未完成，common_task')
                        list1 = data['list1_name']  # 数据格式依旧是numpy.array
                        if x == 0:
                            list_tasks_mem = list1
                        else:
                            list_tasks_mem = np.concatenate((list_tasks_mem, list1), axis=1)

                # time.sleep(30)


            for k in range(len(list_current_stateppo_mem)):
                # print('更新的k',k)
                # print(len(list_current_state_for_common[k]), list_current_state_for_common[k])
                # print(k,'kkk')
                # print(list_task[k])
                # print(list_task[k][0])
                vc_local = [[], [], []]
                vc_neigh = [[], [], []]
                ques = []
                new_ppo_mem = np.zeros((3, 60))
                new_common_mem = np.zeros((180))
                for i in range(3):
                    if i == 0:
                        ques.append(list_tasks_mem[k][0])
                        # answer.append(ans_list1[k])
                    elif i == 1:
                        ques.append(list_tasks_mem[k][1])
                        # answer.append(ans_list2[k])
                    elif i == 2:
                        ques.append(list_tasks_mem[k][2])
                        # answer.append(ans_list3[k])
                    # print(type(list_current_stateppo[0]), len(list_current_stateppo[0]))
                    # print(len(list_current_state_for_common[0]))
                for m in range(3):
                    for n in range(30):
                        if n<30:
                            # print(list_current_stateppo[k][m][n],'ddd')
                            vc_local[m].append(list_states_mem[k][m][n])
                    for ii in range(30):
                        new_ppo_mem[m][ii] = vc_local[m][ii]
                    # print(new_ppo,'???2')
                    att = self.policy.linear_layer(self.policy.transen(torch.tensor(towhee_embedding(ques[m][0])).view(1,1,768).to(self.device)).view(1, 768))
                    # print(att.to(device="cpu").detach().numpy()[0])
                    for ii in range(30):
                        new_ppo_mem[m][ii+30] = att.to(device="cpu").detach().numpy()[0][ii]
                    # print(new_ppo,'ddd')
                #     print(list_current_stateppo[k][m])
                #     print(att)
                #     print(new_ppo[m])
                #     print('???')
                #
                # print(list_current_stateppo[k])
                # print(new_ppo)
                # print(list_current_stateppo[k])
                list_current_stateppo_mem[k][0] = new_ppo_mem[index_for_large]
                # list_current_stateppo_mem[k] = np.expand_dims(new_ppo_mem[index_for_large], axis=0)


                # print(list_current_stateppo[k])
                # print(list_current_stateppo[k],'wdet')
                for ii in range(180):
                    if ii<60:
                        new_common_mem[ii] = new_ppo_mem[0][ii]
                    elif 60<=ii<120:
                        new_common_mem[ii] = new_ppo_mem[1][ii-60]
                    else:
                        new_common_mem[ii] = new_ppo_mem[2][ii-120]

                list_current_state_for_common_mem[k][0] = new_common_mem

        print(len(list_rewards_mem), 'len(list_rewards_mem)')
        for l in range(len(list_rewards_mem)):
            learn_rewards.append(list_rewards_mem[l][0])
            learn_states_for_ppo.append(
                torch.from_numpy(list_current_stateppo_mem[l][0]).to(torch.float32).to(device))
            learn_states_for_common.append(
                torch.from_numpy(list_current_state_for_common_mem[l][0]).to(torch.float32).to(device))
            learn_actions.append(torch.tensor(list_action_mem[l][0]).to(device))
            learn_logprobs.append(torch.tensor(list_action_pro_mem[l][0][0]).to(device))

        print(len(learn_rewards),'lens2')
        print(len(learn_actions), learn_actions,'wudi2')

        rewards = []
        discounted_reward = 0
        print('start')
        # print(self.buffer.rewards, type(self.buffer.rewards),'self.buffer.rewards')

        # print(len(self.buffer.rewards), 'lenself.buffer.rewards')
        # for reward in reversed(self.buffer.rewards):
        #     # print(reward, 'reward')
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        for reward in reversed(learn_rewards):
            # print(reward, 'reward')
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        print('end')
        # Normalizing the rewards 归一化奖励值
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states_for_ppo = torch.stack(learn_states_for_ppo, dim=0).detach().to(self.device)
        old_states_for_common = torch.stack(learn_states_for_common, dim=0).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(learn_actions, dim=0)).detach().to(self.device)
        # print(learn_logprobs)
        # print(len(learn_logprobs))
        # print(torch.stack(learn_logprobs, dim=0).shape)
        old_logprobs = torch.squeeze(torch.stack(learn_logprobs, dim=0)).detach().to(self.device)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # 动作一直是这个，学习的是采取这个动作的概率，这个变化了。这两个动作的采取概率在最初学习时是相同的，学习一次网络对于相同的动作采取的概率输出就不同了。
            logprobs, states_for_common_values, dist_entropy = self.policy.evaluate(old_states_for_ppo,
                                                                                    old_states_for_common, old_actions,
                                                                                    old_logprobs)
            # match states_for_common_values tensor dimensions with rewards tensor
            states_for_common_values = torch.squeeze(states_for_common_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            # 比较新的动作选择概率和旧的差异是多少？
            # print(logprobs.shape,"logprobs")
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss替代损失函数
            # 优势函数等于奖励-状态价值函数
            advantages = rewards - states_for_common_values.detach()
            print(logprobs.shape,'logprobs')
            print(old_logprobs.detach().shape,'old_logprobs.detach()')
            # 动作选择概率函数不再变时surr1接近于0
            print(rewards.shape,'reward.shape')
            print(states_for_common_values.shape,'states_for_common_values')
            print(ratios.shape,"ratios.shape")
            print(advantages.shape,"advantages")
            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(states_for_common_values,
                                                                 rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        # 清空记录?
        self.buffer.clear()

        model_policy = self.policy_old

        torch.save(model_policy.state_dict(), 'model_network.pth')
        # self.policy_old.load_state_dict(torch.load("model_network.pth"))
        print('模型保存成功')

        return loss



    def save(self, model_path_and_name):
        torch.save(self.policy_old.state_dict(), model_path_and_name)

    def load(self, model_path_and_name):
        self.policy_old.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path_and_name, map_location=lambda storage, loc: storage))
