import config
import os
from datetime import datetime
import torch
import numpy as np
from module_ppo_vec import PPO
from module_env_vec import bigenv
from module_others_vec import new_arg, state_decompose
from dropcache import cache_drop
# import visdom
# viz = visdom.Visdom()
import warnings
warnings.filterwarnings("ignore")
# viz.line([0],[-1],win='reward',opts=dict(title='reward'))
# viz.line([0],[-1],win='loss',opts=dict(title='loss'))
# 接入数据库
# from towhee import triton_client
# client = triton_client.Client(url=config.TRITON_URL)
# from pymilvus import (！
#     connections,
#     utility,
#     FieldSchema,
#     CollectionSchema,
#     DataType,
#     Collection,
# )
import time
# 使用默认数据库 ‘default’,也可以自己建数据库
# In this program, you need only to modify the specialized hyperparmeters part

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
################ PPO hyperparameters ################
# 最大step次数
episode_timestep_length = 4000  # max steps in one episode
# 最大训练步数
total_timestep_length = 10  # maximum training steps
# 每2000打印一次log？
log_episode_interval = 2
log_timestep_interval = episode_timestep_length * log_episode_interval
# 每10000打印一次
printout_episode_interval = 10
printout_timestep_interval = episode_timestep_length * printout_episode_interval

# 保存间隔和网络更新间隔
save_timestep_interval = printout_timestep_interval * 5  # save model frequency
# update_timestep_interval = 5  # update policy
update_timestep_interval = 300  # update policy
# update_timestep_interval = episode_timestep_length * 4  # update policy

# 策略更新频率
K_epochs = 50         # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

####### initialize environment hyperparameters ######
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
args = new_arg()
############ Specialized hyperparameters ############
# 这个node是什么->中间层的单元数
args.actor_node=256
args.critic_node=384
# 智能体数量
args.agent_num=3
################# save setting file #################
# 结果保存
env_dir = "PPO_alg" + '/'
if not os.path.exists(env_dir):
    os.makedirs(env_dir)
# 每次运行程序都会创建新的log保存地址出来
env_num = len(next(os.walk(env_dir))[2])
env_name = env_dir + 'env_' + str(env_num) + '/'

if not os.path.exists(env_name):
    os.makedirs(env_name)
set_file_handler = open(env_dir+'env_'+str(env_num)+'.csv', 'w+')
set_file_handler.write('actor_node,{}\n'.format(args.actor_node))
set_file_handler.write('critic_node,{}\n'.format(args.critic_node))

set_file_handler.close()
###################### logging ######################
log_path_and_name = env_name + "log.csv"
print("Data logging at : " + log_path_and_name)
################### model-saving ####################
print("Model saving at : " + env_name)
################ training procedure #################
env = bigenv(args)
# initialize PPO agents
for index_for_large in np.arange(args.agent_num):
    #  locals() 返回字典类型的局部变量
    # 为每个智能体创建对应的字典空
    locals()['ppo_agent_'+str(index_for_large).zfill(3)] = PPO(env.state_dim_critic,
                                                               env.state_dim_actor, env.action_dim, args.actor_node, args.critic_node, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
    # print(locals())
# ppo_agent=PPO(3,2,lr_actor, lr_critic, gamma, K_epochs, eps_clip)
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("============================================================================================")
# logging file
log_handler = open(log_path_and_name, "w+")
log_handler.write('episode,timestep,reward\n')
# printing and logging variables
# log_sum_reward为什么是四维，和智能体数量不一致
log_sum_reward = np.zeros(args.agent_num+1)
# print(log_sum_reward.shape)
printout_sum_reward = np.zeros(args.agent_num+1)
#
episode_counter = 0
timestep_counter = 0
# 状态复位
# 状态复位对我们的实验有用吗？
# 没用应该是，因为训练开始以后，是没有终止状态的，即需要时刻赋予奖励值并更新。
# initial_data = get_ques()
start_random_number = 0
np.random.seed(start_random_number)
# initial_data = np.load('a.npy')
# ques_ansk = shuffle_data(initial_data, start_random_number)
# task_status = np.random.randint(0, 2, args.agent_num)
# 读取task_status
# data = np.load('task_status.npz')
# list1 = data['list1_name']  # 数据格式依旧是numpy.array
status_number = 0
# task_status = list1[status_number]
# status_number = status_number + 1
#
# ques = ques_ansk[0][0:sum(task_status)]
# answer = ques_ansk[1][0:sum(task_status)]
data = np.load(os.path.join(config.DATA_DIR, 'task_que_node1_0.npz'))
que_list1 = data['list1_name']  # 数据格式依旧是numpy.array
data = np.load(os.path.join(config.DATA_DIR, 'task_que_node2_0.npz'))
que_list2 = data['list1_name']  # 数据格式依旧是numpy.array
data = np.load(os.path.join(config.DATA_DIR, 'task_que_node3_0.npz'))
que_list3 = data['list1_name']  # 数据格式依旧是numpy.array

data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node1_0.npz'))
ans_list1 = data['list1_name']  # 数据格式依旧是numpy.array
data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node2_0.npz'))
ans_list2 = data['list1_name']  # 数据格式依旧是numpy.array
data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node3_0.npz'))
ans_list3 = data['list1_name']  # 数据格式依旧是numpy.array
task_status = []
ques = []
answer = []

for i in range(3):
    if i == 0:
        ques.append(que_list1[0])
        answer.append(ans_list1[0])
        if que_list1[0] == '':
            task_status.append(1)
        else:
            task_status.append(1)
    elif i == 1:
        ques.append(que_list2[0])
        answer.append(ans_list2[0])
        if que_list2[0] == '':
            task_status.append(1)
        else:
            task_status.append(1)
    elif i == 2:
        ques.append(que_list3[0])
        answer.append(ans_list3[0])
        if que_list3[0] == '':
            task_status.append(1)
        else:
            task_status.append(1)

current_state, task_array, send_nodes = env.reset(task_status, ques, answer)
# 状态为3，1
print(current_state.shape, 'current_state')
# 一共训练3e6次
# 训练最多执行1000步为一个周期？
# 21904 27903/3 = 9301
# 每个epoch遍历一次任务，但是可能任务未到达？
# 随机任务列表
# state_xieru = []
data_xieru_task = []
data_xieru_current_stateppo = []
data_xieru_state_for_common = []
data_xieru_current_action_buffer = []
data_xieru_current_action_logprob = []
data_xieru_reward_list = []
data_xieru_sim_list = []
data_xieru_delay_list = []
data_xieru_llm_use_yn = []
# 12
iter_dataset = 0
# while timestep_counter <= total_timestep_length:
qiwang_task_num = 0
actual_task_num = 0
current_episode_loss = 0
# update_times = 7
update_times = 1
while timestep_counter<13501:
    current_episode_reward = np.zeros(args.agent_num+1)
    # current_episode_loss = 0

    for _ in np.arange(60000):

        # 0 110
        # 1 010
        # 2 000
        if _ == 0:
            continue

        # 查看数据集是否遍历完一次
        if _+1 == len(que_list1):
            iter_dataset = iter_dataset + 1
            # 没有下一个分片时（例如只下载了示例数据）回到第 0 个分片
            if not os.path.exists(os.path.join(config.DATA_DIR, 'task_que_node1_' + str(iter_dataset) + '.npz')):
                iter_dataset = 0

            data = np.load(os.path.join(config.DATA_DIR, 'task_que_node1_' + str(iter_dataset) + '.npz'))
            que_list1 = data['list1_name']  # 数据格式依旧是numpy.array
            data = np.load(os.path.join(config.DATA_DIR, 'task_que_node2_' + str(iter_dataset) + '.npz'))
            que_list2 = data['list1_name']  # 数据格式依旧是numpy.array
            data = np.load(os.path.join(config.DATA_DIR, 'task_que_node3_' + str(iter_dataset) + '.npz'))
            que_list3 = data['list1_name']  # 数据格式依旧是numpy.array

            data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node1_' + str(iter_dataset) + '.npz'))
            ans_list1 = data['list1_name']  # 数据格式依旧是numpy.array
            data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node2_' + str(iter_dataset) + '.npz'))
            ans_list2 = data['list1_name']  # 数据格式依旧是numpy.array
            data = np.load(os.path.join(config.DATA_DIR, 'task_ans_node3_' + str(iter_dataset) + '.npz'))
            ans_list3 = data['list1_name']  # 数据格式依旧是numpy.array
            task_status = []
            ques = []
            answer = []
            for i in range(3):
                if i == 0:
                    ques.append(que_list1[0])
                    answer.append(ans_list1[0])
                    if que_list1[0] == '':
                        task_status.append(1)
                    else:
                        task_status.append(1)
                elif i == 1:
                    ques.append(que_list2[0])
                    answer.append(ans_list2[0])
                    if que_list2[0] == '':
                        task_status.append(1)
                    else:
                        task_status.append(1)
                elif i == 2:
                    ques.append(que_list3[0])
                    answer.append(ans_list3[0])
                    if que_list3[0] == '':
                        task_status.append(1)
                    else:
                        task_status.append(1)
            current_state, task_array, send_nodes = env.reset(task_status, ques, answer)
            # 状态为3，1
            break

        print(_,'timestap')
        # if _ < 1199 and episode_counter == 0:
        #     timestep_counter += 1
        #     continue
        # task_status = np.random.randint(0, 2, args.agent_num)
        current_task = []
        current_action = []
        current_action_buffer = []
        current_action_logprob = []
        current_stateppo = []
        task_length = current_state[-3:]
        # 状态搬移
        state_for_common = torch.FloatTensor(current_state).to(device)
        # 对智能体依次进行使用
        for index_for_large in np.arange(args.agent_num):
            # state_for
            # print(state_for_common[index_for_large],'state_for_common')
            # 把状态从cuda上搬下来，且拆开为单智能体的输入
            print(index_for_large)
            state_for_ppo = state_decompose(state_for_common, index_for_large)
            # print(state_for_ppo.shape,'shapeshapeshape')
            # print(state_for_ppo,'state_for_ppo')
            # 依次获取动作,多智能体强化学习的状态维数不同
            if index_for_large == 2:
                print(task_array,'要完成的任务')
            action_for_ppo, action_logprob_for_ppo, new_state = locals()[
                'ppo_agent_'+str(index_for_large).zfill(3)].select_action(timestep_counter, state_for_ppo, task_array[index_for_large],
                                                                          send_nodes[index_for_large], index_for_large)
            # print(new_state,type(new_state),'xxx')
            if index_for_large == 0:
                save_state_for_common = new_state
            else:
                save_state_for_common = torch.cat([save_state_for_common, new_state], 0)
            # print(save_state_for_common,type(save_state_for_common),save_state_for_common.shape,'yyy')
            # print(save_state_for_common,type(save_state_for_common))
            # print(torch.from_numpy(task_length),type(torch.from_numpy(task_length)))

            # new_state = torch.cat([state_for_ppo, att.squeeze()], 0)
            print(action_for_ppo, 'action')
            current_task.append(task_array[index_for_large])
            current_stateppo.append(state_for_ppo)
            current_action.append(action_for_ppo)
            current_action_buffer.append(np.array(action_for_ppo))
            current_action_logprob.append(np.array(action_logprob_for_ppo.cpu()))

        # print(state_for_common,type(state_for_common))
        # print(save_state_for_common, type(save_state_for_common), '??')
        # 当前动作添加到numpy数组里?
        current_action=np.array(current_action)
        # 获取下一时隙的状态
        task_status = []
        ques = []
        answer = []

        for i in range(3):
            if i == 0:
                ques.append(que_list1[_])
                answer.append(ans_list1[_])
                if que_list1[_] == '':
                    task_status.append(1)
                else:
                    task_status.append(1)
            elif i == 1:
                ques.append(que_list2[_])
                answer.append(ans_list2[_])
                if que_list2[_] == '':
                    task_status.append(1)
                else:
                    task_status.append(1)
            elif i == 2:
                ques.append(que_list3[_])
                answer.append(ans_list3[_])
                if que_list3[_] == '':
                    task_status.append(1)
                else:
                    task_status.append(1)
        print(_, task_status,ques,'ques')
        timestep_counter += 1

        current_state, reward_list, task_array, sim_list, delay_list, llm_use_yn, send_nodes, qiwang_task, actual_task = env.step(task_status, ques, answer, state_for_common, current_action)
        qiwang_task_num = qiwang_task_num + qiwang_task
        actual_task_num = actual_task_num + actual_task
        # 依次保存每个智能体的奖励值、公共状态？

        for index_for_large in np.arange(args.agent_num):
            # print(index_for_large,'index_for_large')
            locals()['ppo_agent_'+str(index_for_large).zfill(3)
                     ].buffer.rewards.append(reward_list[index_for_large])
            locals()['ppo_agent_'+str(index_for_large).zfill(3)
                     ].buffer.states_for_common.append(save_state_for_common)
        # print(state_for_common,'state_for_common')
        # print(locals(),'locals()')
        current_episode_reward += np.array(reward_list)
        # update PPO agent
        # 模型更新
        # data_xieru = []k8
        data_xieru_task.append(current_task)
        data_xieru_current_stateppo.append(current_stateppo)
        data_xieru_state_for_common.append(np.array(state_for_common.cpu().detach()))
        data_xieru_current_action_buffer.append(current_action_buffer)
        data_xieru_current_action_logprob.append(current_action_logprob)
        data_xieru_reward_list.append(np.array(reward_list))
        data_xieru_sim_list.append(np.array(sim_list))
        data_xieru_delay_list.append(np.array(delay_list))
        data_xieru_llm_use_yn.append(np.array(llm_use_yn))
        # sim_list, delay_list, llm_use_yn
        # state_xieru.append(data_xieru)
        # print(state_xieru,len(state_xieru),'state_xieru')
        if timestep_counter % update_timestep_interval == 0:
            file = open('qiwang_task_num.txt', 'a')
            # 写入内容
            file.write(str(qiwang_task_num))
            file.write('\n')
            # 关闭文件
            file.close()
            file = open('actual_task_num.txt', 'a')
            # 写入内容
            file.write(str(actual_task_num))
            file.write('\n')
            # 关闭文件
            file.close()
            qiwang_task_num = 0
            actual_task_num = 0
            buffer_name = 'data_greedy_sim_current_stateppo'+ str(timestep_counter) + '.npz'
            # print(buffer_name,'buffer_name')
            # print(data_xieru_current_stateppo[0],'data_xieru_current_stateppo')
            # print(len(data_xieru_current_stateppo[0]),'data_xieru_current_stateppo')
            np.savez(buffer_name, list1_name=data_xieru_current_stateppo)
            # state_xieru = []
            buffer_name = 'data_greedy_sim_state_for_common'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_state_for_common)
            buffer_name = 'data_greedy_sim_current_action_buffer'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_current_action_buffer)
            buffer_name = 'data_greedy_sim_current_action_logprob'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_current_action_logprob)
            buffer_name = 'data_greedy_sim_reward_list'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_reward_list)

            buffer_name = 'data_greedy_sim_sim_list'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_sim_list)
            buffer_name = 'data_greedy_sim_delay_list'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_delay_list)
            buffer_name = 'data_greedy_sim_llm_use_yn'+ str(timestep_counter) + '.npz'
            np.savez(buffer_name, list1_name=data_xieru_llm_use_yn)

            # buffer_name = 'data_greedy_sim_task_array'+ str(timestep_counter) + '.npz'
            # np.savez(buffer_name, list1_name=data_xieru_task)
            # print(len(data_xieru_current_stateppo))
            # print(len(data_xieru_state_for_common))
            # print(len(data_xieru_current_action_buffer))
            # print(len(data_xieru_current_action_logprob))
            # print(len(data_xieru_reward_list))
            data_xieru_task = []
            data_xieru_current_stateppo = []
            data_xieru_state_for_common = []
            data_xieru_current_action_buffer = []
            data_xieru_current_action_logprob = []

            if timestep_counter % (2000) == 0:
                print(data_xieru_reward_list, 'data_xieru_reward_list')
                # update_neighbor_use()
            data_xieru_reward_list = []
            data_xieru_sim_list = []
            data_xieru_delay_list = []
            data_xieru_llm_use_yn = []

            # for index_for_large in np.arange(args.agent_num):
            #     loss = locals()['ppo_agent_'+str(index_for_large).zfill(3)].update()
            #     current_episode_loss += np.array(torch.tensor(loss.clone().detach(), device='cpu'))
            #     viz.line([current_episode_loss / (timestep_counter/update_timestep_interval) ], [timestep_counter / update_timestep_interval], win='loss', update='append')
        # log file

        if timestep_counter % (update_timestep_interval) == 0 or timestep_counter == 2:
            # print(data_xieru_reward_list,'data_xieru_reward_list' )
            for index_for_large in np.arange(args.agent_num):
                loss = locals()['ppo_agent_'+str(index_for_large).zfill(3)].update(index_for_large, update_times)
                current_episode_loss += np.array(torch.tensor(sum(loss.clone().detach()), device='cpu'))
                # viz.line([current_episode_loss / (timestep_counter/update_timestep_interval) ], [timestep_counter / update_timestep_interval], win='loss', update='append')
            file = open('current_episode_loss.txt', 'a')
            # 写入内容
            file.write(str(current_episode_loss))
            file.write('\n')
            # 关闭文件
            file.close()
            update_times = update_times + 1
        if timestep_counter % (update_timestep_interval*8) == 0:
            cache_drop(str(timestep_counter / (update_timestep_interval*8)))
        if timestep_counter % log_timestep_interval == 0:
            log_avg_reward = log_sum_reward / log_timestep_interval
            log_avg_reward = np.around(log_avg_reward, 4)
            log_handler.write('{},{},{}\n'.format(
                str(episode_counter-log_episode_interval+2)+'-'+str(episode_counter+1), timestep_counter, log_avg_reward))
            log_handler.flush()
            log_sum_reward = np.zeros(args.agent_num+1)
        # print average reward
        if timestep_counter % printout_timestep_interval == 0:
            print_avg_reward = printout_sum_reward / printout_timestep_interval
            print_avg_reward = np.around(print_avg_reward, 2)
            print("Episode : {}-{} \t Timestep : {} \t Average Reward : {}".format(
                episode_counter-printout_episode_interval+2, episode_counter+1, timestep_counter, print_avg_reward))
            printout_sum_reward = np.zeros(args.agent_num+1)
        # save model weights
        if timestep_counter % save_timestep_interval == 0:
            print(
                "--------------------------------------------------------------------------------------------")
            print("saving model at : " + env_name)
            for index_for_large in np.arange(args.agent_num):
                model_path_and_name = env_name + \
                    "model_" + str(index_for_large) + ".pth"
                locals()['ppo_agent_'+str(index_for_large).zfill(3)
                         ].save(model_path_and_name)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(
                microsecond=0) - start_time)
            print(
                "--------------------------------------------------------------------------------------------")



    # viz.line([current_episode_reward[-1]/100],[timestep_counter/100],win='reward',update='append')
    episode_counter += 1
    log_sum_reward += current_episode_reward
    printout_sum_reward += current_episode_reward

log_handler.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")