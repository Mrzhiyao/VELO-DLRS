import config
import os
from datetime import datetime
import torch
import numpy as np
from module_ppo_vec import PPO
from module_env_vec import bigenv
from module_others_vec import new_arg, state_decompose
from dropcache import cache_drop
from show_file_len import show_file_len
from mv_file import mvfile
from show_file_name import show_file_name
from ftp_upload import push_weights
import warnings
warnings.filterwarnings("ignore")

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

# 运行时用到的目录（经验文件上传 / 归档）
for _d in ['node1', 'node2', 'node3', 'node1_log', 'node2_log', 'node3_log']:
    os.makedirs(_d, exist_ok=True)
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
# current_state, task_array, send_nodes = env.reset(task_status, ques, answer)
# # 状态为3，1
# print(current_state.shape, 'current_state')
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
update_times = 1
# update_times = 1
modelfile_path = './model_weight/model_network.pth'  # 文件路径
initial_time = 1706086425.6194556
while True:
    current_episode_reward = np.zeros(args.agent_num+1)
    timestep_counter = timestep_counter + 1
    # print(timestep_counter,'timestep_counter')
    len1, len2, len3 = show_file_len(update_times)
    # print(len1,len2,len3)
    # file1, file2, file3 = show_file_name()
    if len1 == 8 and len2 == 8 and len3 == 8:
        for index_for_large in np.arange(args.agent_num):
            print(index_for_large,'index_for_large')
            loss = locals()['ppo_agent_' + str(index_for_large).zfill(3)].update(index_for_large, update_times)
            print(loss,'loss')
            current_episode_loss += np.array(torch.tensor(sum(loss.clone().detach()), device='cpu'))
            file = open('current_episode_loss.txt', 'a')
            # 写入内容
            file.write(str(current_episode_loss))
            file.write('\n')
            # 关闭文件
            file.close()
        for index_for_large in np.arange(args.agent_num):
            if index_for_large == 0:
                source_dir = './node1'
                destination_dir = './node1_log'
                mvfile(source_dir, destination_dir, update_times)
            elif index_for_large == 1:
                source_dir = './node2'
                destination_dir = './node2_log'
                mvfile(source_dir, destination_dir, update_times)
            else:
                source_dir = './node3'
                destination_dir = './node3_log'
                mvfile(source_dir, destination_dir, update_times)
        # content = './model_network.pth'
        # send_file(content)
        # multi_send()
        # os.system('python ./send1.py')
        # os.system('python ./send2.py')
        # os.system('python ./send3.py')
        # 将聚合后的权重推送到各边缘节点
        push_weights('./model_network.pth')
        update_times = update_times + 1
        # 发送该记录到master中
        # for root, dirs, files in os.walk(path):
        #     for i in range(send_file_len):
        #         # if i >0:
        #         #     break
        #         # print(files[i])
        #         # print(type(files[i]))
        #         content = 'node1' + ' ' + path + files[i]
        #
        #         print(content)
        #         send_file(content)
        # # 移动该文件到目录log中
        # mvfile()
    # initial_time
    # weigth_time = os.path.getmtime(modelfile_path)
    # print(weigth_time, initial_time, 'weighttime')
    # if weigth_time == initial_time:
    #     pass
    # else:
    #     # print(weigth_time,'weigth_timeweigth_time')
    #     if weigth_time != initial_time:
    #         print(time.time())
    #
    #         initial_time = weigth_time


log_handler.close()

# print total training time
print("============================================================================================")
end_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)
print("Finished training at (GMT) : ", end_time)
print("Total training time  : ", end_time - start_time)
print("============================================================================================")