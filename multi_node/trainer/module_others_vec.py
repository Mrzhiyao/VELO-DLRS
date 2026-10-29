import argparse
import numpy as np
import torch

# You can add more variable definitions in def new_arg()
# and then change their values in module_train.py
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
def new_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_node', default=256, type=int)
    parser.add_argument('--critic_node', default=384, type=int)
    parser.add_argument('--agent_num', default=3, type=int)
    args=parser.parse_args()
    return args


def state_decompose(state_for_common, index):
    # print(state_for_common.shape,'state_for_common')
    state_for_ppo = [state_for_common[index]]
    mid = state_for_common[30 * index:30 * (index + 1)]
    # print([state_for_common[-3:]])
    # common_resoure = state_for_common[-3:]
    # print(mid.shape,common_resoure.shape)
    # mid = torch.cat((mid, common_resoure))
    state_for_ppo = torch.squeeze(mid)

    # print(state_for_ppo)
    # state_for_ppo_cpu = state_for_ppo.cpu()
    state_for_ppo_cpu = torch.tensor(state_for_ppo.clone().detach(), device='cpu')
    state_for_ppo = np.array(state_for_ppo_cpu)
    return state_for_ppo # has to be a horizontial numpy vector
    # [tensor(0., device='cuda:0')] state_for_ppo
# [tensor(0., device='cuda:0')] state_for_ppo