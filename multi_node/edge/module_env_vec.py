import config
import numpy as np
import random
import time
# 编码部分
from llm_use import use_llm
from delete_insert import dinsert
from towhee_embedding import towhee_embedding
# from towhee_embedding_search import towhee_embedding_search
from utils import get_uuid, get_current_time
# from ssh_send import ssh_connect
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
# connections.connect("default", host=config.MILVUS_HOSTS[0], port=config.MILVUS_PORT)
# from pymilvus import MilvusClient
# client = MilvusClient(
# )

np.random.seed(101)
class bigenv(object):
    def __init__(self, args):


        # a simple example, three agents have reward equaling the minus of their observations
        # the goal is to maximize the sum rewards.

        self.agent_num = args.agent_num
        # 假设寻找任务的时候，寻找最相似的10个
        # 个体部分：通过节点缓存任务的高评分(向量相似性、类型、访问频率、结果平均满意度、-平均成本决定加权决定)数据
        # 还差了个部分，传输目标节点的记录->
        # 公共部分用：LLM需要处理的任务队列情况。任务数？
        # 10*4+1
        self.state = np.random.randn(180)
        # self.state = np.array([0,0,0]) # the state has to be a horizontial numpy vector
        # print(self.state1.shape,self.state1)
        # 状态和观测的区别？
        # 状态指的是空间内所有智能体状态的集合，观测指的是一个智能体网络的输入
        self.state_dim_critic = 180  # dim for critic is also the dim for state
        # actor网络的状态是维度为3的一个vector
        self.state_dim_actor = 60  # dim for actor is also the dim for observation
        # 几种可能的动作
        self.action_dim = 2  # 10个利用大模型的、一个直接请求LLM的，一个传送到其他节点的，一个保持静止的
        # 利用本地缓存命中
        # 将任务发送选择发送到另一节点，并产生通信成本与延时(内容差异大的任务) 不进行新缓存
        # 利用本地缓存中的知识库进行补充后利用 LLM 模型返回答案 进行缓存
        # 无匹配知识库，直接利用 LLM 返回答案 进行缓存
        # 静置
        self.search_delay = []
        self.state_id = [[], [], []]
        self.roles_record = [[], [], []]
        self.answer_content = [[], [], []]
        self.answer_id = [[], [], []]
        self.pa_content = [[], [], []]
        self.send_text = [[], [], []]
        self.send_vec = [[], [], []]
        self.send_nodes = [[], [], []]
        # 标志节点上的哪个任务是来自其他节点
        self.send_signal = [[], [], []]
        self.send_from_id = [[], [], []]

        self.task_array = [[],[],[]]
        self.answer_array = [[], [], []]
        self.task_send_array = [[], [], []]
        self.task_send_array_answer = [[], [], []]

        self.task_array_embedding = [[],[],[]]
        self.llm_array = []
        # 一个节点请求其他节点的记录
        self.neighbor_code = [[[],[]],[[],[]],[[],[]]]
        self.neighbor_code_vec = [[], [], []]
        self.neighbor_code_similarity = [[], [], []]
        # 每个节点存10条记录？因为传递过去的节点三可以把处理记录送回来的
        # 所以一个任务可以有向量、结果满意度返回
        # 所以传递表只保存最高的10条？
        # 还是说这个也找个数据库存一下？这个有没有必要？存了的话->搜索最匹配的10个？错，这个还是要以满意度为标准的，因此不需要太多。
        # 存10个最高满意度的记录？更满意的会逐出旧的内容
        # 再追加20条、（相似度、满意度）
        # LLM任务队列中时延超过要求的任务自动失效


    def search(self,search_vec, collection):
        # print(vectors_to_search, len(vectors_to_search))

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        oasst1_val = collection
        result = oasst1_val.search(search_vec, "vec", search_params, limit=10,
                                   output_fields=["id", "text","role", "review_count", "rank", "text_answer","text_answerid"])
        # result = oasst1_val.search(search_vec, "title_vector", search_params, limit=1,
        #                            aaaaaaaaaaaaaaaaoutput_fields=["title_vector"])
        ids = []
        texts = []
        roles = []
        review_counts = []
        ranks = []
        text_answers = []
        text_answerids = []
        distances = []
        for hits in result:
            for hit in hits:
                # print(hit.entity.get('text'))
                # print(hit.entity.get('id'))
                # print(hit)
                distance = hit.distance
                # print(distance,'distance')
                distances.append(distance)
                # print(hit.distance, hit.entity.get('review_count'))
                id = hit.entity.get('id')
                ids.append(id)
                # print(id)

                text = hit.entity.get('text')
                texts.append(text)
                role = hit.entity.get('role')
                roles.append(role)
                review_count = hit.entity.get('review_count')
                review_counts.append(review_count)
                rank = hit.entity.get('rank')
                if rank == 'nan':
                    rank = 0
                ranks.append(rank)
                text_answer = hit.entity.get('text_answer')
                text_answers.append(text_answer)
                text_answerid = hit.entity.get('text_answerid')
                text_answerids.append(text_answerid)

        return distances, ids, texts, roles, review_counts, ranks, text_answers, text_answerids


    def get_zerouuid(self):
        zero_idlist = np.random.randint(0,1,32)
        zero_uuid = ''
        for i in range(32):
            zero_uuid = zero_uuid + str(zero_idlist[i])
            if i == 7:
                zero_uuid = zero_uuid + '-'
            if i == 11:
                zero_uuid = zero_uuid + '-'
            if i == 15:
                zero_uuid = zero_uuid + '-'
            if i == 19:
                zero_uuid = zero_uuid + '-'

        return zero_uuid

    def step(self, task_status, ques, answer, state_for_common, action):
        qiwang_task = 0
        actual_task = 0
        # 还有一个超过时延的数据筛选
        # 任务完成延时，完成任务数，完成任务数是正的，越大越好。
        # 任务传输成本
        # 任务完成时延
        dt = np.zeros(self.agent_num)
        dloss = np.zeros(self.agent_num)
        dcost = np.zeros(self.agent_num)

        # 计算奖励值和下一时隙的状态值
        # reward=系统来看：-当前时隙下任务完成的平均时延-处理任务的差距-向量数据库成本(cpu算力和内存使用)
        reward_list = np.zeros(self.agent_num + 1)  # reward_list has to be a horizontial numpy vector
        sim_list = np.zeros(self.agent_num)
        delay_list = np.zeros(self.agent_num)
        llm_use_yn = np.zeros(self.agent_num)

        # reward_list[0:-1] = - np.random.randn(3)
        # for i in range(self.agent_num):
        #     reward_list[i] = -dt[i] - dloss[i] - dcost[i]
        print(self.agent_num,'self.agent_num')
        for i in np.arange(self.agent_num):
            # try:
            #     print(self.task_array[i][0], self.answer_array[i][0],'q和a',i)
            # except:
            #     print(self.task_array[i], self.answer_array[i],'无任务')
            connections.disconnect("default")
            connections.connect("default", host=config.EDGE_MILVUS_HOST, port=config.MILVUS_PORT)
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=config.milvus_uri(config.EDGE_MILVUS_HOST)
            )
            collection_name1 = config.EDGE_COLLECTION
            collection1 = Collection(config.EDGE_COLLECTION)

            collection_send1 = Collection(config.EDGE_SEND_COLLECTION)
            collection_step = collection1
            collection_name = collection_name1
            collection_send = collection_send1

            state_agent = state_for_common[30*i:30*(i+1)]
            # 相似度、类型、频率、评分*4 + 相似度、排序后评分*2

            # print(self.state_id)
            if action[i] == 0:
                value = -1000
                suoyin = 0
                # 0 1 2 3
                # 4 5 6 7
                # 8 9 10 11

                for tt in range(30):
                    if tt % 3 != 0:
                        continue
                    # print(tt,'tt')
                    if -state_agent[tt] * 3 + state_agent[tt + 2] * 0.1 > value:
                        value = -state_agent[tt] * 3 + state_agent[tt + 2] * 0.1
                        suoyin = tt
                # sorted_index = np.argsort(value)
                action_select = int(suoyin / 3)
                print(action_select, 'action_select')
                # 数据库处理方式：
                # 直接返回：调用频率+1，重新计算平均满意度
                # prompt+LLM：插入问题和答案两条记录。
                # state_agent[4*action[i]]为状态的开头
                import random
                # 问题:更新满意度 频率 答案:更新满意度 频率
                # 先删除再插入
                deleteid = self.state_id[i][action_select]
                # print(deleteid,'request_id')
                # 找出相似度
                result_sim = state_agent[3 * action_select]
                # 找出回答内容
                result_ans = self.answer_content[i][action_select]
                # 找出回答id
                result_ans_id = self.answer_id[i][action_select]
                # 找出答案类型
                result_role = self.roles_record[i][action_select]
                # 找到答案/
                maybe_zhishiku = self.pa_content[i][action_select]

                # delay = 0.1
                # print(self.task_array[i],'self.task_array[i]')
                # print(deleteid,'idd')
                # 相似度
                reward_mid, sim_, delay_ = dinsert(client, self.task_array[i], collection_step, collection_name,
                                                       deleteid, result_sim, result_ans, result_ans_id, result_role,
                                                       self.search_delay[i], self.answer_array[i], maybe_zhishiku,
                                                       self.send_signal[i], collection_send)
                reward_list[i] = reward_mid * 10
                sim_list[i] = sim_
                delay_list[i] = delay_

                if self.task_array[i] != []:
                    qiwang_task = qiwang_task + 1
                    actual_task = actual_task + 1
                else:
                    qiwang_task = qiwang_task + 1
                    # self.state[(self.state_dim_actor-1)*i:(self.state_dim_actor-1)*(i+1)] = np.random.randn(self.state_dim_actor-1)
            elif action[i] == 1:

                # 直接利用LLM输出
                # 数据库处理方式：插入这条记录
                print(i,'iiii')
                if self.task_array[i] != []:
                    llm_use_yn[i] = 1
                    reward_mid, sim_, consume = use_llm(client, collection_name, collection_step,
                                                      self.task_array[i][0], self.answer_array[i][0])
                    reward_list[i] = (reward_mid-0.6*llm_use_yn[i])*10
                    delay_list[i] = consume
                    sim_list[i] = sim_
                    # 判断任务是否来自其他节点
                    qiwang_task = qiwang_task + 1
                    actual_task = actual_task + 1
                else:
                    qiwang_task = qiwang_task + 1
                    reward_list[i] = -12
            print('i:', i,reward_list[i], action[i])
        reward_list[-1] = np.sum(reward_list[0:-1])  # the last term is the sum reward among agents
        print(reward_list[-1],'sum_reward')

        self.search_delay = []

        # 去除已经完成的任务
        self.task_array_new = [[],[],[]]
        self.task_array_answer_new = [[],[],[]]
        for i in range(self.agent_num):
            if action[i] != 3:
                self.task_array_new[i] = self.task_array[i][1:]
            else:
                self.task_array_new[i] = self.task_array[i]
        self.task_array = self.task_array_new

        for i in range(self.agent_num):
            if action[i] != 3:
                self.task_array_answer_new[i] = self.answer_array[i][1:]
            else:
                self.task_array_answer_new[i] = self.answer_array[i]
        self.answer_array = self.task_array_answer_new

        # print(self.task_array, 'end')
        self.task_send_array = [[], [], []]
        self.task_send_array_answer = [[], [], []]
        self.send_text = [[], [], []]
        self.send_vec = [[], [], []]
        self.send_nodes = [[], [], []]
        self.send_from_id = [[], [], []]

        # print(self.task_array, 'self.task_array_xian')

        self.state = self.get_new_state(task_status, ques, answer, Collection)
        print(self.task_array, 'self.task_array', len(self.task_array))

        return self.state, reward_list, self.task_array, sim_list, delay_list, llm_use_yn, self.send_nodes, qiwang_task, actual_task


    def get_new_state(self, task_status, ques, answer, Collection):
        # print(task_status,'task_status')
        # print(len(ques),'ques')
        self.state = np.random.randn(90)
        self.task_array_embedding = [[],[],[]]
        self.state_id = [[], [], []]
        self.roles_record = [[], [], []]
        self.answer_content = [[], [], []]
        self.answer_id = [[], [], []]
        self.pa_content = [[], [], []]

        self.task_status = task_status
        ques_list = ques
        answer_list = answer
        # print(ques_list)
        # 三个智能体分别使用3个集合
        # 生成任务
        # 观察是否有新成的任务
        # print(self.task_status)
        # j = 0
        for i in range(self.agent_num):

            if self.task_status[i] == 1:
                self.task_array[i].insert(0, ques_list[i])
                # 因为新加了任务，所以需要更新任务的编号
                try:
                    for b in range(len(self.send_signal[i])):
                        self.send_signal[i][b][0] = self.send_signal[i][b][0] + 1
                except:
                    print('传输任务列表无内容')
                self.answer_array[i].insert(0,answer_list[i])
                # j = j + 1
                # task.append(ques_list[i])
            if self.task_array[i] != []:
                self.task_array_embedding[i].append(towhee_embedding(self.task_array[i][0]))
                # task_embeding.append(self.towhee_embedding(ques_list[i]))
            # else:
            #     task.append('无任务到达')
            #     task_embeding.append([[random.random() for _ in range(768)]])
        # 计算相似性
        # state的起点
        quotidian = 0
        # print(len(self.task_array_embedding[0]),len(self.task_array_embedding[1]),len(self.task_array_embedding[2]),'self.task_array_embedding')
        for i in range(self.agent_num):
            connections.disconnect("default")
            connections.connect("default", host=config.EDGE_MILVUS_HOST, port=config.MILVUS_PORT)
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=config.milvus_uri(config.EDGE_MILVUS_HOST)
            )
            collection_name1 = config.EDGE_COLLECTION
            collection1 = Collection(config.EDGE_COLLECTION)
            collection = collection1
            collection_send1 = Collection(config.EDGE_SEND_COLLECTION)
            collection_step = collection1
            collection_name = collection_name1
            collection_send = collection_send1

            if self.task_array[i] != []:
                start_time = time.time()
                distances, ids, texts, roles, review_counts, ranks, text_answers, text_answerids = self.search(
                    self.task_array_embedding[i], collection)
                # print(distances,'distances')
                for j in range(10):
                    # print(quotidian,'quotidian',i,j)
                    self.state[quotidian] = distances[j]
                    quotidian = quotidian + 1
                    if roles[j] == 'prompter':
                        self.state[quotidian] = 0
                    else:
                        self.state[quotidian] = 1
                    quotidian = quotidian + 1
                    self.state[quotidian] = review_counts[j]
                    quotidian = quotidian + 1
                    # self.state[quotidian] = ranks[j]
                    # quotidian = quotidian + 1
                    self.pa_content[i].append(texts[j])
                    self.state_id[i].append(ids[j])
                    self.roles_record[i].append(roles[j])
                    self.answer_content[i].append(text_answers[j])
                    self.answer_id[i].append(text_answerids[j])
                # print(self.task_array_embedding[i], 'show_yichang')
                # similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index = self.search_neighbor(
                #     symbol=i+1, neirong=self.task_array_embedding[i])
                # 补充边缘节点处理信息到state里
                time_consume = time.time() - start_time
                self.search_delay.append(time_consume)
            else:  # 该节点现在无任务
                start_time = time.time()
                for j in range(10):
                    self.state[quotidian] = 10
                    quotidian = quotidian + 1
                    self.state[quotidian] = 2  # 无内容
                    quotidian = quotidian + 1
                    self.state[quotidian] = 0
                    quotidian = quotidian + 1
                    # self.state[quotidian] = -30
                    # quotidian = quotidian + 1
                    self.pa_content[i].append('无知识库')
                    self.state_id[i].append(self.get_zerouuid())
                    self.roles_record[i].append('prompter')
                    self.answer_content[i].append('无答案')
                    self.answer_id[i].append(' ')
                # similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index = self.search_neighbor(
                #     symbol=i+1,
                #     neirong=[[random.randint(1, 1) for _ in range(768)]])
                # 补充边缘节点处理信息到state里
                # print('sds')
                time_consume = time.time() - start_time
                self.search_delay.append(time_consume)
        # self.state[-1] = 0
        return self.state
        # print(self.state)
    def reset(self, task_status, ques, answer):
        self.state = np.random.randn(90)
        task = []
        task_embeding = []
        self.task_array = [[],[],[]]
        self.task_status = task_status
        ques_list = ques
        answer_list = answer
        # print(ques_list)
        # 三个智能体分别使用3个集合
        # 生成任务
        # 观察是否有新成的任务
        print(self.task_status)
        for i in range(self.agent_num):
            if self.task_status[i] == 1:
                self.task_array[i].insert(0, ques_list[i])
                self.answer_array[i].insert(0, answer_list[i])
                # task.append(ques_list[i])
                self.task_array_embedding[i].append(towhee_embedding(self.task_array[i][0]))
                # task_embeding.append(self.towhee_embedding(ques_list[i]))
            # else:
            #     task.append('无任务到达')
            #     task_embeding.append([[random.random() for _ in range(768)]])
        # 计算相似性
        # state的起点
        quotidian = 0
        for i in range(self.agent_num):
            if i == 0:
                connections.disconnect("default")
                connections.connect("default", host=config.EDGE_MILVUS_HOST, port=config.MILVUS_PORT)
                from pymilvus import MilvusClient
                client = MilvusClient(
                    uri=config.milvus_uri(config.EDGE_MILVUS_HOST)
                )
                collection_name1 = config.EDGE_COLLECTION
                collection1 = Collection(config.EDGE_COLLECTION)
                collection = collection1
                collection_send1 = Collection(config.EDGE_SEND_COLLECTION)
                collection_step = collection1
                collection_name = collection_name1
                collection_send = collection_send1

            # print(self.task_array[i],'self.task_array[i]')
            if self.task_array[i] != []:
                # print(self.task_array[i],'self.task_array[i]')
                # print(self.task_array_embedding[i], 'embedding')
                start_time = time.time()
                # print(self.task_array_embedding[i], collection,'self.task_array_embedding[i], collection')
                distances, ids, texts, roles, review_counts, ranks, text_answers, text_answerids = self.search(self.task_array_embedding[i], collection)
                # 如果向量数据库中无内容？
                for j in range(10):
                    self.state[quotidian] = distances[j]
                    quotidian = quotidian + 1
                    if roles[j] == 'prompter':
                        self.state[quotidian] = 0
                    else:
                        self.state[quotidian] = 1
                    quotidian = quotidian + 1
                    self.state[quotidian] = review_counts[j]
                    quotidian = quotidian + 1
                    # self.state[quotidian] = ranks[j]
                    # quotidian = quotidian + 1
                    self.pa_content[i].append(texts[j])
                    self.state_id[i].append(ids[j])
                    self.roles_record[i].append(roles[j])
                    self.answer_content[i].append(text_answers[j])
                    self.answer_id[i].append(text_answerids[j])

                time_consume = time.time() - start_time
                self.search_delay.append(time_consume)
            else:  # 该节点现在无任务
                start_time = time.time()
                for j in range(10):
                    self.state[quotidian] = 10
                    quotidian = quotidian + 1
                    self.state[quotidian] = 2  # 无内容
                    quotidian = quotidian + 1
                    self.state[quotidian] = 0
                    quotidian = quotidian + 1
                    # self.state[quotidian] = -30
                    # quotidian = quotidian + 1
                    self.pa_content[i].append('无知识库')
                    self.state_id[i].append(self.get_zerouuid())
                    self.roles_record[i].append('prompter')
                    self.answer_content[i].append('无答案')
                    self.answer_id[i].append(' ')
                # similarity, rank_value, id, vec, text, send_node, rank_sorted, sorted_index = self.search_neighbor(symbol=i+1,
                #                                                                                                    neirong=[[random.randint(1,1) for _ in range(768)]])
        # 通过动作，更新大模型需要处理的LLM序列中的任务数
        # 起始为0
        # self.state[-1] = 0
        # print(len(self.state))
        # print(len(self.task_array[0]),len(self.task_array[1]),len(self.task_array[2]))
        # self.state[-1] = len(self.task_array[2])
        # self.state[-2] = len(self.task_array[1])
        # self.state[-3] = len(self.task_array[0])
        return self.state, self.task_array, self.send_nodes

# if __name__ == '__main__':
#     args = new_arg()
#     args.agent_num = 3
#     x = bigenv(args)
#     action = np.array([20, 2, 3])
#     # x.step(action)
#     from pymilvus import (
#         connections,
#         utility,
#         FieldSchema,
#         CollectionSchema,
#         DataType,
#         Collection,
#     )
#
#     import time
#     # 使用默认数据库 ‘default’,也可以自己建数据库
#
#     state = x.reset(Collection)
#     print(state.shape)
#     x.step(state, action)