import config
from towhee_embedding import towhee_embedding
import numpy
import time
# from llm_api import llm_solve_problem
from llm_solve_problem_greedy import llm_solve_problem
from utils import get_uuid, get_current_time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

def use_llm(client, collection_name, collect, inputt, result_label):

    label_answer = result_label
    label_vec_value = towhee_embedding(label_answer)
    input_embedding = towhee_embedding(inputt)
    label_vec = [label_vec_value]
    oasst1_val = collect
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    may_time = time.time()
    response, record_vec, time_consume = llm_solve_problem(inputt, input_embedding)
    connections.disconnect("default")
    if collection_name == 'two_input1_diffusion':
        connections.connect("default", host=config.MILVUS_HOSTS[0], port=config.MILVUS_PORT)
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=config.milvus_uri(config.MILVUS_HOSTS[0])
        )
        oasst1_val = Collection("two_input1_diffusion")
    elif collection_name == 'two_input2_diffusion':
        connections.connect("default", host=config.MILVUS_HOSTS[1], port=config.MILVUS_PORT)
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=config.milvus_uri(config.MILVUS_HOSTS[1])
        )
        oasst1_val = Collection("two_input2_diffusion")
    elif collection_name == 'two_input3_diffusion':
        connections.connect("default", host=config.MILVUS_HOSTS[2], port=config.MILVUS_PORT)
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=config.milvus_uri(config.MILVUS_HOSTS[2])
        )
        oasst1_val = Collection("two_input3_diffusion")


    # try:
    #     response, time_consume = llm_solve_problem(inputt)
    # except:
    #     time_consume = time.time() - may_time
    #     response = ('输入太长，无法得到需要的答案')
    # response, time_consume = llm_solve_problem(inputt)
    # 插入答案
    record_message_id = get_uuid()
    record_idvalue = record_message_id
    record_created_date = get_current_time()
    record_text = str(response)
    # record_vec = towhee_embedding(response)
    record_role = 'assistant'
    record_lang = '默认'
    record_review_count = 0
    record_rank = 0
    record_text_answer = '无'
    record_text_answerid = ' '
    if len(record_text) > 9000:
        record_text = record_text[:9000]
    if len(record_text_answer) > 9000:
        record_text_answer = record_text_answer[:9000]
    entities_answer = [
        [record_message_id],  # 信息id
        [record_created_date],  # field created_date
        [record_text],  # 文本内容
        [record_vec],  # pro/assistant
        [record_role],
        [record_lang],  #
        [record_review_count],
        [str(record_rank)],
        [record_text_answer],
        [record_text_answerid]
    ]
    oasst1_val.insert(entities_answer)

    # 查看和标准答案的差距
    filter_content = "message_id==" + "\'" + str(record_message_id) + "\'"
    # print(filter_content,'filter_content')
    # result_sim = oasst1_val.search(label_vec, "vec", search_params, limit=1, output_fields=["id", "text"], expr=exprr)
    result_sim = client.search(data=label_vec, collection_name=collection_name,
                               filter=filter_content, output_fields=["id", "text"], search_params=search_params,
                               limit=16384)
    num_insert = 0
    while result_sim == [[]]:
        if num_insert != 0 and num_insert % 1000 == 0:
            print('插入的数据未找到,重新插入 delete—insert')
            oasst1_val.insert(entities_answer)
            oasst1_val.flush()
        result_sim = client.search(data=label_vec, collection_name=collection_name,
                                   filter=filter_content, output_fields=["id", "text"], search_params=search_params,
                                   limit=16384)
        num_insert = num_insert + 1
        # # print(result_sim[0][-1]['distance'], '未找到阿')
        # record_rank = (-result_sim[0][0]['distance'] - time_consume)
        # return_value = record_rank
        # # print(res[0][0]['distance'])
    # else:
        # print(result_sim[0][0]['distance'])
    # llm激励
    record_rank = (-result_sim[0][0]['distance'] - 0.1*time_consume)
    return_value = record_rank

    # 插入问题
    record_message_id = get_uuid()
    record_created_date = get_current_time()
    record_text = inputt
    record_vec = towhee_embedding(inputt)
    record_role = 'prompter'
    record_lang = '默认'
    record_review_count = 0
    record_text_answer = response
    record_text_answerid = record_idvalue
    if len(record_text) > 9000:
        record_text = record_text[:9000]
    if len(record_text_answer) > 9000:
        record_text_answer = record_text_answer[:9000]
    entities = [
        [record_message_id],  # 信息id
        [record_created_date],  # field created_date
        [record_text],  # 文本内容
        [record_vec],  # pro/assistant
        [record_role],
        [record_lang],  #
        [record_review_count],
        [str(record_rank)],
        [record_text_answer],
        [record_text_answerid]
    ]
    oasst1_val.insert(entities)

    # print('assistant插入成功')

    return return_value, -result_sim[0][0]['distance'], time_consume
