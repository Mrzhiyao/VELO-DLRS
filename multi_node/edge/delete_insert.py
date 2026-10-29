#
from towhee_embedding import towhee_embedding
import numpy
import time
from llm_api import llm_solve_problem
from utils import get_uuid, get_current_time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
# from pymilvus import  MilvusClient
#
# client = MilvusClient(
# )
def dinsert(client, task_array, collect, collection_name, input_id, result_sim, result_ans, result_ans_id, result_role, delay, result_label, maybe_zhishiku, send_record_list, collection_send):
    send_status = send_record_list
    # print(task_array)
    # print(result_role)
    if task_array != []:
        # print('任务是：',task_array[0])
        oasst1_val = collect
        # print(input_id,'deleteid')
        deleteid = input_id
        # deleteid = "445087024983140766"
        exprr = "id in [" + str(deleteid) + "]"
        # id    message_id    created_date    text    vec    role    lang    review_count    rank    text_answer    text_answerid
        que = oasst1_val.query(expr=exprr, output_fields=["id","message_id","created_date","text","vec","role","lang","review_count","rank","text_answer","text_answerid"])
        if que != []:
            for hit in que:
                record_message_id = hit['message_id']
                # record_created_date = hit['created_date']
                record_text = hit['text']
                record_vec = hit['vec']
                record_role = hit['role']
                record_lang = hit['lang']
                record_review_count = hit['review_count']
                record_rank = hit['rank']
                record_text_answer = hit['text_answer']
                record_text_answerid = hit['text_answerid']

                arecord_message_id = hit['message_id']
                arecord_text = hit['text']
                arecord_vec = hit['vec']
                arecord_role = hit['role']
                arecord_lang = hit['lang']
                arecord_review_count = hit['review_count']
                arecord_rank = hit['rank']
                arecord_text_answer = hit['text_answer']
                arecord_text_answerid = hit['text_answerid']
        else:
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            que_new = client.search(data=[towhee_embedding(task_array[0])], collection_name=collection_name,
                                       output_fields=["id", "message_id", "created_date", "text", "vec", "role", "lang",
                                                      "review_count", "rank", "text_answer", "text_answerid"],
                                       search_params=search_params,
                                       limit=1)
            for hits in que_new:
                for hit in hits:
                    deleteid = hit['id']
                    # print(hit['distance'])
                    record_message_id = hit['entity']['message_id']
                    # zhishiku_id = record_message_id
                    # record_created_date = hit['created_date']
                    record_text = hit['entity']['text']
                    maybe_zhishiku = record_text
                    record_vec = hit['entity']['vec']
                    record_role = hit['entity']['role']
                    result_role = record_role
                    record_lang = hit['entity']['lang']
                    record_review_count = hit['entity']['review_count']
                    record_rank = hit['entity']['rank']
                    record_text_answer = hit['entity']['text_answer']
                    record_text_answerid = hit['entity']['text_answerid']

                    arecord_message_id = hit['entity']['message_id']
                    arecord_text = hit['entity']['text']
                    arecord_vec = hit['entity']['vec']
                    arecord_role = hit['entity']['role']
                    arecord_lang = hit['entity']['lang']
                    arecord_review_count = hit['entity']['review_count']
                    arecord_rank = hit['entity']['rank']
                    arecord_text_answer = hit['entity']['text_answer']
                    arecord_text_answerid = hit['entity']['text_answerid']

        label_answer = result_label[0]
        label_vec_value = towhee_embedding(label_answer)
        label_vec = [label_vec_value]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        if record_rank == 'nan':
            record_rank = 0
        else:
            record_rank = float(record_rank)
        if result_role == 'prompter':
            filter_content = "message_id==" + "\'" + str(record_text_answerid) + "\'"
            result_sim = client.search(data=label_vec, collection_name=collection_name,
                                filter=filter_content, output_fields=["id", "text"], search_params=search_params,limit=16384)

            if result_sim == [[]]:
                # 为数据库内原有但没答案的问题补充答案
                while 1:
                    try:
                        print('sign_location')
                        qa_search = client.search(data=label_vec,
                                            collection_name=collection_name,
                                            output_fields=["id", "text","vec"],
                                            search_params=search_params, limit=1)
                        break
                    except:
                        print('incomplete query result')
                for hits in qa_search:
                    for hit in hits:
                        qa_text = hit['entity']['text']
                        qa_vec = hit['entity']['vec']
                record_created_date = get_current_time()
                qa_new_message_id = get_uuid()
                entities = [
                    [qa_new_message_id],  # 信息id
                    [record_created_date],  # field created_date
                    [qa_text],  # 文本内容
                    [qa_vec],  # pro/assistant
                    ['assistant'],
                    ['默认'],  #
                    [0],
                    [str(0)],
                    ['无'],
                    [' ']
                ]
                # insert_result =
                oasst1_val.insert(entities)
                record_rank = (-qa_search[0][0]['distance']-0.05*delay) + record_rank / 2
                out = (-qa_search[0][0]['distance'] - 0.05*delay)
                out_sim = -qa_search[0][0]['distance']

                exprr = "id in [" + str(deleteid) + "]"
                oasst1_val.delete(exprr)
                record_created_date = get_current_time()
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
                    [record_review_count+1],
                    [str(record_rank)],
                    [qa_text],
                    [qa_new_message_id]
                ]
                # insert_result =
                oasst1_val.insert(entities)
            else:
                # print(result_sim[0][0]['distance'], '任务相似性2')
                record_rank = (-result_sim[0][0]['distance']-0.05*delay) + record_rank / 2
                out = (-result_sim[0][0]['distance'] - 0.05*delay)
                out_sim = -result_sim[0][0]['distance']

                exprr = "id in [" + str(deleteid) + "]"
                oasst1_val.delete(exprr)
                record_created_date = get_current_time()
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
                    [record_review_count+1],
                    [str(record_rank)],
                    [record_text_answer],
                    [record_text_answerid]
                ]
                # insert_result =
                oasst1_val.insert(entities)
                # print('更新prompter成功')

            # 操作send_biao
            send_insert_message_id = get_uuid()
            send_insert_created_date = get_current_time()
            if len(task_array[0]) > 9000:
                neirong = task_array[0][:9000]
            else:
                neirong = task_array[0]
            # if send_status != []:
            #     if send_status[0][0] == 0:
            #         oasst1_send = collection_send
            #         # print(type(float(12)))
            #         entities_send_insert = [
            #             [send_insert_message_id],  # 信息id
            #             [record_created_date],  # field created_date
            #             [neirong],  # 文本内容
            #             [towhee_embedding(task_array[0])],  # 向量编码
            #             ['mr'],  # 语言
            #             [out],  # 用户满意度
            #             [send_status[0][1]],  # field text_answerid
            #         ]
            #         oasst1_send.insert(entities_send_insert)
            return out, out_sim, delay
        else:
            # 'assistant'
            # print('assistant处理')
            inputt = '有一些可能的相关参考信息，但不确定是否对你解决问题有帮助' + maybe_zhishiku + '\n' + '问题' + task_array[0]
            # print(inputt,'inputt')
            may_time = time.time()
            if len(inputt)>2300:
                time_consume = time.time()-may_time
                response=('输入太长，无法得到需要的答案')
            else:
                numberr = 0
                while 1:
                    numberr = numberr +1
                    try:
                        response, time_consume = llm_solve_problem(inputt)
                        break
                    except:
                        print(numberr, '网络连接错误，重新连接中')
            # 插入答案
            record_message_id = get_uuid()
            record_idvalue = record_message_id
            record_created_date = get_current_time()
            record_text= str(response)
            record_vec = towhee_embedding(response)
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
                                filter=filter_content, output_fields=["id", "text"], search_params=search_params,limit=16384)
            num_insert = 0
            while result_sim == [[]]:
                if num_insert != 0 and num_insert % 100 == 0:
                    print('插入的数据未找到,重新插入 delete—insert')
                    oasst1_val.insert(entities_answer)
                    oasst1_val.flush()
                result_sim = client.search(data=label_vec, collection_name=collection_name,
                                filter=filter_content, output_fields=["id", "text"], search_params=search_params,limit=16384)
                num_insert = num_insert + 1

            return_value = result_sim[0][0]['distance']
            # print(result_sim[0][0]['distance'])
            record_rank = (-result_sim[0][0]['distance'] - 0.05*time_consume - 0.05*delay)

            # 插入问题
            # 先插入问题，再插入答案 x 因为需要计算reward
            record_message_id = get_uuid()
            record_created_date = get_current_time()
            record_text= inputt
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

            # insert_result
            oasst1_val.insert(entities)
            # print('assistant插入成功')

            exprr = "id in [" + str(deleteid) + "]"
            oasst1_val.delete(exprr)
            arecord_created_date = get_current_time()
            if len(arecord_text) > 9000:
                arecord_text = arecord_text[:9000]
            if len(record_text_answer) > 9000:
                arecord_text_answer = arecord_text_answer[:9000]
            entities = [
                [arecord_message_id],  # 信息id
                [arecord_created_date],  # field created_date
                [arecord_text],  # 文本内容
                [arecord_vec],  # pro/assistant
                [arecord_role],
                [arecord_lang],  #
                [arecord_review_count + 1],
                [str(record_rank)],
                [arecord_text_answer],
                [arecord_text_answerid]
            ]
            # insert_result =
            oasst1_val.insert(entities)
            # print('更新assistant成功')

            # 操作send_biao
            send_insert_message_id = get_uuid()
            send_insert_created_date = get_current_time()
            if len(task_array[0]) > 9000:
                neirong = task_array[0][:9000]
            else:
                neirong = task_array[0]
            # if send_status != []:
            #     if send_status[0][0] == 0:
            #         oasst1_send = collection_send
            #         # print(type(float(12)))
            #         entities_send_insert = [
            #             [send_insert_message_id],  # 信息id
            #             [arecord_created_date],  # field created_date
            #             [neirong],  # 文本内容
            #             [towhee_embedding(task_array[0])],  # 向量编码
            #             ['mr'],  # 语言
            #             [record_rank],  # 用户满意度
            #             [send_status[0][1]],  # field text_answerid
            #         ]
            #         oasst1_send.insert(entities_send_insert)

            return record_rank, -return_value, time_consume+delay
        # oasst1_val.flush()
    #  还差两部分(打印奖励、给出新的状态。即reset?)

    else:
        print('无任务到达')
        return -12, 10, 0
