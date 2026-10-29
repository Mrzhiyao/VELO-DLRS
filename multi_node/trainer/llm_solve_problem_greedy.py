# from langchain.chat_models import ChatOpenAI
# import os
#
# chat_model = ChatOpenAI()
#
# chat_model.invoke("介绍下你自己")
import config
import random
import openai
import time
from towhee_embedding import towhee_embedding
from pymilvus import MilvusClient
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from openai import OpenAI
import random
import time

clientai = OpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
)



# time1 = time.time()
# # Modify OpenAI's API key and API base to use vLLM's API server.
#
# models = openai.Model.list()
# model = models["data"][0]["id"]

def llm_solve_problem(inputt, inputt_embedding):
    # 使用默认数据库 ‘default’,也可以自己建数据库
    connections.disconnect("default")
    connections.connect("default", host=config.LLM_CACHE_MILVUS_HOST, port=config.MILVUS_PORT)
    oasst1_val = Collection("oasst1_llm_result")
    client = MilvusClient(
        uri=config.milvus_uri(config.LLM_CACHE_MILVUS_HOST))
    # print('???')
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    result = oasst1_val.search([inputt_embedding], "vec", search_params, limit=3, output_fields=["message_id"])
    # print(result[0][0].entity.get('message_id'))
    father_id = result[0][0].entity.get('message_id')
    filter_content = "father_id==" + "\'" + str(father_id) + "\'"

    # print(filter_content,'filter_content')
    # print(filter_content,'filter_content')
    # result_sim = oasst1_val.search(label_vec, "vec", search_params, limit=1, output_fields=["id", "text"], expr=exprr)
    result_sim = client.search(data=[inputt_embedding], collection_name="oasst1_llm_result",
                               filter=filter_content, output_fields=["text", "vec", "time_consume"], search_params=search_params,
                               limit=16384)

    # result_sim = oasst1_val.query(expr="id!=0", filter=filter_content,
    #                        output_fields=["text", "vec"])
    # print(result_sim, father_id)
    if result_sim == [[]]:
        numberr = 0
        while 1:
            numberr = numberr + 1
            try:
                start_time = time.time()
                if len(inputt) >= 2300:
                    response = '您的问题太长，作为一个AI模型，可接受的问题长度有限，暂时无法回答这么长的问题'
                    time_consume = time.time() - start_time
                    text_embedding = towhee_embedding(response)
                else:
                    chat_completion = clientai.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": inputt,
                            }
                        ],
                        model=config.LLM_MODEL,
                        # stream=True,
                    )
                    # print('\n')
                    response = chat_completion.choices[0].message.content
                    time_consume = time.time() - start_time
                    text_embedding = towhee_embedding(response)
                # response, time_consume = llm_solve_problem(inputt)
                break
            except:
                print(numberr, '网络连接错误，重新连接中')

    else:
        response = result_sim[0][0]['entity']['text']
        text_embedding = result_sim[0][0]['entity']['vec']
        # time_consume = result_sim[0][0]['entity']['time_consume']
        time_consume = 7.2293

    return response, text_embedding, time_consume

# if __name__ == '__main__':
#     llm_solve_problem('iqoo12的配置是什么')