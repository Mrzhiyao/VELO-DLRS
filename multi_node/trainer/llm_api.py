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
time1 = time.time()
# Modify OpenAI's API key and API base to use vLLM's API server.
from openai import OpenAI
import random
import time

client = OpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
)
# models = openai.Model.list()
# model = models["data"][0]["id"]

def llm_solve_problem(input_context):
    random.seed(10)
    # ques = []
    start_time = time.time()
    if len(input_context)>= 2300:
        response = '您的问题太长，作为一个AI模型，可接受的问题长度有限，暂时无法回答这么长的问题'
        time_consume = time.time() - start_time
        return response, time_consume
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input_context,
                }
            ],
            model=config.LLM_MODEL,
            # stream=True,
        )
        # print('\n')
        response = chat_completion.choices[0].message.content
        time_consume = time.time() - start_time
    # print("Time consuming: {:.2f}s".format((time_consume)))
    # print(response)
    # print(towhee_embedding(response))
    return response, time_consume

# if __name__ == '__main__':
#     llm_solve_problem('iqoo12的配置是什么')