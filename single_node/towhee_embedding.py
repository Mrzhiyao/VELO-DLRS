import config
from towhee import triton_client

import numpy as np
import time
from towhee import pipe, ops, DataCollection
import random
from towhee import triton_client

client = triton_client.Client(url=config.TRITON_URL)

# 使用默认数据库 ‘default’,也可以自己建数据库

def towhee_embedding(inputt):
    res = client(inputt)
    out = res[0][0]
    return out



# if __name__ == '__main__':
#     # random.seed(10)
#     ques = []
#     n = 1
#     rep = 1
#     # c
#     start_time = time.time()
#     for i in range(n):
#         que_pro = '测试文本'
#         # print(que_pro)
#         out = towhee_embedding(que_pro)
#         print(out)
#     print('\n')
#     print("Time consuming: {:.2f}s".format((time.time() - start_time) / float(n * rep)))

