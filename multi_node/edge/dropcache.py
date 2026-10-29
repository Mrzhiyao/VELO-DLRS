#
import config
import math

import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus import MilvusClient
import time
# 使用默认数据库 ‘default’,也可以自己建数据库
# oasst1_val = Collection("oasst1_val_linshi")
#
# import random
# expr = "id in [4407]"
# oasst1_val.delete(expr)
def cache_drop(time):
    # 缓存剔除

    connections.disconnect("default")
    connections.connect("default", host=config.EDGE_MILVUS_HOST, port=config.MILVUS_PORT)
    oasst1_val = Collection(config.EDGE_COLLECTION)
    collection_name = config.EDGE_COLLECTION
    client = MilvusClient(
        uri=config.milvus_uri(config.EDGE_MILVUS_HOST)
    )
    oasst1_val.flush()
    exprr = "id>1"
    result_id = []
    result_count = []
    result_rank = []
    result_role = []
    while 1:
        result = oasst1_val.query(expr=exprr, output_fields=["id", "review_count", "rank","role"], limit=10000)
        if result == []:
            break
        for i in result:
            result_id.append(i["id"])
            result_count.append(i["review_count"])
            result_rank.append(float(i["rank"]))
            result_role.append(i["role"])
            max_id = i["id"]
        exprr = "id>" + str(max_id)
    if len(result_count) != 0 and len(result_rank) != 0:
        delete_id = []
        delete_rank = []
        delete_count = []
        delete_role = []
        ave_count = sum(result_count)/len(result_count)
        ave_rank = sum(result_rank)/len(result_rank)
        num = 0
        # print(math.floor(ave_count))
        # print(ave_rank)
        for i in range(len(result_count)):
            if result_count[i] != 0 and result_rank[i]<ave_rank:
                num = num + 1
                delete_id.append(result_id[i])
                delete_rank.append(result_rank[i])
                delete_count.append(result_count[i])
                delete_role.append(result_role[i])
                # print(i)
        # 记录保存

        buffer_name = 'delete_rank0' + str(time) + '.npz'
        np.savez(buffer_name, list1_name=delete_rank)
        buffer_name = 'delete_count0' + str(time) + '.npz'
        np.savez(buffer_name, list1_name=delete_count)
        print(num, '需删除的数量')
        file = open('dropcache_num0.txt', 'a')
        # 写入内容
        file.write(str(num))
        file.write('\n')
        # 关闭文件
        file.close()
        num = 0
        for i in delete_id:
            # print(i)
            exprr = "id in [" + str(i) + "]"
            oasst1_val.delete(exprr)
            if delete_role[num] == 'assistant':
                # 需要删除对应的问题
                delete_answerid = i
                # exprr = "text_answerid in [" + str(delete_answerid) + "]"
                # result = oasst1_val.query(expr=exprr, output_fields=["id", "review_count", "rank", "role"], limit=10000)
                exprr = "text_answerid in [\"" + str(delete_answerid) + "\"]"
                out = client.query(filter=exprr, output_fields=["id", "review_count", "rank", "role"], limit=10000,
                                   collection_name=collection_name)
                if out != []:
                    oasst1_val.delete(out[0]['id'])
            num = num + 1
        oasst1_val.flush()

# if __name__ == '__main__':
#     cache_drop(str(10000))

