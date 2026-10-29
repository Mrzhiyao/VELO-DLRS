import numpy as np
import uuid
import datetime

def get_zerouuid():
    zero_idlist = np.random.randint(0, 1, 32)
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


def get_uuid():
    # return str(uuid.uuid1())
    return str(uuid.uuid1())[:-12] + str(uuid.uuid4())
def get_current_time():
    # 获取当前时间
    current_time = datetime.datetime.now()
    return str(current_time)

# if __name__ == '__main__':
#     print(get_current_time())