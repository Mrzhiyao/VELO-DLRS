# tcp_continue_big_file_server.py
import socket
import os
import json
import time
import argparse
import config

parser = argparse.ArgumentParser(description='接收某个边缘节点上传的经验数据')
parser.add_argument('--node', type=int, required=True, choices=[1, 2, 3])
node_id = parser.parse_args().node

server = socket.socket() # 默认就是tcp
server.bind((config.TRAINER_BIND_HOST, config.TRAINER_BASE_PORT + node_id))
server.listen()

os.makedirs('./node%d/' % node_id, exist_ok=True)
print('服务启动...', config.TRAINER_BASE_PORT + node_id)
while True:
    conn, addr = server.accept()
    print('新的客户端：', addr)
    while True:
        try:
            is_conn = True
            data = conn.recv(2048)
            dic = json.loads(data.decode())
            if dic.get('option') == '1':
                # base_path = os.path.dirname(os.path.abspath(__file__))
                base_path = './node%d/' % node_id
                file_path = os.path.join(base_path, dic.get('name'))
                if os.path.exists(file_path):
                    file_seek = os.path.getsize(file_path)
                else:
                    file_seek = 0
                # 将文件指针发送过去，同时也可以解决粘包
                conn.send(str(file_seek).encode())
                if file_seek == dic['file_size']:
                    print('文件已经传输完成，退出此次传输...')
                else:
                    # 重新设置需要接收的文件大小
                    new_size = dic['file_size'] - file_seek
                    # 准备接收发来的追加文件内容
                    print(file_path,'file_path')
                    with open(file_path, 'ab') as f:
                        while new_size:
                            content = conn.recv(1024)
                            f.write(content)
                            new_size -= len(content)
                            # 因为Python中recv()是阻塞的，只有连接断开或异常时，接收到的是b''空字节类型，因此需要判断这种情况就断开连接。
                            if content == b'':
                                is_conn = False
                                break
            if not is_conn:
                print('有连接客户端断开...')
                break
        except Exception as e:
            print('有连接出现异常断开：', str(e))
            break
    conn.close()
server.close()
