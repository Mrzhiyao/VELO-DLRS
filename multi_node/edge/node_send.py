#encoding=utf-8
import config
import shutil
import socket
import os
import time
import json

# file_content = content.split(' ')[1]
def print_bar1(percent):
    bar = '\r' + '*' * int((percent * 100)) + ' %3.0f%%|' % (percent * 100) + '100%'
    print(bar, end='', flush=True)
def send_file(content):
    # tcp_continue_big_file_client.py

    client = socket.socket()
    client.connect((config.TRAINER_HOST, config.TRAINER_BASE_PORT + config.NODE_ID))
    file_content = content.split(' ')[1]
    file_path = file_content
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    dic = {"option": "1", "name": file_name, "file_size": file_size}
    client.send(json.dumps(dic).encode())
    file_seek = int(client.recv(100).decode())
    if file_seek == file_size:
        print('文件已经存在服务端，退出此次传输...')
    else:
        new_size = file_size - file_seek
        begin_size = new_size
        # 服务端表示准备好接收文件了，开始循环发送文件
        with open(file_path, 'rb') as f:
            f.seek(file_seek)
            while new_size:
                content = f.read(1024)
                client.send(content)
                new_size -= len(content)
                print_bar1(round((begin_size - new_size) / begin_size, 2))
                time.sleep(0.2)
            print('')
    client.close()

# if __name__ == '__main__':
#     path = './send_file_list/'
#     num = 0
#     for root, dirs, files in os.walk(path):
#         print(len(files))
#         num = len(files)
#         # send_file()
#     if num == 9:
#         for root, dirs, files in os.walk(path):
#             for i in range(num):
#                 # if i >0:
#                 #     break
#                 # print(files[i])=
#                 # print(type(files[i]))
#                 content = 'node1' + ' ' + path + files[i]
#                 print(content)
#                 send_file(content)


    # src_folder = './send_file_list/'
    #
    # dst_folder = './mv_file_list/'
    #
    # shutil.move(src_folder, dst_folder)