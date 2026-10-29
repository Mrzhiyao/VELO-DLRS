from ftplib import FTP
import config


def ftpconnect(host, username, password):
    ftp = FTP()
    ftp.connect(host, 21)
    ftp.login(username, password)
    return ftp


# 从本地上传文件到ftp
def uploadfile(ftp, remotepath, localpath):
    bufsize = 1024
    with open(localpath, 'rb') as fp:
        ftp.storbinary('STOR ' + remotepath, fp, bufsize)
    ftp.set_debuglevel(0)


def push_weights(localpath='./model_network.pth'):
    """把训练端的权重依次上传到每个边缘节点（用户名 ftpnode1 / ftpnode2 / ftpnode3）。"""
    for k, host in enumerate(config.EDGE_FTP_HOSTS):
        ftp = ftpconnect(host, config.FTP_USER_PREFIX + str(k + 1), config.FTP_PASSWORD)
        uploadfile(ftp, config.FTP_REMOTE_WEIGHT_PATH, localpath)
        ftp.quit()
        print('weights pushed to node', k + 1)


if __name__ == '__main__':
    push_weights()
