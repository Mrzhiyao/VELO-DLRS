"""
VELO / DLRS 运行配置。

所有地址、密钥、路径都从环境变量读取（见 .env.example），代码里不再硬编码。
单机版、多机训练端、多机边缘端共用这一个文件。
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """读取仓库根目录的 .env（KEY=VALUE 格式），已存在的环境变量优先。"""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(ROOT, '.env'))


def _env(name, default):
    return os.environ.get(name, default)


def _env_list(name, default):
    return [x.strip() for x in _env(name, default).split(',') if x.strip()]


def _path(name, default):
    # 相对路径一律相对于仓库根目录，与在哪个子目录启动脚本无关
    return os.path.join(ROOT, _env(name, default))


# ---------------- 数据 ----------------
# 任务数据集目录，包含 task_que_node{1,2,3}_{shard}.npz / task_ans_node{1,2,3}_{shard}.npz
DATA_DIR = _path('VELO_DATA_DIR', 'data/tasks')
# 专家演示（Greedy 策略经验）目录，包含 data_greedy_sim_*{300..1500}.npz
EXPERT_DIR = _path('VELO_EXPERT_DIR', 'data/expert')

# ---------------- Milvus ----------------
MILVUS_PORT = _env('VELO_MILVUS_PORT', '19530')
# 单机版 / 训练端：三个边缘节点各自的 Milvus 地址（按节点顺序）
MILVUS_HOSTS = _env_list('VELO_MILVUS_HOSTS', '127.0.0.1,127.0.0.1,127.0.0.1')
# 多机边缘端：本节点的 Milvus 地址
EDGE_MILVUS_HOST = _env('VELO_EDGE_MILVUS_HOST', '127.0.0.1')
# 存放 LLM 历史回答的 Milvus（collection: oasst1_llm_result）
LLM_CACHE_MILVUS_HOST = _env('VELO_LLM_CACHE_MILVUS_HOST', '127.0.0.1')


def milvus_uri(host):
    return 'http://{}:{}'.format(host, MILVUS_PORT)


# ---------------- Embedding (Towhee + Triton) ----------------
TRITON_URL = _env('VELO_TRITON_URL', 'localhost:8010')

# ---------------- 云端 LLM（OpenAI 兼容接口） ----------------
LLM_BASE_URL = _env('VELO_LLM_BASE_URL', 'http://127.0.0.1:8000/v1')
LLM_API_KEY = _env('VELO_LLM_API_KEY', 'EMPTY')
LLM_MODEL = _env('VELO_LLM_MODEL', 'qwen-7b-chat-int8')

# ---------------- 多机部署 ----------------
# 边缘节点编号（1/2/3），决定读取哪个节点的数据和上传端口
NODE_ID = int(_env('VELO_NODE_ID', '1'))
# 边缘端本地 Milvus collection 名称
EDGE_COLLECTION = _env('VELO_EDGE_COLLECTION', 'diffusion_two_input11_0150_6000')
EDGE_SEND_COLLECTION = _env('VELO_EDGE_SEND_COLLECTION', 'diffusion_two_input_nodes11_0150_6000')

# 训练端地址：边缘端通过 TCP 上传经验，端口为 TRAINER_BASE_PORT + NODE_ID
TRAINER_HOST = _env('VELO_TRAINER_HOST', '127.0.0.1')
TRAINER_BIND_HOST = _env('VELO_TRAINER_BIND_HOST', '0.0.0.0')
TRAINER_BASE_PORT = int(_env('VELO_TRAINER_BASE_PORT', '40060'))

# 训练端通过 FTP 把新权重推送到各边缘节点
EDGE_FTP_HOSTS = _env_list('VELO_EDGE_FTP_HOSTS', '127.0.0.1,127.0.0.1,127.0.0.1')
FTP_USER_PREFIX = _env('VELO_FTP_USER_PREFIX', 'ftpnode')   # 用户名为 ftpnode1 / ftpnode2 / ftpnode3
FTP_PASSWORD = _env('VELO_FTP_PASSWORD', '')
FTP_REMOTE_WEIGHT_PATH = _env('VELO_FTP_REMOTE_WEIGHT_PATH', '/var/ftp/upload/model_network.pth')

# 边缘端读取新权重的本地路径（即 FTP 上传目录）
EDGE_WEIGHT_PATH = _env('VELO_EDGE_WEIGHT_PATH', '/var/ftp/upload/model_network.pth')
# 边缘端启动时加载的初始（专家预训练）权重
INIT_WEIGHT_PATH = _path('VELO_INIT_WEIGHT_PATH', 'weights/pretrained.pth')
