# 配置统一放在仓库根目录的 velo_config.py，这里只是转发，便于在本目录直接运行脚本
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from velo_config import *  # noqa: E402,F401,F403

# 多机边缘端的任务分片（第 0 片每节点 10001 条，其余每片 7501 条）
if 'VELO_DATA_DIR' not in os.environ:
    DATA_DIR = os.path.join(ROOT, 'data', 'tasks', 'multi_node')
