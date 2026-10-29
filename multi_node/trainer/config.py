# 配置统一放在仓库根目录的 velo_config.py，这里只是转发，便于在本目录直接运行脚本
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from velo_config import *  # noqa: E402,F401,F403

# 单机版与多机版各自有一套专家演示数据
if 'VELO_EXPERT_DIR' not in os.environ:
    EXPERT_DIR = os.path.join(ROOT, 'data', 'expert', 'multi_node')
