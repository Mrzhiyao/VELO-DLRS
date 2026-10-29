"""Download the sample task dataset and expert demonstrations from Google Drive.

Usage:  python download_data.py
"""
import os
import zipfile

import gdown

ROOT = os.path.dirname(os.path.abspath(__file__))

# Google Drive file ID of velo_data.zip (the part after /d/ in the share link)
FILE_ID = '1KLYnrl_WvWkDe53fpSAdqZXyr3tXW6nr'

if __name__ == '__main__':
    out = os.path.join(ROOT, 'velo_data.zip')
    gdown.download(id=FILE_ID, output=out, quiet=False)
    # zip 内已包含 data/tasks/ 和 data/expert/{single_node,multi_node}/
    with zipfile.ZipFile(out) as z:
        z.extractall(ROOT)
    os.remove(out)
    print('data extracted to', os.path.join(ROOT, 'data'))
