import os
import shutil
def mvfile():
    source_dir = './send_file_list'
    destination_dir = './log'
    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        shutil.move(source_file, destination_file)