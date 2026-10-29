import os
import shutil
def mvfile(source_dir, destination_dir, update_times):
    times = str(update_times*300)
    files = os.listdir(source_dir)
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        if times in source_file:
            shutil.move(source_file, destination_file)

# if __name__ == '__main__':
#     source_dir = './node1'
#     destination_dir = './node1_log'
#     update_times = 2
#     mvfile(source_dir, destination_dir, update_times)