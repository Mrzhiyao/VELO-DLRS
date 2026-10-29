import os
import shutil
def copy_weight_file(source_dir, destination_dir, update_time):
    source_file = source_dir
    # source_file = os.path.join(source_dir, file)
    file = source_file.split('/')[-1]
    file = str(update_time) + file
    destination_file = os.path.join(destination_dir, file)
    shutil.copy(source_file, destination_file)

# if __name__ == '__main__':
#     copy_weight_file("./ceshi/model_network.pth", "./weightsave_log",1)