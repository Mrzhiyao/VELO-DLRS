import os

def show_file_len():
    path = './send_file_list/'

    for root, dirs, files in os.walk(path):
        length = len(files)
    print(length)

    return length

# if __name__ == '__main__':
#     show_file_len()