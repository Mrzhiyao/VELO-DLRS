import os

def show_file_name(update_times):
    path1 = './node1/'
    path2 = './node2/'
    path3 = './node3/'
    times = update_times * 300
    for root, dirs, files in os.walk(path1):
        file1 = files

    for root, dirs, files in os.walk(path2):
        file2 = files

    for root, dirs, files in os.walk(path3):
        file3 = files
    file1_out = []
    file2_out = []
    file3_out = []
    for i in range(len(file1)):
        if str(times) in file1[i]:
            file1_out.append('./node1/' + file1[i])
    for i in range(len(file2)):
        if str(times) in file2[i]:
            file2_out.append('./node2/' + file2[i])
    for i in range(len(file3)):
        if str(times) in file3[i]:
            file3_out.append('./node3/' + file3[i])

    return file1_out, file2_out, file3_out

# if __name__ == '__main__':
#     x,y,z = show_file_name(1)
#     print(x,y,z)
#     for i in x:
#         print(i)