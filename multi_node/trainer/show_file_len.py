import os

def show_file_len(update_times):
    times = update_times*300
    path1 = './node1/'
    path2 = './node2/'
    path3 = './node3/'

    for root, dirs, files in os.walk(path1):
        # print(files)
        length1 = len(files)
        number1 = 0
        for tar_file in files:
            if str(times) in tar_file:
                number1 = number1 + 1
        # print(number1)
        # print(files)

    for root, dirs, files in os.walk(path2):
        length2 = len(files)
        number2 = 0
        for tar_file in files:
            if str(times) in tar_file:
                number2 = number2 + 1
        # print(number2)


    for root, dirs, files in os.walk(path3):
        length3 = len(files)
        number3 = 0
        for tar_file in files:
            if str(times) in tar_file:
                number3 = number3 + 1
        # print(number3)

    return number1, number2, number3

# if __name__ == '__main__':
#     x = show_file_len(1)
#     print(x)