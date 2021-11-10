import os
import random

cwd = '/home/rhosane/Reconhecimento_de_imagens/1.Geracao_de_dados/Data_3/imgs_rot'

train_test_filepath = '/home/rhosane/Reconhecimento_de_imagens/1.Geracao_de_dados/Data_3'
                   

def recollect_data():
    file_list = []
    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.endswith(".png"):
                file_path_txt = os.path.join(root, file)
                image_file_path = os.path.splitext(file_path_txt)[0]
                file_list.append(file)

    return file_list

def create_train_test_lists(file_list):
    total_list = file_list
    total_lenght = len(file_list)
    train_lenght = int(0.95*total_lenght)
    test_lenght = int(0.05*total_lenght)

    testlist = random.sample(total_list, test_lenght)

    for i, element in enumerate(testlist):
        el_index = total_list.index(element)
        total_list.pop(el_index)

    trainlist = total_list

    return trainlist, testlist

def create_txt_paths(set, train_lenght):
    traintxtfilename = train_test_filepath+'/train.txt'
    testtxtfilename = train_test_filepath+'/test.txt'

    if len(set) == train_lenght:
        txt_file = open(traintxtfilename, 'w+')
    else:
        txt_file = open(testtxtfilename, 'w+')

    path = '/content/darknet/data/obj/'

    for element in set:
        txt_file.write(path+element+'\n')


if __name__ == '__main__':
    file_list = recollect_data()
    print('Lenght of initial list:', len(file_list))
    train_list, test_list = create_train_test_lists(file_list)
    print('Lenght of train list:', len(train_list))
    print('Lenght of test list:', len(test_list))
    print('Train + test:', len(train_list)+len(test_list))
    create_txt_paths(train_list, len(train_list))
    create_txt_paths(test_list, len(train_list))
