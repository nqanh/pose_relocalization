import os
import pickle
import numpy as np
import random
import cv2

from keras.preprocessing import image
from dataset_script.events_to_img import padzero

random.seed(99999)

def load_data(main_path, pkl_file):
    try:
        print 'Loading pickle file ....', os.path.join(main_path, pkl_file)
        X, Y = pickle.load(open(os.path.join(main_path, pkl_file), 'rb'))
        print '........................ done!'
    except Exception:
        print 'ERROR: File ', pkl_file, 'does not exist!'
    
    return X, Y


def write_data(list_data, img_folder, out_folder, pkl_file):
    X, Y = [], []
    
    for l in list_data:
        l = l.rstrip('\n')
        gt_arr = l.split(' ')
    
        # Y
        px = float(gt_arr[1])
        py = float(gt_arr[2])
        pz = float(gt_arr[3])
        qx = float(gt_arr[4])
        qy = float(gt_arr[5])
        qz = float(gt_arr[6])
        qw = float(gt_arr[7])    
        current_y = [px, py, pz, qx, qy, qz, qw]
        Y.append(current_y)
        
        # print debug
        
        # X
        img_id = gt_arr[8]
        img_path = os.path.join(img_folder, img_id)
        X.append(img_path)
        
        print '--------------------------------'
        print 'current l: ', l
        print 'current y: ', current_y
        print 'img id: ', img_id
        
    
    # process Image
    for ind, val in enumerate(X):
        
        img = image.load_img(val, target_size=(224, 224)) ## change all pixel values
        X[ind] = image.img_to_array(img)
        
        # TODO: zero mean?
        X[ind] /= 255.    # fix NAN loss problem with X[ind] /= 255.
        
#         print 'X[ind] shape: ', X[ind].shape
#         cv2.imshow('img', X[ind])
#         cv2.waitKey(0)
        
        print 'Process input images: ', ind, '/', len(list_data), ' -- img path: ', val
        
        #break
        
    # convert to numpy array   ## memory leak???
    #X = np.array(X)
    #Y = np.array(Y)

    #save to pickle file for next time
    print 'Saving pickle file ... ', os.path.join(out_folder, pkl_file), '... done!'
    pickle.dump((X, Y), open(os.path.join(out_folder, pkl_file), 'wb'))
        
            
        
def save_txt(list_data, out_folder, txt_file):
    file_path = os.path.join(out_folder, txt_file)
    fwriter = open(file_path, 'w')
    for l in list_data:
        fwriter.write(l)

    fwriter.close()
    
def add_img_id_to_gt(in_grth_path, grth_with_img_id_path):   
    #in_grth_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/groundtruth.txt'
    
    list_grth = list(open(in_grth_path, 'r'))
    print 'len grth: ', len(list_grth)
    
    #grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    fwriter = open(grth_with_img_id_path, 'w')
    
    for i in range(len(list_grth)):
        gt_line = list_grth[i].rstrip('\n')
        gt_line = gt_line + ' ' + padzero(i) + '.png' + '\n'
        print 'gt line: ', gt_line
        fwriter.write(gt_line)
        #break
    #fwriter.close() 
    
def main_create_train_test(grth_with_img_id_path, img_folder, out_folder, is_small_500_set):
    
    #grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    list_grth_id = list(open(grth_with_img_id_path, 'r'))
    
    ## for testing 
    if is_small_500_set:
        #list_grth_id = list_grth_id[0:1000]
        #list_grth_id = list_grth_id[0:100]
        print 'GET ONLY 500 FIRST IMAGE!'
        list_grth_id = list_grth_id[0:500]
    
    # shuffle list
    random.shuffle(list_grth_id)
    
    num_train = int(0.7 * len(list_grth_id))
    num_test = len(list_grth_id) - num_train
    print 'total train sample: ', num_train
    print 'total test sample: ', num_test
    print list_grth_id[0]
    
    # slice first num_train sample as training
    list_train = list_grth_id[0:num_train]
    print 'len list train: ', len(list_train)
    
    list_test = list(set(list_grth_id) - set(list_train))
    print 'len list test: ', len(list_test)
    
    
    #img_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/event_img'
    #out_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation'
    
    train_pkl_file = 'train.pkl'
    test_pkl_file = 'test.pkl'


    #train_pkl_file = 'train_cnn_all.pkl'
    #test_pkl_file = 'test_cnn_all.pkl'

    #train_pkl_file = 'train_cnn_small.pkl'
    #test_pkl_file = 'test_cnn_small.pkl'
        
    #train_pkl_file = 'train_cnn_small_100.pkl'
    #test_pkl_file = 'test_cnn_small_100.pkl'
    
    #train_pkl_file = 'train_cnn_small_500.pkl'
    #test_pkl_file = 'test_cnn_small_500.pkl'
    
    # save to txt file
    train_txt_file = train_pkl_file.replace('.pkl', '.txt')
    test_txt_file = test_pkl_file.replace('.pkl', '.txt')
    save_txt(list_train, out_folder, train_txt_file)
    save_txt(list_test, out_folder, test_txt_file)
    
    # save to pkl file
    write_data(list_train, img_folder, out_folder, train_pkl_file)
    write_data(list_test, img_folder, out_folder, test_pkl_file)     




def main_create_percentage_test(grth_with_img_id_path, img_folder, out_folder):
    '''
    create test.pkl for sub percentage image folder
    '''
    
    #grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    list_grth_id = list(open(grth_with_img_id_path, 'r'))
    
#     ## for testing 
#     if is_small_500_set:
#         #list_grth_id = list_grth_id[0:1000]
#         #list_grth_id = list_grth_id[0:100]
#         print 'GET ONLY 500 FIRST IMAGE!'
#         list_grth_id = list_grth_id[0:500]
    
    # shuffle list
    random.shuffle(list_grth_id)
    
    num_train = int(0.7 * len(list_grth_id))
    num_test = len(list_grth_id) - num_train
    print 'total train sample: ', num_train
    print 'total test sample: ', num_test
    print list_grth_id[0]
    
    # slice first num_train sample as training
    list_train = list_grth_id[0:num_train]
    print 'len list train: ', len(list_train)
    
    list_test = list(set(list_grth_id) - set(list_train))
    print 'len list test: ', len(list_test)
    
    
    #img_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/event_img'
    #out_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation'
    
    train_pkl_file = 'train.pkl'
    test_pkl_file = 'test.pkl'


    #train_pkl_file = 'train_cnn_all.pkl'
    #test_pkl_file = 'test_cnn_all.pkl'

    #train_pkl_file = 'train_cnn_small.pkl'
    #test_pkl_file = 'test_cnn_small.pkl'
        
    #train_pkl_file = 'train_cnn_small_100.pkl'
    #test_pkl_file = 'test_cnn_small_100.pkl'
    
    #train_pkl_file = 'train_cnn_small_500.pkl'
    #test_pkl_file = 'test_cnn_small_500.pkl'
    
    # save to txt file
    train_txt_file = train_pkl_file.replace('.pkl', '.txt')
    test_txt_file = test_pkl_file.replace('.pkl', '.txt')
    save_txt(list_train, out_folder, train_txt_file)
    save_txt(list_test, out_folder, test_txt_file)
    
    # save to pkl file
    #write_data(list_train, img_folder, out_folder, train_pkl_file)  ## dont need to create train.pkl
    write_data(list_test, img_folder, out_folder, test_pkl_file)     



def main_convert_percentage(scene_processed_folder, img_folder, out_folder, keep_id):
    '''
    create test.pkl for sub percentage image folder
    create train.pkl when keep_id is 100  --> use all events
    '''
    
    train_txt_path = scene_processed_folder + '/train.txt'
    test_txt_path = scene_processed_folder + '/test.txt'
    
    list_train = list(open(train_txt_path, 'r'))
    list_test = list(open(test_txt_path, 'r'))
    
    # save to pkl file
    train_pkl_file = 'train.pkl'
    test_pkl_file = 'test.pkl'
    if keep_id == 100:
        write_data(list_train, img_folder, out_folder, train_pkl_file)  
           
    write_data(list_test, img_folder, out_folder, test_pkl_file)     



if __name__ == '__main__':
    
    grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    list_grth_id = list(open(grth_with_img_id_path, 'r'))
    
    ## for testing 
    #list_grth_id = list_grth_id[0:1000]
    #list_grth_id = list_grth_id[0:100]
    #list_grth_id = list_grth_id[0:500]
    
    # shuffle list
    random.shuffle(list_grth_id)
    
    num_train = int(0.7 * len(list_grth_id))
    num_test = len(list_grth_id) - num_train
    print 'total train sample: ', num_train
    print 'total test sample: ', num_test
    print list_grth_id[0]
    
    # slice first num_train sample as training
    list_train = list_grth_id[0:num_train]
    print 'len list train: ', len(list_train)
    
    list_test = list(set(list_grth_id) - set(list_train))
    print 'len list test: ', len(list_test)
    
    
    img_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/event_img'
    out_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation'
    
    train_pkl_file = 'train_cnn_all.pkl'
    test_pkl_file = 'test_cnn_all.pkl'

    #train_pkl_file = 'train_cnn_small.pkl'
    #test_pkl_file = 'test_cnn_small.pkl'
        
    #train_pkl_file = 'train_cnn_small_100.pkl'
    #test_pkl_file = 'test_cnn_small_100.pkl'
    
    #train_pkl_file = 'train_cnn_small_500.pkl'
    #test_pkl_file = 'test_cnn_small_500.pkl'
    
    # save to txt file
    train_txt_file = train_pkl_file.replace('.pkl', '.txt')
    test_txt_file = test_pkl_file.replace('.pkl', '.txt')
    save_txt(list_train, out_folder, train_txt_file)
    save_txt(list_test, out_folder, test_txt_file)
    
    # save to pkl file
    write_data(list_train, img_folder, out_folder, train_pkl_file)
    write_data(list_test, img_folder, out_folder, test_pkl_file)
    


    print 'ALL DONE!'

    