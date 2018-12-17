# create test image based on the percentage of the input events

import os
import sys
import numpy as np
import scipy.misc as spm

import random
random.seed(99999)


#root_path = '/home/anguyen/workspace/paper_src/2018.icra.event.source'  # not .source/dataset --> wrong folder
cwd = os.getcwd()
print 'current dir: ', cwd
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'root path: ', root_path  
sys.path.insert(0, root_path)

from dataset_script.events_to_img import convert_event_to_img, create_empty_img, padzero
from dataset_script.data_io_cnn import add_img_id_to_gt, main_convert_percentage
from dataset_script.count_events_with_gt import main_count_event_gt

##split_id = 'img_pose_all' ## random split
split_id = 'img_pose_all_novel_split'  ## novel split

def create_index_event_img_file(in_count_events_gt_file, index_event_img_file):
    fcounter = open(in_count_events_gt_file, 'r')
    findex = open(index_event_img_file, 'w')
    
    all_lines = fcounter.read().split('\n')
    #print 'all line: ', all_lines
    
    ins_counter = 0
    
    for l in all_lines:
        if l.isdigit():  
            #print 'current l: ', l  
            ins_counter += long(l)
            #print 'current index: ', ins_counter
            findex.write(str(ins_counter) + '\n')
    

def write_txt(list_data, txt_file):
    fwriter = open(txt_file, 'w')
    for l in list_data:
        fwriter.write(l)

    fwriter.close()    
    
    
def create_train_test_list(grth_with_img_id_path, train_path, test_path): ## random split
    #grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    list_grth_id = list(open(grth_with_img_id_path, 'r'))
    
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
        
    # save to txt file
    write_txt(list_train, train_path)
    write_txt(list_test, test_path)
    


def create_train_test_list_novel_split(grth_with_img_id_path, train_path, test_path): ## 1st 70% for training, last 30% for testing
    #grth_with_img_id_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/groundtruth_with_img_id.txt'
    list_grth_id = list(open(grth_with_img_id_path, 'r'))
    
    total_length = len(list_grth_id)
    print 'total grth id len: ', total_length
    
    split_index = int(total_length * 0.7)
    
    # slice first 50 as training
    list_train = list_grth_id[0:split_index]
    print 'len list train: ', len(list_train)
    
    list_test = list(set(list_grth_id) - set(list_train))  # the rest as testing
    print 'len list test: ', len(list_test)
    
    # shuffle train and test list - no need to keep sequential order
    random.shuffle(list_train)
    random.shuffle(list_test)
    
    # save to txt file
    write_txt(list_train, train_path)
    write_txt(list_test, test_path)


def create_percentage_image(kp, index_event_img_file, in_count_events_gt_file, in_raw_events_file, out_percentage_img_foler):
    # read index file
    findex = open(index_event_img_file, 'r')
    all_index = findex.read().split('\n')
    #print 'all index: ', all_index
    
    list_event = list(open(in_raw_events_file, 'r'))

    
    # 1st image is empty - no previous events
    img = create_empty_img()
    fname = padzero(0) + '.png'
    spm.imsave(os.path.join(out_percentage_img_foler, fname), img)  # save 1 chanel as greyscale
    
        
    start_index = 0
    for i in range(len(all_index)):
        if all_index[i].isdigit():
            end_index = int(all_index[i])
              
            print '-----------------------'
            print 'i: ', i
            print 'start index: ', start_index
            print 'end index  : ', end_index
            
            total_events = end_index - start_index
            keep_num_events = int(total_events * (float(kp)/100.0))
            print 'total events for image: ', total_events
            print 'keep num events: ', keep_num_events
            
            new_start_index = start_index + (total_events - keep_num_events)
            print 'new start index: ', new_start_index
              
            img = convert_event_to_img(new_start_index, end_index, list_event)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
              
            # save image
            fname = padzero(i+1) + '.png'  
            spm.imsave(os.path.join(out_percentage_img_foler, fname), img)  # save 1 chanel as greyscale
               
           
            # update new start index
            start_index = end_index

        
            # debug    
            #if i == 80: break
         

        
        
def create_folder_structure(scene_id, split_id):
    
    print 'CREATING DATA FOR: ', scene_id    
    # create folder structure
    raw_folder = os.path.join(root_path, 'event_data', 'raw_data')
    processed_folder = os.path.join(root_path, 'event_data', 'processed')
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        
    
    
    #scene_raw_folder = os.path.join(raw_folder, scene_id)             #/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/raw_data/shapes_rotation
    scene_raw_folder = os.path.join(raw_folder, scene_id)    
    if not os.path.exists(scene_raw_folder):
        print 'ERROR: NO RAW DATA FOR: ', scene_id, 'SCENE!'
        
     
    #scene_processed_folder = os.path.join(processed_folder, scene_id) #/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/shapes_rotation
    scene_processed_folder = os.path.join(processed_folder, scene_id, split_id) #/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/shapes_rotation
    #print 'processed folder: ', scene_processed_folder
    if not os.path.exists(scene_processed_folder):
        os.makedirs(scene_processed_folder)

    return scene_raw_folder, scene_processed_folder





def create_files(scene_raw_folder, scene_processed_folder):
    
    in_gt_file = scene_raw_folder + '/groundtruth.txt'
    in_event_file = scene_raw_folder + '/events.txt'
        
    print '-------------- COUNT NUMBER OF EVENTS --------------'
    counter_event_grt_path = scene_processed_folder + '/count_events_gt.txt'
    if not os.path.exists(counter_event_grt_path):
        main_count_event_gt(in_event_file, in_gt_file, counter_event_grt_path)
    else:
        print 'FILE: ', counter_event_grt_path ,' already exists. Not create new! Delete old file if you want to re-run.'
    
    
    print '-------------- ADD IMAGE ID TO GROUNDTRUTH --------------'
    grth_with_img_id_path = scene_processed_folder + '/groundtruth_with_img_id.txt'
    if not os.path.exists(grth_with_img_id_path):
        add_img_id_to_gt(in_gt_file, grth_with_img_id_path)
    else:
        print 'FILE: ', grth_with_img_id_path, ' already exists. Not create new! Delete old file if you want to re-run.'
        
       
    print '-------------- CREATE TRAIN + TEST LIST --------------'
    train_path = scene_processed_folder + '/train.txt'
    test_path = scene_processed_folder + '/test.txt'
    if not os.path.exists(train_path):
        #create_train_test_list(grth_with_img_id_path, train_path, test_path)
        create_train_test_list_novel_split(grth_with_img_id_path, train_path, test_path)
    else:
        print 'FILE: ', train_path, ' already exists. Not create new! Delete old file if you want to re-run.'
    
    print '-------------- CREATE INDEX EVENT FILE --------------'
    index_event_img_file = scene_processed_folder + '/index_event_img.txt'
    if not os.path.exists(index_event_img_file):
        create_index_event_img_file(counter_event_grt_path, index_event_img_file)
    else:        
        print 'FILE: ', index_event_img_file, ' already exists. Not create new! Delete old file if you want to re-run.'


def create_images_from_events(list_percentage, scene_raw_folder, scene_processed_folder):
    
    main_percentage_folder = os.path.join(scene_processed_folder, 'percentage_img')
    if not os.path.exists(main_percentage_folder):
        os.makedirs(main_percentage_folder)
  
    
    in_count_events_gt_file = scene_processed_folder + '/count_events_gt.txt'
    in_raw_events_file = scene_raw_folder + '/events.txt'

    # create index file    
    #index_event_img_file = '/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/boxes_translation/index_event_img.txt'
    index_event_img_file = scene_processed_folder + '/index_event_img.txt'
    
    
    for kp in list_percentage:
        out_percentage_img_foler = os.path.join(main_percentage_folder, str(kp)) 
        if not os.path.exists(out_percentage_img_foler):
            os.makedirs(out_percentage_img_foler)
            # only create if not exists
            create_percentage_image(kp, index_event_img_file, in_count_events_gt_file, in_raw_events_file, out_percentage_img_foler)
        else:
            print 'FOLDER: ', out_percentage_img_foler, ' already exists. SKIP!'
    
    
def convert_images_to_pkl(list_percentage, scene_processed_folder):
    

    for keep_id in list_percentage:
        #image_event_folder = '/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/shapes_rotation/percentage_img/10'
        image_event_folder = os.path.join(scene_processed_folder, 'percentage_img', str(keep_id))
        
        out_folder = os.path.join(scene_processed_folder, 'percentage_pkl', str(keep_id))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            # only create if not exists
            main_convert_percentage(scene_processed_folder, image_event_folder, out_folder, keep_id)
        else:
            print 'FOLDER: ', out_folder, ' already exists. SKIP!'
            
            
def main():
    '''
    0. Create folder structure
    1. Create files: count_events_gt.txt, groundtruth_with_img_id.txt, index_event_img.txt, train.txt, test.txt
    2. Create images from events
    3. Convert list of images to 1 single .pkl file
    '''
    
    
    list_scene = ['shapes_6dof']
    #list_scene = ['poster_translation']
    #list_scene = ['poster_6dof']
    
    for ls in list_scene:
            
        #scene_id = 'boxes_translation'
        scene_id = ls
        
        
        #list_percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        list_percentage = [100]
        
        # 0. Create folder structure
        scene_raw_folder, scene_processed_folder = create_folder_structure(scene_id, split_id)
    
        # 1. Create files
        create_files(scene_raw_folder, scene_processed_folder)
    
#         # 2. Create images from events
        create_images_from_events(list_percentage, scene_raw_folder, scene_processed_folder)
#         
#         # 3. Convert list of images to 1 single .pkl file
        convert_images_to_pkl(list_percentage, scene_processed_folder)
    
    
    
if __name__ == '__main__':
    
    main()
    
    print 'ALL DONE!'