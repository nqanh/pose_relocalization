import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
#from dataset_script.CREATE_DATA import split_id


#force matplotlib use any Xwindows
matplotlib.use('Agg')

import sys
cwd = os.getcwd()
print 'current dir: ', cwd
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'root path: ', root_path  
sys.path.insert(0, root_path)

import tensorflow as tf

from dataset_script.data_io_cnn import load_data
from main_keras.vgg16 import VGG_LSTM2


##split_id = 'img_pose_all' ## random split
split_id = 'img_pose_all_novel_split'  ## novel split


def train_net(model, train_image, train_pose, log_model_path, log_plot_path):
    
    # loss function
    model.compile(loss="mean_squared_error", optimizer="sgd",  lr=0.00001, decay=1e-6, momentum=0.9)  # OK WITH VGG-LSTM2
    
    # REDUCE learning rate - NOT COVERAGE FASTER
    #model.compile(loss="mean_squared_error", optimizer="sgd",  lr=0.000001, decay=1e-6, momentum=0.9)  # OK WITH VGG-LSTM2
     
    # use for loop to fit
    num_epoch = 20000
    loss_history = []
    val_history = []
    for e in range(num_epoch):
        print 'EPOCH e =', e, '/', num_epoch 
        train_history = model.fit(train_image, train_pose, batch_size=20, epochs=1, validation_split=0.05, verbose=1)  #verbose control output print)
        #print '\n'
         
        if e > 0: # ignore first  epochs
            loss_history.append(train_history.history['loss'])
            val_history.append(train_history.history['val_loss'])
         
        #if e > 5 and e % 500 == 0: # save model every 500 epochs
        if e > 0 and e % 50 == 0: # save model every 100 epochs
            model_file_name = 'full_model_epoch_e' + str(e) + '.hdf5'
            out_model_path = os.path.join(log_model_path, model_file_name) 
            model.save(out_model_path)
            print 'saved model at epoch e =', e, 'ok!'
              
        if e > 0 and e % 10 == 0: # save plot loss
            # list all data in history
            print(train_history.history.keys())
            # summarize history for loss
            plt.plot(loss_history, color='r')
            plt.plot(val_history, color='g')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
             
            plot_file_name = 'plot_epoch_e' + str(e) + '.png'
            out_plot_path = os.path.join(log_plot_path, plot_file_name)
            plt.savefig(out_plot_path)
            #plt.show()
             
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Just another deepsh*t!')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use - default is 0',
                        default='0', type=str)
    parser.add_argument('--scene', dest='scene_id',
                        help='Scene ID to train - default is `shapes_rotation` ',
                        default='shapes_rotation', type=str)
    parser.add_argument('--weight', dest='weight_path',
                        help='Pretrained weight - default is `None` ',
                        default=None, type=str)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    # parse arg
    args = parse_args()
    
    # config gpu
    gpu_id = args.gpu_id
    gpu_id_string = '/gpu:' + gpu_id
    #with tf.device('/gpu:0'):
    #with tf.device('/cpu:0'):
    with tf.device(gpu_id_string):
        print 'Using gpu: ', gpu_id_string
            
        pretrained_weight_path = args.weight_path
        if pretrained_weight_path != None:
            print 'USING PRETRAINED WEIGHT: ', pretrained_weight_path
        else:
            print 'TRAIN FROM SCRATCH'
        

        model = VGG_LSTM2(pretrained_weight_path)
        model.summary()
        
        main_output = os.path.join(root_path, 'output')
        
        # scene/sequence id
        #scene_id = 'shapes_rotation'
        scene_id = args.scene_id  
        
        net_id = 'vgg_lstm2'
        
        
        out_net_id_path = os.path.join(main_output, scene_id, net_id) 
        print 'out net id path: ', out_net_id_path
        
        # log training model, plot, etc.
        log_model = 'log_model'
        log_plot  = 'log_plot'
        #predition = 'prediction'  # not use
        
        log_model_path = os.path.join(main_output, scene_id, net_id, split_id, log_model)
        log_plot_path  = os.path.join(main_output, scene_id, net_id, split_id, log_plot)
        #predition_path = os.path.join(main_output, scene_id, net_id, split_id, predition)         
        if not os.path.exists(log_model_path):
            os.makedirs(log_model_path)
        if not os.path.exists(log_plot_path):
            os.makedirs(log_plot_path)
        #if not os.path.exists(predition_path):
        #    os.makedirs(predition_path)    
        
        # load data
        dataset_folder = os.path.join(root_path, 'event_data', 'processed' )
        #data_path = os.path.join(dataset_folder, scene_id, split_id)
        data_path = os.path.join(dataset_folder, scene_id, split_id, 'percentage_pkl', '100')  ## -->> always test with all events i.e. 100%
        #print 'data path: ', data_path
        train_image, train_pose = load_data(data_path, 'train.pkl')
        print 'convert to numpy ...'
        train_image = np.array(train_image)
        train_pose = np.array(train_pose)
        print 'train_image shape: ', train_image.shape
        print 'train_pose  shape: ', train_pose.shape
    
        # train
        train_net(model, train_image, train_pose, log_model_path, log_plot_path)
    
    print 'ALL DONE!'
