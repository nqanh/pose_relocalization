# transform list of events to images

import os
import numpy as np
import scipy.misc as spm

import cv2

# set print all
#np.set_printoptions(threshold=np.nan)



def padzero(x):
    if x>=0 and x<10:
        return '0000000' + str(x)
    elif x>=10 and x<100:
        return '000000' + str(x)
    elif x>=100 and x<1000:
        return '00000' + str(x)
    elif x>=1000 and x<10000:
        return '0000' + str(x)
    elif x>=10000 and x<100000:
        return '000' + str(x)
    elif x>=100000 and x<1000000:
        return '00' + str(x)
    elif x>=1000000 and x<10000000:
        return '0' + str(x)
    else:
        return '___error___'
    
def create_empty_img():
    img = np.full((180, 240), 0.5)
    img[0][0] = 1.0 # enable 1 pixel just fix scipy imsave problem
    img[1][1] = 0.0
    
    return img

def convert_event_to_img(start_index, end_index, list_event):
    #print '-- current start_index: ', start_index
    #print '-- current end_index: ', end_index
    
    #if (start_index == end_index or start_index == end_index + 1):
    if (start_index == end_index or end_index == start_index + 1):
        print 'Warning: found start index = end_index (+1). No events? - return empty image'
        img = create_empty_img()
        return img  
    
    
    #img = np.zeros((240, 180))
    img = np.full((180, 240), 0.5)  # create one channel - (row = y, column = x)
    #print 'img: ', img
    #print 'img shape: ', img.shape
    #img = np.dstack((img, img, img)) ## stack to three channels here????
    #print 'img shape: ', img.shape
    
    for i in range(start_index, end_index):
        #print 'i=', i
        ev_line = list_event[i].rstrip('\n')
        #print 'event line: ', ev_line
        ev_arr = ev_line.split(' ')
        ex = int(ev_arr[1])
        ey = int(ev_arr[2])
        ep = float(ev_arr[3])
        
        img[ey][ex] = ep  # row=y, col=x
        

    #print 'img: ', img
    
    # save to folder 
    
    
    return img


def main_event_to_image(in_event_path, in_counter_path, out_event_img_folder):
#     in_event_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/events.txt'
#     in_counter_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/count_events_gt.txt'
#     out_event_img_folder = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/event_img'
    
    
    list_event = list(open(in_event_path, 'r'))
    list_counter = list(open(in_counter_path, 'r'))
    
    print 'num events: ', len(list_event)
    print 'num coutner: ', len(list_counter)
    
    # 1st image is empty - no previous events
    img = create_empty_img()
    fname = padzero(0) + '.png'
    spm.imsave(os.path.join(out_event_img_folder, fname), img)  # save 1 chanel as greyscale
        
    # create an image from previous events
    total_event = 0
    start_index = 0
    for i in range(len(list_counter)-1):
        end_index = total_event + int(list_counter[i])
          
        print '-----------------------'
        print 'i: ', i
        print 'start index: ', start_index
        print 'end index  : ', end_index
          
        img = convert_event_to_img(start_index, end_index, list_event)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
          
        # save image
        fname = padzero(i+1) + '.png'  
        spm.imsave(os.path.join(out_event_img_folder, fname), img)  # save 1 chanel as greyscale
          
      
        # update new start index
        start_index = end_index
        total_event += int(list_counter[i]) 
           
    #     if i == 61:
    #         break


if __name__ == "__main__":
        
    #main_event_to_image()        
    
#     # 1st image is empty - no previous events
#     img = create_empty_img()
#     fname = padzero(0) + '.png'
#     spm.imsave(os.path.join(out_event_img_folder, fname), img)  # save 1 chanel as greyscale
#         
#     # create an image from previous events
#     total_event = 0
#     start_index = 0
#     for i in range(len(list_counter)-1):
#         end_index = total_event + int(list_counter[i])
#           
#         print '-----------------------'
#         print 'i: ', i
#         print 'start index: ', start_index
#         print 'end index  : ', end_index
#           
#         img = convert_event_to_img(start_index, end_index)
#         #cv2.imshow('img', img)
#         #cv2.waitKey(0)
#           
#         # save image
#         fname = padzero(i+1) + '.png'  
#         spm.imsave(os.path.join(out_event_img_folder, fname), img)  # save 1 chanel as greyscale
#           
#       
#         # update new start index
#         start_index = end_index
#         total_event += int(list_counter[i]) 
#            
#     #     if i == 61:
#     #         break
         
    
    
    print 'ALL DONE!'