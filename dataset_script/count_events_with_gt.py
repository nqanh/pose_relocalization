# count number of events occur until they have GT

import os
import numpy as np
from distlib.util import in_venv


def main_count_event_gt(in_event_path, in_grt_path, out_event_grt_path):
#     in_event_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/events.txt'
#     in_grt_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/groundtruth.txt'
#     out_event_grt_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/count_events_gt.txt'
    
    # read file
    list_event = list(open(in_event_path, 'r'))
    list_grt = list(open(in_grt_path, 'r'))
    print 'total #events: ', len(list_event)
    print 'total #grt   : ', len(list_grt)
    
    
    fwriter = open(out_event_grt_path, 'w')
             
    # start_index = 0
    # for ev in list_event:
    #     ev = ev.replace('\n', '')
    #      
    #     ev_line = ev.split(' ')
    #     ev_time_stamp = float(ev_line[0])
    #      
    #      
    #      
    #     print 'event time: ', ev_time_stamp
    #      
    #     #t1 = 0.0
    #     # find biggest time stamp in grt
    #     for i in range(start_index, len(list_grt)):
    #         gt_arr = list_grt[i].split(' ')
    #         gt_time = float(gt_arr[0])
    #          
    #         if ev_time_stamp < gt_time:
    #             print 'found: ', list_grt[i]
    #             out_line = ev + ' ' + gt_arr[1] + ' ' + gt_arr[2] + ' ' + gt_arr[3] + ' ' + gt_arr[4] + ' ' + gt_arr[5] + ' ' + gt_arr[6] + ' ' + gt_arr[7] # + '\n'
    #             fwriter.write(out_line)
    #              
    #             start_index = i  # set index for next loop - ignore the pass
    #              
    #             break # important!!!
        
        
        
    start_index = 0  ## index for looping all events - jump over for faster
    #start_index = 112355460 # DEBUG
         
    list_counter = np.zeros((len(list_grt), ), dtype=np.int) ## count number of events in each gt
       
    
    
    total_events = len(list_event)
    # get last ev_time index
    #last_ev_time = list_event[total_events-1].split(' ')[0]
    #print 'last ev_time: ', last_ev_time
    
    #start_gt = 11947  # DEBUG
    
    #for ind, gt in enumerate(list_grt):
    #for gt_ind in range(start_gt, len(list_grt)):
    for gt_ind in range(len(list_grt)):
#         gt_arr = gt.split(' ')
#         gt_time = float(gt_arr[0])
        gt_arr = list_grt[gt_ind].split(' ')
        gt_time = float(gt_arr[0])
        
        print '-----------------------------'
#         print 'start index: ', start_index
#         print 'list_vent[start_index]', list_event[start_index]
        
        for i in range(start_index, total_events):
#             if start_index >= len(list_event):  # in "boxes_translation", some last grt don't have any events
#                 print 'FOUND start_index >= len(list_event ', '-- start_index=', start_index, '-- len(list_event)=', len(list_event)
#                 list_counter[ind] = 0
#                 break
            
            ev_arr = list_event[i].split(' ')
            ev_time = float(ev_arr[0])
            #print 'ev_time: ', ev_time
            
#             if ev_time == last_ev_time:   # reach to the last event  --> set start index to the final list
#                 start_index = total_events
#                 break
            if ev_time < gt_time:
                #list_counter[ind] += 1 ## found 1 events with time < gt
                list_counter[gt_ind] += 1 ## found 1 events with time < gt
                if start_index + list_counter[gt_ind] == total_events:
                    print 'REACH FINAL EVENT LINE'
                    start_index = total_events
                    break
                
            else:
                start_index = i  ## for next loop
                break
            
        
        #print 'ind: ', ind
        print 'ind: ', gt_ind
        #print 'gt : ', gt
        print 'gt: ', list_grt[gt_ind]
        #print 'list_coutner[ind]: ', list_counter[ind]
        print 'list_counter[ind]: ', list_counter[gt_ind]
        print '-----------------------------'
        
        #break # for testing
    
    print 'list counter: ', list_counter
    print 'max counter : ', max(list_counter)
    
    for ct in list_counter:
        fwriter.write(str(ct) + '\n')

    #fwriter.close()
    
if __name__ == "__main__":
    
    #main_count_event_gt()
    in_event_file = '/home/anguyen/workspace/dataset/Event/raw_data/boxes_translation/events.txt'
    list_events = list(open(in_event_file, 'r'))
    
    print 'total events: ', len(list_events)
    
    
    
    print 'ALL DONE!'