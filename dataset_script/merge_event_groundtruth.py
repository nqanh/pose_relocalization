import os

import sys


in_event_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/events.txt'
in_grt_path = '/home/anguyen/workspace/dataset/Event/raw_data/shapes_rotation/groundtruth.txt'

out_event_grt_path = '/home/anguyen/workspace/dataset/Event/processed/shapes_rotation/event_grt.txt'

# read file
list_event = list(open(in_event_path, 'r'))
list_grt = list(open(in_grt_path, 'r'))
print 'total #events: ', len(list_event)
print 'total #grt   : ', len(list_grt)

list_egt = list(open(out_event_grt_path, 'r'))
print 'total #egt   : ', len(list_egt)


# fwriter = open(out_event_grt_path, 'w')
#         
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
            
        


print 'ALL DONE!'
        
        
    