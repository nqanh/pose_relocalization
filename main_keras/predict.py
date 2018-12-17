import os
import numpy as np
import time
import math
from keras.models import load_model



import sys

cwd = os.getcwd()
#print 'current dir: ', cwd
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
#print 'root path: ', root_path  
sys.path.insert(0, root_path)

from dataset_script.data_io_cnn import load_data


#split_id = 'img_pose_500'
#split_id = 'img_pose_all'
split_id = 'img_pose_all_novel_split'

def predict_pose(in_scene_id, in_net_id, in_model_file_name):
    '''
    only predict, not evaluate results (run evaluate.py script to evaluate)
    '''
    
    main_output = os.path.join(root_path, 'output')
    #scene_id = 'shapes_rotation'
    #scene_id = 'boxes_translation'
    scene_id = in_scene_id
    
    net_id = in_net_id
    
    model_file_name = in_model_file_name
    
    log_model = 'log_model'
    predition = 'prediction'
    
    # save prediction result to file
    predition_path = os.path.join(main_output, scene_id, net_id, split_id, predition)   
    prediction_file = os.path.join(predition_path, 'prediction.pre')
    
    # load test data
    #dataset_folder = '/home/anguyen/workspace/dataset/Event/processed/'
    dataset_folder = os.path.join(root_path, 'event_data', 'processed')
    data_path = os.path.join(dataset_folder, scene_id, split_id)    
    testX, testY = load_data(data_path, 'test.pkl')
    # convert to numpy array
    testX = np.array(testX)
    testY = np.array(testY)
    
    # load trained model
    log_model_path = os.path.join(main_output, scene_id, net_id, split_id, log_model)
    model_file = os.path.join(log_model_path, model_file_name)
    
    trained_model = load_model(model_file)
    
    predicted_result = trained_model.predict(testX)
    
    print 'writing predicted result to file ...'
    fwriter = open(prediction_file, 'w')
    for p in predicted_result:
        out_line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3]) + ' ' + str(p[4]) + ' ' + str(p[5]) + ' ' + str(p[6]) + '\n' 
        print out_line
        fwriter.write(out_line)
        

def predict_and_evaluate(in_scene_id, in_net_id, in_model_file_name):
    '''
    do prediction and evaluation for main results (use all events)
    '''
    
    main_output = os.path.join(root_path, 'output')
    #scene_id = 'shapes_rotation'
    #scene_id = 'boxes_translation'
    scene_id = in_scene_id
    
    
    #net_id = 'vgg_lstm2'
    #net_id = 'vgg'
    #net_id = 'vgg_conv_lstm2'
    #net_id = 'vgg_lsltm2_no_dense'
    net_id = in_net_id
    
    #split_id = 'img_pose_500'
    #split_id = 'img_pose_all'
    
    #model_file_name = 'full_model_epoch_e600.hdf5'
    #model_file_name = 'full_model_epoch_e1000.hdf5'
    #model_file_name = 'full_model_epoch_e1200.hdf5'
    model_file_name = in_model_file_name
    
    log_model = 'log_model'
    predition = 'prediction'
    
    # save prediction result to file
    predition_path = os.path.join(main_output, scene_id, net_id, split_id, predition)   
    prediction_file = os.path.join(predition_path, 'prediction.pre')
    
    # load test data
    #dataset_folder = '/home/anguyen/workspace/dataset/Event/processed/'
    dataset_folder = os.path.join(root_path, 'event_data', 'processed')
    data_path = os.path.join(dataset_folder, scene_id, split_id)    
    testX, testY = load_data(data_path, 'test.pkl')
    # convert to numpy array
    testX = np.array(testX)
    testY = np.array(testY)
    
    # load trained model
    log_model_path = os.path.join(main_output, scene_id, net_id, split_id, log_model)
    model_file = os.path.join(log_model_path, model_file_name)
    
    trained_model = load_model(model_file)
    
    start_time = time.time()
    
    # run prediction
    predicted_result = trained_model.predict(testX)
    
    elapsed_time = time.time() - start_time
    print 'Predict: ', len(testX), ' event images in: ', elapsed_time, ' seconds --> ', (elapsed_time*1000)/len(testX), 'miliseconds per image'
    
    print 'Writing results to file ...'
    fwriter = open(prediction_file, 'w')
    for p in predicted_result:
        out_line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3]) + ' ' + str(p[4]) + ' ' + str(p[5]) + ' ' + str(p[6]) + '\n' 
        #print out_line
        fwriter.write(out_line)

    
    # evaluate result
    # initilize 'results' to store all results
    results = np.zeros((len(predicted_result),2))
    i = 0
    
    # statistic
    counter_good_pred = 0  # count #image has < 0.05m AND 2deg accuracy
    
    #for line_grth, line_pred in zip(grth_content, pred_content):
    for line_pred, line_grth in zip(predicted_result, testY):
        # parse content to use PoseNet code: https://github.com/alexgkendall/caffe-posenet/blob/master/posenet/scripts/test_posenet.py
        pose_x = [line_grth[0], line_grth[1], line_grth[2]]
        pose_q = [line_grth[3], line_grth[4], line_grth[5], line_grth[6]]
        
        predicted_x = [line_pred[0], line_pred[1], line_pred[2]]
        predicted_q = [line_pred[3], line_pred[4], line_pred[5], line_pred[6]]
        
#         print '-------------------------------------------'
#         print 'i        : ', i
#         print 'pose x   : ', pose_x
#         print 'pose q   : ', pose_q
#         print 'Predicted x : ', predicted_x
#         print 'Predicted q : ', predicted_q
         
        
        
        # convert to numpy
        pose_q = np.array(pose_q)
        pose_x = np.array(pose_x)
        predicted_q = np.array(predicted_q)
        predicted_x = np.array(predicted_x)
    
        #Compute Individual Sample Error
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-predicted_x)
     
        results[i,:] = [error_x,theta]
        
        if error_x < 0.08 and theta < 4.0:
            counter_good_pred+=1
        
        i = i + 1
        #print 'Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta, '\n'
    
    median_result = np.median(results,axis=0)
    median_error = 'Median_error  ' + str(median_result[0]) + ' m  and ' + str(median_result[1]) + ' degrees'
    print median_error
    
    average_result = np.average(results, axis=0)
    average_error = 'Average_error ' + str(average_result[0]) + ' m  and ' + str(average_result[1]) + ' degrees'
    print average_error
    
    acc_statistic = (float(counter_good_pred)/float(len(testY))) * 100
    print 'Total good prediction: ', counter_good_pred
    print 'Accuracy: ', acc_statistic
    
    # save error results to file
    error_all_path = os.path.join(main_output, scene_id, net_id, split_id, predition, 'error_all.txt')
    error_summary_path = os.path.join(main_output, scene_id, net_id, split_id, predition, 'error_summary.txt')

    np.savetxt(error_all_path, results, delimiter=' ', fmt='%02.20f')

    fout = open(error_summary_path, 'w')
    #fout.write(median_error + '\n')
    #fout.write(average_error + '\n')
    fout.write('scene_id median_error_met average_error_met median_error_deg average_error_deg accuracy_statistic\n')
    fout.write(scene_id + ' ' + str(median_result[0]) + ' ' + str(average_result[0]) + ' ' + str(median_result[1]) + ' ' + str(average_result[1]) + ' ' + str(acc_statistic))
    
    


def predict_and_evaluate_percentage_img(in_scene_id, in_net_id, trained_model, current_percentage):
    '''
    do prediction and evaluation for percentage images
    '''
    
    main_output = os.path.join(root_path, 'output')
    #scene_id = 'shapes_rotation'
    #scene_id = 'boxes_translation'
    scene_id = in_scene_id
    
    
    #net_id = 'vgg_lstm2'
    #net_id = 'vgg'
    #net_id = 'vgg_conv_lstm2'
    #net_id = 'vgg_lsltm2_no_dense'
    net_id = in_net_id
    
    
    
    
    #predition = 'prediction'
    
    # out result paths
    percentage_path = os.path.join(main_output, scene_id, net_id, split_id, 'results_all', str(current_percentage))
    # save prediction result to file
    #predition_path = os.path.join(main_output, scene_id, net_id, split_id, predition)   
    #predition_path = os.path.join(percentage_path, predition)
    predition_path = percentage_path
    if not os.path.exists(predition_path):
        os.makedirs(predition_path)
    prediction_file = os.path.join(predition_path, 'prediction.pre')
    
    # load test data
    #dataset_folder = '/home/anguyen/workspace/dataset/Event/processed/'
    dataset_folder = os.path.join(root_path, 'event_data', 'processed')
    data_path = os.path.join(dataset_folder, scene_id, split_id, 'percentage_pkl', str(current_percentage))    
    #data_path = '/home/anguyen/workspace/paper_src/2018.icra.event.source/event_data/processed/shapes_rotation/img_pose_all/percentage_test/10'
    testX, testY = load_data(data_path, 'test.pkl')
    # convert to numpy array
    testX = np.array(testX)
    testY = np.array(testY)
    
#     # load trained model
#     log_model_path = os.path.join(main_output, scene_id, net_id, split_id, log_model)
#     model_file = os.path.join(log_model_path, model_file_name)    
#     trained_model = load_model(model_file)
    
    start_time = time.time()
    
    # run prediction
    predicted_result = trained_model.predict(testX)
    
    elapsed_time = time.time() - start_time
    print 'Predict: ', len(testX), ' event images in: ', elapsed_time, ' seconds --> ', (elapsed_time*1000)/len(testX), 'miliseconds per image'
    
    print 'Writing results to file ...'
    fwriter = open(prediction_file, 'w')
    for p in predicted_result:
        out_line = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + str(p[3]) + ' ' + str(p[4]) + ' ' + str(p[5]) + ' ' + str(p[6]) + '\n' 
        #print out_line
        fwriter.write(out_line)

    
    # evaluate result
    # initilize 'results' to store all results
    results = np.zeros((len(predicted_result),2))
    i = 0
    
    # statistic
    counter_good_pred = 0  # count #image has < 0.05m AND 2deg accuracy
    
    #for line_grth, line_pred in zip(grth_content, pred_content):
    for line_pred, line_grth in zip(predicted_result, testY):
        # parse content to use PoseNet code: https://github.com/alexgkendall/caffe-posenet/blob/master/posenet/scripts/test_posenet.py
        pose_x = [line_grth[0], line_grth[1], line_grth[2]]
        pose_q = [line_grth[3], line_grth[4], line_grth[5], line_grth[6]]
        
        predicted_x = [line_pred[0], line_pred[1], line_pred[2]]
        predicted_q = [line_pred[3], line_pred[4], line_pred[5], line_pred[6]]
        
#         print '-------------------------------------------'
#         print 'i        : ', i
#         print 'pose x   : ', pose_x
#         print 'pose q   : ', pose_q
#         print 'Predicted x : ', predicted_x
#         print 'Predicted q : ', predicted_q
         
        
        
        # convert to numpy
        pose_q = np.array(pose_q)
        pose_x = np.array(pose_x)
        predicted_q = np.array(predicted_q)
        predicted_x = np.array(predicted_x)
    
        #Compute Individual Sample Error
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        
#         # fix NAN isues
#         if d>1.0:  # d>1.0 --> arccoss(d) = nan
#             d=1.0
        
        theta = 2 * np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(pose_x-predicted_x)
     
        results[i,:] = [error_x,theta]
        
#         print 'd=', d
#         # CHECK IF theta is NAN
#         if math.isnan(theta):
#             print 'Found theta is NAN -- d=', d, '-- arccos(d)=', np.arccos(d)
        
        if error_x < 0.08 and theta < 4.0:
            counter_good_pred+=1
        
        i = i + 1
        #print 'Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta, '\n'
    
    median_result = np.median(results,axis=0)
    median_error = 'Median_error  ' + str(median_result[0]) + ' m  and ' + str(median_result[1]) + ' degrees'
    print median_error
    
    average_result = np.average(results, axis=0)
    average_error = 'Average_error ' + str(average_result[0]) + ' m  and ' + str(average_result[1]) + ' degrees'
    print average_error
    
    acc_statistic = (float(counter_good_pred)/float(len(testY))) * 100
    print 'Total good prediction: ', counter_good_pred
    print 'Accuracy: ', acc_statistic
    
    # save error results to file
    error_all_path = os.path.join(predition_path, 'error_all.txt')
    error_summary_path = os.path.join(predition_path, 'error_summary.txt')

    np.savetxt(error_all_path, results, delimiter=' ', fmt='%02.20f')

    fout = open(error_summary_path, 'w')
    #fout.write(median_error + '\n')
    #fout.write(average_error + '\n')
    fout.write('scene_id current_percentage median_error_met average_error_met median_error_deg average_error_deg accuracy_statistic\n')
    results_string = scene_id + ' lstm' + str(current_percentage) + ' '+ str(median_result[0]) + ' ' + str(average_result[0]) + ' ' + str(median_result[1]) + ' ' + str(average_result[1]) + ' ' + str(acc_statistic)
    #fout.write(scene_id + ' ' + str(median_result[0]) + ' ' + str(average_result[0]) + ' ' + str(median_result[1]) + ' ' + str(average_result[1]) + ' ' + str(acc_statistic))
    fout.write(results_string)
    
    return results_string
    
    
def main_predict_and_evaluate_percentage_img(in_scene_id, in_net_id, in_model_file_name):
    
    # load 1 model to test all list percentage
    #log_model_path = os.path.join(root_path, 'output', in_scene_id, in_net_id, 'img_pose_all', 'log_model')
    log_model_path = os.path.join(root_path, 'output', in_scene_id, in_net_id, split_id, 'log_model')
    model_file = os.path.join(log_model_path, in_model_file_name)    
    trained_model = load_model(model_file)
    print 'Loading model: ', model_file, '...done!'
     
    all_results = []
    
     
    #list_percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    list_percentage = [100]
     
    for l in list_percentage:
        
        print '----------------------------------'
        current_percentage = l
        #current_percentage = 100
        results_string = predict_and_evaluate_percentage_img(in_scene_id, in_net_id, trained_model, current_percentage)
        print 'CURRENT RESULT STRING: ', results_string
        all_results.append(results_string)
     
     
    # write all_results to file
    #out_summary_all_folder = '/home/anguyen/workspace/paper_src/2018.icra.event.source/output/shapes_rotation/vgg_lstm2/img_pose_all'
    out_summary_all_folder = os.path.join(root_path, 'output', in_scene_id, in_net_id, split_id)
    out_summary_file = out_summary_all_folder + '/ALL_RESULTS.TXT'
    fsum = open(out_summary_file, 'w')
     
    for ar in all_results:
        fsum.write(ar + '\n')
    
        
    
if __name__ == '__main__':
    
    in_scene_id = 'shapes_6dof'
    in_net_id = 'vgg_lstm2'
    
    in_model_file_name = 'full_model_epoch_e1000.hdf5'
      
    main_predict_and_evaluate_percentage_img(in_scene_id, in_net_id, in_model_file_name)

        
    print 'ALL DONE!'
