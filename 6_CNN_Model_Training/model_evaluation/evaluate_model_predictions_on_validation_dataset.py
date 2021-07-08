#!/usr/bin/env python
import confusionmatrix as cm
import glob
import numpy as np
import scipy.misc
import sys
import os.path

RGB_LABEL_IMAGES = True
RGB_CLASS_0 = (166,86,40)
RGB_CLASS_1 = (228,26,28)
RGB_CLASS_2 = (155,126,184)
RGB_CLASS_3 = (255,127,0)


def check_directories(input_dir):
    # Check if all subdirectories exist and create variables with their paths. If not, exit with error message

    # Create subfolder paths
    prediction_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    # Semantic subfolders

    # Check all folder paths
    if not os.path.isdir(prediction_dir):
        sys.exit('Submission directory ({0}) does not exist.'.format(prediction_dir))
    if not os.path.isdir(truth_dir):
        sys.exit('Submission directory ({0}) does not exist.'.format(truth_dir))

    return prediction_dir, truth_dir

def semantic_evaluation(prediction_dir, truth_dir):

    num_classes = 4

    obsImagePaths = glob.glob(os.path.join(truth_dir, '*.png'))
    predImagePaths = glob.glob(os.path.join(prediction_dir, '*.png'))

    print('Num. observed images:  ' + str(len(obsImagePaths)))
    print('Num. predicted images: ' + str(len(predImagePaths)))
    if (len(predImagePaths) > 0):
        # Make sure that there are as many predicted images and ground truth images
        assert len(obsImagePaths) == len(predImagePaths)
    
        # Create confusion matrix for accumulating the results
        confMat = cm.confusionmatrix(num_classes)
        confMat.Reset()

        # Loop through all files in specified folder
        for imageNo, obsImagePath in enumerate(obsImagePaths):

            imageName = os.path.basename(obsImagePath)

            # Print progress
            print('{:2d}/{:d}'.format(imageNo+1, len(obsImagePaths)) + ' : ' + imageName)

            # Load ground truth and predicted image
            obsImage = scipy.misc.imread(obsImagePath)
            predImagePath = os.path.join(prediction_dir, imageName)
            predImage = scipy.misc.imread(predImagePath)
            
            #if RGB_LABEL_IMAGES:
            if obsImage.shape[-1]==3:
                obsImageMapped = np.zeros((obsImage.shape[0],obsImage.shape[1]))
                rgb_mask = obsImage==RGB_CLASS_0
                obsImageMapped[rgb_mask[:,:,0]] = 0
                rgb_mask = obsImage==RGB_CLASS_1
                obsImageMapped[rgb_mask[:,:,0]] = 1
                rgb_mask = obsImage==RGB_CLASS_2
                obsImageMapped[rgb_mask[:,:,0]] = 2
                rgb_mask = obsImage==RGB_CLASS_3
                obsImageMapped[rgb_mask[:,:,0]] = 3
                
                obsImage = obsImageMapped
			
            # Add predictions to confusion matrix
            confMat.Append(obsImage, predImage)

        print(confMat)

        # Extract quantitative results from confusion matrix
        IoU = confMat.intersectionOverUnion()*100
        accuracy = confMat.accuracy()*100
        class_accuracy = confMat.class_accuracy()*100

        # Create output results
        result_dict = {'mean_IoU': np.mean(IoU),
                    'soil_IoU': IoU[0],
                    'clover_IoU': IoU[1],
                    'broadleaf_IoU': IoU[2],
                    'grass_IoU': IoU[3],
                    'mean_accuracy': accuracy,
                    'soil_accuracy': class_accuracy[0],
                    'clover_accuracy': class_accuracy[1],
                    'broadleaf_accuracy': class_accuracy[2],
                    'grass_accuracy': class_accuracy[3]}
        
    return result_dict


if __name__ == '__main__':
    input_dir = "input_512x512"#sys.argv[1]
    print('Python version:')
    print(sys.version)

    ###########################
    ## Evaluate model ##
    ###########################
    prediction_dir = 'openvino/css_pots_512x512_FP32_small/Images/'
    truth_dir = '../liveDemo/val_dataset2/labels/crop_rgb_1080x1080_scaled_512x512/'
    # Evaluate semantic segmentation
    results_semantic = semantic_evaluation(prediction_dir, truth_dir)
    fields = [results_semantic, prediction_dir, truth_dir]
    fd = open('results.txt','a')
    fd.write((','.join(map(str, fields)) + '\n'))
    fd.close()

    print(results_semantic)

