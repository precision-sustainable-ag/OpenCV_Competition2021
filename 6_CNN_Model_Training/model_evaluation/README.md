# Model evaluation:

This folder contains scripts used for evaluating the trained models. 

_deploy_model_for_dataset_evaluation.py_ loads a frozen tensorflow 1.15 model and performs semantic segmentation on the images in the specified folder and exports the results to a result-folder.

_evaluate_model_predictions_on_evaluate_model_predictions_on_validation_dataset.py_ loads the predicted segmentations from above and compares them with an annotated validation dataset. Based on the discrepancy between the two sets, accuracy and intersection over union measurements are exported.
