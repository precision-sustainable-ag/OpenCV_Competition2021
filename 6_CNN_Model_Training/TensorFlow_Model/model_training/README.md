## CNN training
For training the CNN models, follow the guide below:
1. Install [tensorflow 1.15.0](https://www.tensorflow.org/install/pip).
2. Check out the [tensorflow github repository](https://github.com/tensorflow/models/tree/master/research).
3. Check out this repository and copy it into the repository above. 
4. Download the [training](https://vision.eng.au.dk/?download=/data/oakd/2021/train.zip) and [validation dataset](https://vision.eng.au.dk/?download=/data/oakd/2021/val.zip)
5. Configure and run deeplab/datasets/build_opencv_data.py to generate tfrecord of the downloaded datasets.
6. Download the [pretrained DeepLabV3+ model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). This project uses the deeplab_mnv3_large_cityscapes_trainfine pretrained model.
7. Train the model for 60.000 iterations (see command below for reference).
8. Export the model (see command below for reference)


Command for training the model:
> python deeplab/train_adam.py --logtostderr --training_number_of_steps=60000 --train_split="dataset4_combined_train" --model_variant="mobilenet_v3_large_seg" --tf_initial_checkpoint=deeplab/pretrained_models/deeplab_mnv3_large_cityscapes_trainfine/model.ckpt --dataset_dir=deeplab/datasets/opencv/tfrecord --num_clones=3 --base_learning_rate=0.00001 --initialize_last_layer=False --last_layers_contain_logits_only=False --train_crop_size="768,768" --train_batch_size=18 --dataset="opencv" --save_interval_secs=300 --save_summaries_secs=300 --save_summaries_images=True --train_logdir=deeplab/logs/opencv/mobilenetv3_large_data4_combined_class_weights

Command for validation results during the training process:
> python deeplab/eval_4c_opencv.py --logtostderr --eval_split="dataset3_crop_val" --model_variant="mobilenet_v3_large_seg" --eval_crop_size="512,512" --min_resize_value=512 --max_resize_value=512 --dataset="opencv" --checkpoint_dir=deeplab/logs/opencv/mobilenetv3_large_data4_combined_class_weights --eval_logdir=deeplab/logs/opencv/eval/mobilenetv3_large_data4_combined_class_weights --dataset_dir=deeplab/datasets/opencv/tfrecord

Command for exporting the model (for 512x512 image input resolution)
> python deeplab/export_model_simplified.py --logtostderr --checkpoint_path=deeplab/logs/opencv/mobilenetv3_large_data4_combined_class_weights/model.ckpt-60000 --add_flipped_images=False --export_path="4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512.pb" --model_variant="mobilenet_v3_large_seg" --num_classes=4 --vis_crop_size=512 --vis_crop_size=512 --crop_size=512 --crop_size=512 --inference_scales=1.0 --dataset="opencv"

For additional details please see the [projet wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/6.-CNN-Model-Training)
