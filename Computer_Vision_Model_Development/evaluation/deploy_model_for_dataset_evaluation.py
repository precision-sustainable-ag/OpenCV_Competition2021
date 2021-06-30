import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
import glob
import ntpath
import os
import imageio
from skimage import color
from scipy import ndimage

from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=16)
futures = []

from timeit import default_timer as timer

rescale_and_crop = False

im_input_path = "datasets/val/Images/crop_1080x1080_scaled_512x512/"
im_output_path = "output/4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512/"
current_model = "4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512_without_softmax.pb"
    
    
class DeepLabModel():
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.

        with tf.gfile.GFile(path, 'rb')as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          seg_map: np.array. values of pixels are classes
        """

        batch_seg_map = self.sess.run(
           [self.OUTPUT_TENSOR_NAME],
           feed_dict={self.INPUT_TENSOR_NAME: [image]})

        seg_map = batch_seg_map[0]        

        return seg_map

model = DeepLabModel("frozen_models/" + current_model)
i = 0
im_output_path_vis = im_output_path + "visual/"
im_output_path_vis_seg = im_output_path + "visual_segmentation/"

os.makedirs(im_output_path, exist_ok=True)
os.makedirs(im_output_path_vis, exist_ok=True)
for im_filename_long in glob.glob(im_input_path + '*.jpg'):
    print(i)
    while(executor._work_queue.qsize()>5):
        time.sleep(0.5)
    
    img = Image.open(im_filename_long)
    img = np.asarray(img)
    if(rescale_and_crop):
        img = ndimage.zoom(img, (0.47, 0.47, 1))
        if(img.shape[0]>512 or img.shape[1]>512):
            img = img[0:512,0:512,:]
    
    im_filename = ntpath.basename(im_filename_long)[0:-4]
    seg_map = model.run(img)[0]
    a = executor.submit(imageio.imsave, im_output_path_vis + im_filename + "_segmentation_vis.png",(seg_map*64).astype(np.uint8), compress_level=3)
    a = executor.submit(imageio.imsave, im_output_path_vis + im_filename + "_input.jpg",img)
    a = executor.submit(imageio.imsave, im_output_path + im_filename + "_input.jpg",img)
    a = executor.submit(imageio.imsave, im_output_path + im_filename + ".png",(seg_map).astype(np.uint8), compress_level=3)
    i=i+1

    
while executor._work_queue.qsize():
    print('Queue size: '+str(executor._work_queue.qsize()))
    time.sleep(1)

    executor.shutdown(wait=True)
