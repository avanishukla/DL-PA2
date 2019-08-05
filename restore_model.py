####RESTORING-GRAPH###########
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#%matplotlib inline
import os
import pandas as pd
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
from sklearn.decomposition import PCA
import skimage as sk
from skimage import transform
from skimage import exposure
from numpy import ndarray
import random
import argparse
import sys


tf.reset_default_graph()
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('mymodel46_a.meta')
    new_saver.restore(sess, 'mymodel46_a')
    graph = tf.get_default_graph()
    mycol = graph.get_collection('model')
    accuracy = mycol[4]
    pred = mycol[0]
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    val_acc = sess.run(accuracy, feed_dict={x: val_X,y : val_y})
    print("Validation Accuracy:","{:.5f}".format(val_acc))
    predicted_label = sess.run(pred, feed_dict={x: test_X})
    predicted_label = np.argmax(predicted_label, 1)
    id_vector = np.arange(len(predicted_label))
    add_to_csv = np.column_stack((id_vector, predicted_label))
    f = open("predictions.csv","w")
    f.write('id,label\n')
    np.savetxt(f, add_to_csv.astype(int), fmt='%i', delimiter=",")
    f.close()