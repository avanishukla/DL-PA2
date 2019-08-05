# Import libraries
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

# Id ---- 0-12287 features ---label
# 0-8768-------------------------0-19
# Valid: 0-955
# Test: 0-971


def rotate(image_array: ndarray):
    degree=30
    tempr = random.uniform(-degree, degree)
    return sk.transform.rotate(image_array, tempr).reshape(1,12288)
def rotate90l(image_array: ndarray):
    degree=30
    tempr = random.uniform(-degree, degree)
    return sk.transform.rotate(image_array, -90).reshape(1,12288)
def rotate90r(image_array: ndarray):
    degree=30
    tempr = random.uniform(-degree, degree)
    return sk.transform.rotate(image_array, 90).reshape(1,12288)


def random_noise(image_array: ndarray):
    # add random noise to the image
    
    return sk.util.random_noise(image_array).reshape(-1,12288)

def toGray(image_array: ndarray):
    # add random noise to the image
    
    return sk.color.rgb2gray(image_array).reshape(-1,12288)

def vflip(image_array: ndarray):
    return image_array[...,::-1,:].reshape(-1,12288)
def hflip(image_array: ndarray):
    return image_array[...,:,::-1].reshape(-1,12288)

def contrast(original_image: ndarray):
    v_min, v_max = np.percentile(original_image, (0.2, 99.8))
    return exposure.rescale_intensity(original_image, in_range=(v_min, v_max)).reshape(-1,12288)

def brightness(original_image: ndarray):
    return exposure.adjust_gamma(original_image, gamma=0.4, gain=0.9).reshape(-1,12288)


def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scale = 0.9
    boxes = np.zeros(4)

    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes = [x1, y1, x2, y2]
    batch_size = x.shape[0]
    # Create different crops for an image
    crops = tf.image.crop_and_resize(x, boxes=[boxes]*(batch_size), box_ind=[i for i in range(batch_size)], crop_size=(64, 64))
    # Return a random crop
    return crops


def zoommore(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scale = 0.85
    boxes = np.zeros(4)

    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes = [x1, y1, x2, y2]
    batch_size = x.shape[0]
    # Create different crops for an image
    crops = tf.image.crop_and_resize(x, boxes=[boxes]*(batch_size), box_ind=[i for i in range(batch_size)], crop_size=(64, 64))
    # Return a random crop
    return crops

def zoomtoomuch(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scale = 0.8
    boxes = np.zeros(4)

    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes = [x1, y1, x2, y2]
    batch_size = x.shape[0]
    # Create different crops for an image
    crops = tf.image.crop_and_resize(x, boxes=[boxes]*(batch_size), box_ind=[i for i in range(batch_size)], crop_size=(64, 64))
    # Return a random crop
    return crops

def crop_random(x: tf.Tensor) -> tf.Tensor:
    
    crops = tf.random_crop(x,[8769,54,54,3])
    cropsz = tf.image.resize(crops,[64,64])
    return cropsz



def data_augmentation():
  train_data1 = train_data[0:8769].reshape((-1,64,64,3))

  for i in range(8769):
      train_data[8769+i,:] = rotate(train_data1[i])

  train_label[8769:2*8769] = train_label[0:8769]
  train_data[2*8769:3*8769,:] = hflip(train_data1)

  for i in range(8769):
      train_data[3*8769+i,:] = rotate90l(train_data1[i])

  sess = tf.Session()
  with sess.as_default():
      #train_data[4*8769:5*8769,:] = zoommore(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[5*8769:6*8769,:] = zoom(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[8*8769:9*8769,:] = zoomtoomuch(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[9*8769:10*8769,:] = crop_random(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[10*8769:11*8769,:] = crop_random(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[11*8769:12*8769,:] = crop_random(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[12*8769:13*8769,:] = crop_random(tf.constant(train_data1)).eval().reshape(-1,12288)
      train_data[13*8769:14*8769,:] = hflip((tf.constant(train_data1)).eval())
      train_data[14*8769:15*8769,:] = hflip((tf.constant(train_data1)).eval())
      train_data[15*8769:16*8769,:] = hflip((tf.constant(train_data1)).eval())
      train_data[16*8769:17*8769,:] = hflip((tf.constant(train_data1)).eval())

  train_label[4*8769:5*8769] = train_label[0:8769]
  train_data[4*8769:5*8769,:] = contrast(train_data1)


  for i in range(8769):
      train_data[6*8769+i,:] = rotate90r(train_data1[i])

  train_data[7*8769:8*8769,:] = vflip(train_data1)


  for zz in range(17):
      train_label[zz*8769:(zz+1)*8769] = train_label[0:8769]
  del train_data1


n_input = 64
n_features = 64
n_classes = 20
tf.reset_default_graph()
#print(train_X.dtype)
regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

x = tf.placeholder("float", [None, n_features,n_features,3],name = 'x')
y = tf.placeholder("float", [None, n_classes],name = 'y')
prob = tf.placeholder_with_default(1.0, shape=(),name = 'prob')
prob_conv = tf.placeholder_with_default(0.15, shape=(),name = 'prob_conv')
training = tf.placeholder_with_default(1, shape=(),name = 'training')

def cnn_model_fn(features):
#variance_scaling_initializer()
#xavier_initializer(uniform=False)
    """Model function for CNN."""
    # Input Layer
    #input_layer = tf.reshape(features["x"], [-1, , 28, 1])
    input_layer = features

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      use_bias=True,
      kernel_size=[5, 5],
      padding="same",name='conv1',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv1 = tf.layers.batch_normalization(conv1,name='bconv1', training = (training==1))
    #pool1 = tf.layers.max_pooling2d(inputs=conv1,name='pool1', pool_size=[2, 2], strides=2)
    #dropout1 = tf.layers.dropout(inputs=pool1,name='', rate=prob_conv)
    #bpool1 = tf.layers.batch_normalization(bconv11,name='bpool1', training = (training==1))
    
    # Convolutional Layer #1
    conv2 = tf.layers.conv2d(
      inputs=bconv1,
      filters=32,
      use_bias=True,
      kernel_size=[5, 5],
      padding="same",name='conv2',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv2 = tf.layers.batch_normalization(conv2,name='bconv2', training = (training==1))
    pool2 = tf.layers.max_pooling2d(inputs=bconv2,name='pool2', pool_size=[2, 2], strides=2)
    #dropout1 = tf.layers.dropout(inputs=pool1,name='', rate=prob_conv)
    bpool2 = tf.layers.batch_normalization(pool2,name='bpool2', training = (training==1))
    
    # Convolutional Layer #1
    conv3 = tf.layers.conv2d(
      inputs=bpool2,
      filters=64,
      use_bias=True,
      kernel_size=[3, 3],
      padding="same",name='conv3',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv3 = tf.layers.batch_normalization(conv3,name='bconv3', training = (training==1))
    #pool3 = tf.layers.max_pooling2d(inputs=bconv3,name='pool3', pool_size=[2, 2], strides=2)
    #dropout1 = tf.layers.dropout(inputs=pool1,name='', rate=prob_conv)
    #bpool3 = tf.layers.batch_normalization(pool3,name='bpool3', training = (training==1))
    
    # Convolutional Layer #1
    conv4 = tf.layers.conv2d(
      inputs=bconv3,
      filters=64,
      use_bias=True,
      kernel_size=[3, 3],
      padding="same",name='conv4',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv4 = tf.layers.batch_normalization(conv4,name='bconv4', training = (training==1))
    pool4 = tf.layers.max_pooling2d(inputs=bconv4,name='pool4', pool_size=[2, 2], strides=2)
    #dropout1 = tf.layers.dropout(inputs=pool1,name='', rate=prob_conv)
    bpool4 = tf.layers.batch_normalization(pool4,name='bpool4', training = (training==1))
    
    # Convolutional Layer #1
    conv5 = tf.layers.conv2d(
      inputs=bpool4,
      filters=64,
      use_bias=True,
      kernel_size=[3, 3],
      padding="same",name='conv5',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv5 = tf.layers.batch_normalization(conv5,name='bconv5', training = (training==1))
    #pool5 = tf.layers.max_pooling2d(inputs=bconv5,name='pool5', pool_size=[2, 2], strides=2)
    #dropout5 = tf.layers.dropout(inputs=pool5,name='', rate=prob_conv)
    #bpool5 = tf.layers.batch_normalization(pool5,name='bpool5', training = (training==1))
    
    # Convolutional Layer #1
    conv6 = tf.layers.conv2d(
      inputs=bconv5,
      filters=128,
      use_bias=True,
      kernel_size=[3, 3],
      padding="valid",name='conv6',
      activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    # Pooling Layer #1
    bconv6 = tf.layers.batch_normalization(conv6,name='bconv6', training = (training==1))
    pool6 = tf.layers.max_pooling2d(inputs=bconv6,name='pool6', pool_size=[2, 2], strides=2)
    #dropout6 = tf.layers.dropout(inputs=pool6,name='', rate=prob_conv)
    bpool6 = tf.layers.batch_normalization(pool6,name='bpool6', training = (training==1))
    
    
    
    
    pool6_flat = tf.reshape(bpool6, [-1, 7 * 7 * 128],name='pool6_flat')
    dense1 = tf.layers.dense(inputs=pool6_flat,use_bias=True,name='dense1', units=256, activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    dropout7 = tf.layers.dropout(inputs=dense1,name='dropout7', rate=prob)
    bdropout7 = tf.layers.batch_normalization(dropout7,name='bdropout7', training = (training==1))
#     dense2 = tf.layers.dense(inputs=bdropout7,use_bias=True,name='dense2', units=300, activation=tf.nn.leaky_relu, kernel_initializer=myinit
#     dropout8 = tf.layers.dropout(inputs=dense2, rate=prob,name='dropout8')
#     bdropout8 = tf.layers.batch_normalization(dropout8,name='bdropout8', training = (training==1))
    
#     dense3 = tf.layers.dense(inputs=dropout8,use_bias=True, units=100, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=1e-4))
#     dropout9 = tf.layers.dropout(inputs=dense3, rate=prob)
#     dropout9 = tf.layers.batch_normalization(dropout9, training = (training==1))
    # Logits Layer
    logits = tf.layers.dense(inputs=bdropout7,name='logits', units=20,use_bias=True, activation=tf.nn.leaky_relu, kernel_initializer=myinit)
    blogits = tf.layers.batch_normalization(logits,name='blogits', training = (training==1))
    return [conv1,conv2,conv3,conv4,conv5,conv6,bconv1,bconv2,bconv3,bconv4,bconv5,bconv6,pool2,pool4,pool6,bpool2,bpool4,bpool6,logits,dense1, bdropout7,dropout7,input_layer,pool6_flat],blogits


def train_net():
    # Initializing the variables
  def write_log_files():
      str_1 = '\n'.join(log_str_arr_train[0:join_till])
      str_2 = '\n'.join(log_str_arr_val[0:join_till])
      text_file = open('log_train.txt', "w")
      text_file.write(str_1)
      text_file.close()
      text_file = open('log_val.txt', "w")
      text_file.write(str_2)
      text_file.close()
  sdata,pred = cnn_model_fn(x)
  for data in sdata:
      tf.add_to_collection('model',data)
  tf.identity(pred,name = 'pred')
  tf.add_to_collection('model',pred)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name='cost')
  tf.add_to_collection('model',cost)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,name='optimizer')
  tf.add_to_collection('model',optimizer)
  #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1),name='correct_prediction')
  tf.add_to_collection('model',correct_prediction)
  #calculate accuracy across all the given images and average them out. 

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
  tf.add_to_collection('model',accuracy)
  tf.add_to_collection('model',x)

  earlystop = 0
  earlystopcritera = 5
  # Initializing the variables
  init = tf.global_variables_initializer()
  log_str_arr_train = ["" for x in range(int((len(train_X)*training_iters)/(100*batch_size)))]
  log_str_arr_val = ["" for x in range(int((len(train_X)*training_iters)/(100*batch_size)))]
  join_till = int(training_iters*(len(train_X)/(100*batch_size)))
  step_counter=0
  log_str_arr_indx=0
  with tf.Session() as sess:
      tf.set_random_seed(1234)
      sess.run(init) 
      temp = -1
      for i in range(training_iters):
          for batch in range(len(train_X)//batch_size):
              batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
              batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    

              opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y,prob: 0.5,prob_conv: 0.2, training:1})
              loss= sess.run(cost, feed_dict={x: batch_x,y: batch_y})
              if step_counter%100 == 0:
                  val_loss = sess.run(cost, feed_dict={x: val_X,y : val_y})
                  log_str_arr_train[log_str_arr_indx] = 'Epoch '+str(i)+', Step '+str(step_counter)+', Loss: '+str(loss)
                  log_str_arr_val[log_str_arr_indx] = 'Epoch '+str(i)+', Step '+str(step_counter)+', Loss: '+str(val_loss)
                  log_str_arr_indx+=1
                  #print(log_str_arr_train)
              step_counter+=1
          print("Iter " + str(i) + ", Loss= "+str(loss))
          print("Optimization Finished!")
          # Calculate accuracy for all 10000 mnist test images
          val_acc,val_loss = sess.run([accuracy,cost], feed_dict={x: val_X,y : val_y})
          print("Testing Accuracy:","{:.5f}".format(val_acc))
          if i%5==0:
              write_log_files()
          if i >= 0 and val_acc>temp:
              earlystop = 0
              predicted_label = sess.run(pred, feed_dict={x: test_X})
              predicted_label = np.argmax(predicted_label, 1)
              id_vector = np.arange(len(predicted_label))
              add_to_csv = np.column_stack((id_vector, predicted_label))
              f = open("predictions{0}{1}.csv".format(i,str(val_acc)),"w")
              f.write('id,label\n')
              np.savetxt(f, add_to_csv.astype(int), fmt='%i', delimiter=",")
              f.close()
              saver = tf.train.Saver(max_to_keep = 10000)
              saver.save(sess, 'mymodel{0}{1}'.format(i,'_a'))
              temp = val_acc
          else:
              earlystop+=1
              if earlystop>=earlystopcritera:
                  break

parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--batch_size")
parser.add_argument("--init")
parser.add_argument("--save_dir")
parser.add_argument("--epochs")
parser.add_argument("--dataAugment")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--test")
args = parser.parse_args()

learning_rate = float(args.lr)
batch_size = int(args.batch_size)
init = int(args.init)                  #1 for Xavier, 2 for He
save_dir = args.save_dir
training_iters = int(args.epochs)
dataAugment = int(args.dataAugment)    #1 for yes and 0 for no
train = args.train
val = args.val
test = args.test
n_input = 64
n_features = 64
n_classes = 20
tf.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)

train_data_read = pd.read_csv(train)
if dataAugment==1:
  train_data = np.ones([17*8769,12288], dtype=np.float32)
  train_label = np.ones(17*8769, dtype=int)
else:
    train_data = np.ones([8769,12288], dtype=float)
    train_label = np.ones(8769, dtype=int)
train_label[0:8769] = train_data_read.iloc[:,12289].values
train_data[0:8769,:] = train_data_read.iloc[:,1:12289].values
del train_data_read
train_data[0:8769,:] = train_data[0:8769,:]/255.0

if dataAugment==1:
    data_augmentation()
train_data = (train_data-train_data.mean(axis=0))/train_data.std(axis=0)
train_X = train_data.reshape((-1,64,64,3))
del train_data
gc.collect()
batch_size = int(args.batch_size)
train_y = np.zeros((train_label.size, n_classes))
train_y[np.arange(train_label.size), train_label] = 1

validation_data = pd.read_csv(val)
validation_label = validation_data.iloc[:,12289].values
validation_data = validation_data.iloc[:,1:12289].values
validation_data = validation_data/255.0
validation_data = (validation_data-validation_data.mean(axis=0))/validation_data.std(axis=0)
val_X = validation_data.reshape((-1,64,64,3))
del validation_data
val_y = np.zeros((validation_label.size, n_classes))
val_y[np.arange(validation_label.size), validation_label] = 1

test_data = pd.read_csv(test)
test_data = test_data.iloc[:,1:12289].values
test_data = test_data/255.0
test_data = (test_data-test_data.mean(axis=0))/test_data.std(axis=0)
test_X = test_data.reshape((-1,64,64,3))
del test_data
gc.collect()

tf.reset_default_graph()
x = tf.placeholder("float", [None, n_features,n_features,3],name = 'x')
y = tf.placeholder("float", [None, n_classes],name = 'y')
prob = tf.placeholder_with_default(1.0, shape=(),name = 'prob')
prob_conv = tf.placeholder_with_default(0.15, shape=(),name = 'prob_conv')
training = tf.placeholder_with_default(1, shape=(),name = 'training')
if init==1:
    myinit = tf.contrib.layers.xavier_initializer(uniform=False)
elif init==2:
    myinit = tf.contrib.layers.variance_scaling_initializer()

pred = cnn_model_fn(x)
tf.identity(pred,name = 'pred')
tf.add_to_collection('model',pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name='cost')
tf.add_to_collection('model',cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,name='optimizer')
tf.add_to_collection('model',optimizer)
#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1),name='correct_prediction')
tf.add_to_collection('model',correct_prediction)
#calculate accuracy across all the given images and average them out. 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
tf.add_to_collection('model',accuracy)
train_net()
