#coding:utf-8

import numpy as np
import tensorflow as tf

def get_files(image_root,filename):
    class_train = []
    label_train = []
    image_root = image_root + '/'
    for line in open(filename):
        class_train.append(image_root+line.split(' ')[0])
        label_train.append(line.split(' ')[1].split('\n')[0])
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list,label_list

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels = 3)
    image = tf.image.resize_images(image,(resize_w,resize_h),method = 2)

    image_batch,label_batch = tf.train.shuffle_batch( [image,label],
                                                      batch_size = batch_size,
                                                      num_threads = 64,
                                                      capacity = capacity,
                                                      min_after_dequeue=16)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])
    return images_batch,labels_batch