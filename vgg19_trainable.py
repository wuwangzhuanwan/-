#coding:utf-8
import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [0.0, 0.0, 0.0]

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict  = {}
        self.trainable = trainable
        self.dropout   = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        #这里边输入的RGB图像是已经是经过缩放的了
        rgb_scaled = rgb * 1.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:]   == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:]  == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2]])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")#两个数字分别代表多少个通道,多少个输出 这里就是3通道 64个输出
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")#这里边包含卷积和relu两步
        self.pool1   = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2   = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3   = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4   = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5   = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512 这个计算步骤也懂了
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7    = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7  = tf.nn.relu(self.fc7)
        
        if train_mode is not None:
            self.relu7  = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7  = tf.nn.dropout(self.relu7, self.dropout)
           
    def mynetwork(self,bottomm,name,categorynum):
        self.names = globals()
        self.names['fc_%s'%(name)]  = self.fc_layer(self.relu7,4096,categorynum,'fc_%s'%(name))
        self.names['prob_%s'%(name)] = tf.nn.softmax(self.names['fc_%s'%(name)],name = 'prob_%s'%(name))
        self.data_dict = None
        
    def avg_pool(self, bottom, name):#均值pooling
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):#最大值pooling
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):#真正返回的名字是:name/name+'filter'和name/name+'filter'
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)#返回的变量是带有第四个参数name+'filter'和name+'biase'这个参数名字的
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):#variable_scope主要是变量共享的 在这个scope中去找 
            weights, biases = self.get_fc_var(in_size, out_size, name)#返回的变量是带有第三个参数name+'filter'和name+'biase'这个参数名字的 
            x = tf.reshape(bottom, [-1, in_size])#把bottom给reshape一下,用于全连接
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)#这个是全连接的卷积操作
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):#这个得到卷积层的变量,如果训练模型中有则从训练模型中得到,否则自己构造
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)#这个先用来构造filter初始值
        filters = self.get_var(initial_value, name, 0, name + "_filters")#从训练好的模型中得到以name和0为参数的变量,如果得不到,则使用initial_value
                                                                         #的值作为最后的返回值,名字为name + "_filters",另外,这个函数完成之后
                                                                         #还会把返回值给字典var_dict,另外,如果是可训练,返回的是变量,否则返回的是
                                                                         #常数
        initial_value = tf.truncated_normal([out_channels], .0, .001)    #这个是用来构造偏置项
        biases  = self.get_var(initial_value, name, 1, name + "_biases") #同以上

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):#这个是得到全连接层变量,如果训练模型中有则从训练模型中得到,否则自己构造
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)#用这个构造全连接层的初始参数
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)           #全连接层也是有偏置项的好吧
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):       #得到变量 这个是从训练好的模型中得到变量
        if self.data_dict is not None and name in self.data_dict:#如果data_dict不是空的并且name参数在data_dict字典里边
            value = self.data_dict[name][idx]                    #首先,如果字典里边不为空,那么从字典里读取出来 name 参数以及 idx 参数对应的数据
        else:                                                    #赋值给 value ,如果字典为空,则 value 的值为 "initial_value" 参数
            value = initial_value

        if self.trainable:                                       #如果是可训练的,则 var 的值为变量类型的 value 值
            var = tf.Variable(value, name=var_name)
        else:                                                    #如果是不可训练的,则 var 的值为常数类型的 value 值
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var                         #给 var_dict 这个字典赋值

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()      #var 取得是value的值,那么它的尺度应该和value一样,而value来自于initial_value
                                                                 #所以断言var和initial_value尺度一样
        return var                                               #最终返回的参数名字是最后一个参数var_name

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):                                    #得到所有的参数数目
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
