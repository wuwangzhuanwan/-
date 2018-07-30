#coding:utf-8

import numpy as np
import tensorflow as tf
from haffuman import build_haffuman_tree_for_each_layer,\
                     find_out_all_include_to_make_name,\
                     find_out_a_list_to_make_name,\
                     find_out_layernum_trainlabel_name_from_all_list,\
                     myindex
import vgg16_trainable as vgg16
from readImages import get_files,get_batches
import time
import os
import math
'''
设置参数 控制每一个块的运行
'''
MOVING_AVERAGE_DECAY    = 0.999
LEARNING_RATE_BASE      = 0.001
LEARNING_RATE_DECAY     = 0.9
NUM_CALSS               = 100
IMAGE_SIZE              = 224
NUM_CHANNELS            = 3

batch_size              = 10
image_root              = 'dataset'
filename                = 'dataset/cifar_100/label/train_for_imbalance.txt'
valtxt                  = 'dataset/cifar_100/label/test.txt'
logtxt                  = 'log/'
save_model              = 'trainedmodel'

the_num_of_train_images = 22511
the_num_of_val_images   = 10000
inf_log                 = 'log/'
the_epoch_of_train      = 20
iiter                   = 0
epoch                   = int(math.ceil(float(the_num_of_train_images)/float(batch_size))) - 1
val                     = 0
val_interval            = 200#batch
step_size               = 8000

epoch1                  = 1
val1                    = 1

count_loss              = 0
print_loss_interval     = 10#batch

save_count              = 0
save_interval           = int(math.ceil(float(the_num_of_train_images)/float(batch_size))) #batch

'''
一些函数 loss trainop get_accuracy
'''
def loss(label_batches,logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batches,logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    return cost
 
def training(loss,lr,global_step):
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    return train_op

def get_accuracy(tempprediction,trainedlabel_assemble):
    assert len(tempprediction) == len(trainedlabel_assemble)
    lenth = len(tempprediction)
    count=0
    for predlabel,trainlabel in zip(tempprediction,trainedlabel_assemble):
        if np.argmax(predlabel) == trainlabel:
            count+=1
        else:
            break
    if count == lenth:
        return 'yes'
    else:
        return 'no'

'''
以下开始正常运行
'''        
if __name__ == '__main__':
    '''
    選擇使用哪一個卡,實驗室機器只有一個卡，只能爲０
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    '''
    创建log/xxx.txt文件夹，用于记录信息
    '''
    name = ''
    if not os.path.exists('log/0.txt'):
        open('log/0.txt', 'w').close()
        name = 'log/0.txt'
    else:
        L_name = []
        for root,dirs,files in os.walk('log/'):
            for line in files:
                th_file = line.split('.')[0]
                L_name.append(int(th_file))
        N =  max(L_name)+1  
        name = inf_log+str(N)+'.txt'
        
    fp = open(name,'w')
    
    #＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃從這裏開始就是建立哈弗曼分支的＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
    '''
    以txt方式记录运行日志
    '''
    names = globals()

    dictionary  = {0: 7, 1: 9, 2: 16, 3: 343, 4: 310, 5: 292, 6: 63, 7: 500, 8: 125, 9: 404, 10: 442, 11: 219, 12: 314, 13: 5, 14: 10, 15: 325, 16: 141, 17: 320, 18: 449, 19: 6, 20: 176, 21: 487, 22: 306, 23: 218, 24: 14, 25: 381, 26: 12, 27: 323, 28: 16, 29: 465, 30: 500, 31: 336, 32: 7, 33: 14, 34: 417, 35: 280, 36: 93, 37: 455, 38: 238, 39: 368, 40: 408, 41: 6, 42: 412, 43: 16, 44: 435, 45: 389, 46: 391, 47: 373, 48: 10, 49: 457, 50: 9, 51: 7, 52: 437, 53: 448, 54: 430, 55: 251, 56: 181, 57: 246, 58: 6, 59: 7, 60: 389, 61: 330, 62: 12, 63: 11, 64: 132, 65: 116, 66: 143, 67: 7, 68: 159, 69: 492, 70: 316, 71: 85, 72: 268, 73: 309, 74: 10, 75: 308, 76: 15, 77: 18, 78: 8, 79: 441, 80: 459, 81: 20, 82: 422, 83: 16, 84: 366, 85: 26, 86: 10, 87: 326, 88: 10, 89: 164, 90: 310, 91: 99, 92: 315, 93: 213, 94: 495, 95: 492, 96: 206, 97: 10, 98: 158, 99: 500}
    total_num = sum(list(dictionary.values()))
    print('the number of train data is %d'%(total_num))
    fp.write('the number of train data is %d\n'%(total_num))
    initInlcude = [[vv for vv in range(NUM_CALSS)]]
    '''
    得到树形的层数，以及每一层的叉子列表
    '''
    layer_num, layer_node =  build_haffuman_tree_for_each_layer(dictionary,initInlcude)
    '''
    打印树形中的每一叉子所包含的节点
    '''
    for i in range(layer_num):
        for each_list in layer_node[i]:
            for each_dict in each_list:
                print(each_dict)
                fp.write(str(each_dict))
                fp.write('\n')
                
    print('') 
    fp.write('\n')
    for i in range(layer_num):
        print(layer_node[i])
        fp.write(str(layer_node[i]))
        fp.write('\n')
        print('') 
        fp.write('\n')
    '''
    找到所有的名字 
    '''
    AllName = find_out_all_include_to_make_name(layer_node)
    print('所有名字分别为:') 
    fp.write('所有名字分别为:\n')
    for Name in AllName:
        print(Name)
        fp.write(Name)
        fp.write('\n')
    #＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃建立哈弗曼分支的部分結束＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃＃
    '''
    以下两个函数是用来读取数据的get_files get_batches，包括训练数据和测试数据
    '''  
    image,label = get_files(image_root,filename)
    image_batches,label_batches = get_batches(image,label,IMAGE_SIZE,IMAGE_SIZE,batch_size,64)
    
    image_val,label_val = get_files(image_root,valtxt)
    image_batches_val,label_batches_val = get_batches(image_val,label_val,IMAGE_SIZE,IMAGE_SIZE,1,64)
    '''
    全局变量，用于记录优化次数
    '''
    global_step = tf.Variable(0, trainable=False)
    '''
    要输入的图像以及训练模式 使用placehold()
    '''        
    images     = tf.placeholder(tf.float32, [1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
    train_mode = tf.placeholder(tf.bool)
    
    '''
    指数衰减学习率
    '''
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, the_num_of_train_images, LEARNING_RATE_DECAY,staircase=True)
    '''
    以下两句是用来构建网络的
    '''
    vgg = vgg16.vgg16('pretrained/vgg16.npy')
    vgg.build(images, train_mode)
    '''
    制作标签的真值，这个真值是训练时候的真值，使用placehold()
    '''
    for i in range(layer_num):
        for List in layer_node[i]:
            names['true_out_%dlayer_%s'%(i+1,find_out_a_list_to_make_name(List))] = tf.placeholder(tf.int32, [1, ])
    '''
    为每一层中的List制作loss 
    '''
    for i in range(layer_num):
        for List in layer_node[i]:
            
            name = find_out_a_list_to_make_name(List)  
            names['sum_loss%dlayer_%s'%(i+1,name)] = []
            vgg.mynetwork(vgg.relu7,name,len(List))
            names['loss_%dlayer_%s'%(i+1,name)] = loss(names['true_out_%dlayer_%s'%(i+1,name)],vgg.names['fc_%s'%(name)])
            
            for i_previous in range(i):
                for list_previous in layer_node[i_previous]:
                    name_previous = find_out_a_list_to_make_name(list_previous)  
                    if myindex(name_previous,name):
                        names['sum_loss%dlayer_%s'%(i+1,name)].append(names['loss_%dlayer_%s'%(i_previous+1,name_previous)])
                        
            names['sum_loss%dlayer_%s'%(i+1,name)].append(names['loss_%dlayer_%s'%(i+1,name)])
            temp_lenth = len(names['sum_loss%dlayer_%s'%(i+1,name)])
            avg_num = 1.0/float((temp_lenth*(temp_lenth+1))/2.0)
            temp = []
            for ii in range(temp_lenth):
                temp.append(names['sum_loss%dlayer_%s'%(i+1,name)][ii]*avg_num*(ii+1))
            names['loss_%dlayer_%s'%(i+1,name)] = sum(temp)
    '''
    为每一层制作train_op
    '''
    for i in range(layer_num):
        for List in layer_node[i]:
            name = find_out_a_list_to_make_name(List)   
            names['trainop_%dlayer_%s'%(i+1,name)] = training(names['loss_%dlayer_%s'%(i+1,name)],learning_rate,global_step)
    '''
    建立session
    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    '''
    创建线程队列
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
       for idx in range(int(math.ceil(float(the_num_of_train_images)/float(batch_size))) * the_epoch_of_train):
           if coord.should_stop():
               break
           count_loss+=1
           epoch+=1
           iiter+=1
           val+=1
           save_count+=1

           if epoch % (int(math.ceil(float(the_num_of_train_images)/float(batch_size)))) == 0:
               print('begin the '+str(epoch1)+'th epoch......')
               fp.write('\nbegin the '+str(epoch1)+'th epoch......\n')
               if epoch1 == 1:
                   epoch = 1
               else:
                   epoch = 0
               epoch1+=1
               
           imgss,labss = sess.run([image_batches,label_batches])
           
           for i in range(len(labss)):
               
               imgs = np.reshape(imgss[i,:,:,:],[1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
               
               labs = labss[i]
               
               '''
               规划到来的这一个图像的一些数值 比如说训练时候的标签，placehold的填充
               '''
               layernum_assemble, name_assemble ,trainedlabel_assemble = find_out_layernum_trainlabel_name_from_all_list(layer_node,labs)
               
               feed_net_dict = {}
               feed_net_dict = {train_mode:True,images:imgs} 
               for LayerNum,Name,Label in zip(layernum_assemble,name_assemble,trainedlabel_assemble):
                    feed_net_dict[names['true_out_%dlayer_%s'%(LayerNum,Name)]] = [Label]
                    
               train_op        = names['trainop_%dlayer_%s'%(layernum_assemble[len(layernum_assemble)-1],name_assemble[len(name_assemble)-1])]
               cost_op         = names['loss_%dlayer_%s'%(layernum_assemble[len(layernum_assemble)-1],name_assemble[len(name_assemble)-1])]
               _,loss_value,lr = sess.run([train_op,cost_op,learning_rate],feed_dict=feed_net_dict) 
                    
               if count_loss % print_loss_interval == 0:
                   '''
                   下面开始计算精确率
                   '''
                   tempprediction = []  
                   for nam in name_assemble:
                       tempprediction.append(sess.run(vgg.names['prob_%s'%(nam)] , feed_dict={images: imgs, train_mode: False}))
                    
                   acc = get_accuracy(tempprediction,trainedlabel_assemble)
                   
                   print(str(iiter)+'th iteration,loss is:%f,\tright?:%s,\tlabel is:%d,lr is:%f'%(loss_value,acc,labs,lr))
                   fp.write(str(iiter)+'th iteration,loss is:%f,\tright?:%s,\tlabel is:%d,lr is:%f\n'%(loss_value,acc,labs,lr))
           
           if val % val_interval == 0:
               
               print('\nbegin the '+str(val1)+'th validation...')
               fp.write('\nbegin the '+str(val1)+'th validation...\n')
               val1+=1
               
               acc_val1 = 0
               
               for idx_val in range(the_num_of_val_images):
                   
                   imgs_val,labs_val = sess.run([image_batches_val,label_batches_val])
                   labs_val = labs_val[0]
                   layernum_assemble, name_assemble ,trainedlabel_assemble = find_out_layernum_trainlabel_name_from_all_list(layer_node,labs_val)
                   
                   tempprediction = []  
                   for nam in name_assemble:
                       tempprediction.append(sess.run(vgg.names['prob_%s'%(nam)] , feed_dict={images: imgs_val, train_mode: False}))
                   
                   if get_accuracy(tempprediction,trainedlabel_assemble) == 'yes':
                       acc_val1+=1
                   
               print('      the presion is %f'%(float(acc_val1)/float(the_num_of_val_images)))            
               fp.write('      the presion is %f\n'%(float(acc_val1)/float(the_num_of_val_images)))
               time.sleep(2)
               
           if save_count % save_interval == 0:
               
               print('\nsave npy model:'+'./trained_model/test-save_'+str(save_count)+'.npy'+'\n')
               
               fp.write('\nsave npy model:'+'./trained_model/test-save_'+str(save_count)+'.npy'+'\n')
               
               vgg.save_npy(sess, save_model+'/test-save_'+str(save_count)+'.npy')
               
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    fp.close()
    sess.close()
