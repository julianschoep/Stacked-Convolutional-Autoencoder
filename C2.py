#from inspect import getsourcefile
#from os.path import abspath
import numpy as np

from PIL import Image
#from _overlapped import NULL
import os
import timeit

from numpy import float32

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm_layer
#parameters
 
# def save2(img1, img2,  step):
#     save_path = str(step)+".jpg"
#     fig = plt.figure()
#            
#             
#                 
#     img1ax = fig.add_subplot(2,1,1)
#     img1ax.set_title("original")
#     plt.imshow(img1)
#     img2ax = fig.add_subplot(2,1,2)
#     img2ax.set_title("reconstruction")
#     #plt.imshow(img2)
#     fig.savefig(save_path)
#     plt.close()
# # # # 

# def show(img1):
#     fig = plt.figure()
#     plt.imshow(img1)
#     plt.show()
# def show8(img1,img2, img3, img4, img5, img6, img7, img8):
#     fig = plt.figure()
#     fig.add_subplot(4,2,1).set_title("1")
#     plt.imshow(img1)
#     fig.add_subplot(4,2,2).set_title("2")
#     plt.imshow(img2)
#     fig.add_subplot(4,2,3).set_title("3")
#     plt.imshow(img3)
#     fig.add_subplot(4,2,4).set_title("4")
#     plt.imshow(img4)
#     fig.add_subplot(4,2,5).set_title("5")
#     plt.imshow(img5)
#     fig.add_subplot(4,2,6).set_title("6")
#     plt.imshow(img6)
#     fig.add_subplot(4,2,7).set_title("7")
#     plt.imshow(img7)
#     fig.add_subplot(4,2,8).set_title("8")
#     plt.imshow(img8)
#     plt.show()    
# def show10(img1,img2, img3, img4, img5, img6, img7, img8,img9,img10):
#     fig = plt.figure()
#     fig.add_subplot(5,2,1).set_title("1")
#     plt.imshow(img1, cmap='Greys')
#     fig.add_subplot(5,2,2).set_title("2")
#     plt.imshow(img2, cmap='Greys')
#     fig.add_subplot(5,2,3).set_title("3")
#     plt.imshow(img3, cmap='Greys')
#     fig.add_subplot(5,2,4).set_title("4")
#     plt.imshow(img4, cmap='Greys')
#     fig.add_subplot(5,2,5).set_title("5")
#     plt.imshow(img5, cmap='Greys')
#     fig.add_subplot(5,2,6).set_title("6")
#     plt.imshow(img6, cmap='Greys')
#     fig.add_subplot(5,2,7).set_title("7")
#     plt.imshow(img7, cmap='Greys')
#     fig.add_subplot(5,2,8).set_title("8")
#     plt.imshow(img8, cmap='Greys')
#     fig.add_subplot(5,2,9).set_title("9")
#     plt.imshow(img9, cmap='Greys')
#     fig.add_subplot(5,2,10).set_title("10")
#     plt.imshow(img10, cmap='Greys')
#     plt.show()
# def savebw10(img1, img2, img3, img4, img5, img6,img7,img8, img9, img10, name):
#     print("showing image...")
#     fig = plt.figure()
#   
#     #print(save_path )
#       
#     fig.add_subplot(5,2,1)
#     plt.imshow(img1, cmap='Greys')
#     fig.add_subplot(5,2,2)
#     plt.imshow(img2, cmap='Greys')
#     fig.add_subplot(5,2,3)
#     plt.imshow(img3, cmap='Greys')
#     fig.add_subplot(5,2,4)
#     plt.imshow(img4, cmap='Greys')
#     fig.add_subplot(5,2,5)
#     plt.imshow(img5, cmap='Greys')
#     fig.add_subplot(5,2,6)
#     plt.imshow(img6, cmap='Greys')
#     fig.add_subplot(5,2,7)
#     plt.imshow(img7, cmap='Greys')
#     fig.add_subplot(5,2,8)
#     plt.imshow(img8, cmap='Greys')
#     fig.add_subplot(5,2,9)
#     plt.imshow(img9, cmap='Greys')
#     fig.add_subplot(5,2,10)
#     plt.imshow(img10, cmap='Greys')
#     fig.savefig(name)
#     plt.close()
# # #Network input & parameters
# # 
# def savebw(img, name):
#     print("saving filter...")
#     fig = plt.figure()
#     plt.title(name)
#     #print(save_path )
#        
#     plt.imshow(img, cmap='Greys')
#     #plt.show()
#     fig.savefig(name+".jpg")
#    
#     plt.close()
#     print("saving done")
def unpool(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat(3,[x, tf.zeros_like(x)])
    out = tf.concat(2,[out, tf.zeros_like(out)])
    
    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret
def unpool2(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)],3)
    out = tf.concat([out, tf.zeros_like(out)],2)
    
    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret

def pool(value, name='pool'):
    return  tf.nn.avg_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
    
def pool10(value):
    return tf.nn.max_pool(value, ksize=[1,10,10,1], strides=[1,10,10,1], padding="SAME")
def orderArray(A):
    values=[]
    dtype=[('number', float),("element", object)]
    result=[]
    for e in A:
        
        elementString=str(e)
        number = float(e[11:13])
        values.append((number,elementString))
    orderedA=np.array(values, dtype)
    orderedA =  np.sort(orderedA,order="number")
    for element in orderedA:
        result.append(element[1])
    return result
def multDataline(x):
    line = x.split()
    result=""
    for value in line:
        result+=str(float(value)*1)
        result+=" "
    return result
def orderOutput():
    print('ordering output')
    file_names = open("latentRepresentations.txt", 'r')
    puck4 =[]
    puck3 =[]
    puck2=[]
    puck1=[]
    for line in file_names:
        if(line[5]=="1"):
            puck1.append(line)
        if(line[5]=="2"):
            puck2.append(line)
        if(line[5]=="3"):
            puck3.append(line)
        if(line[5]=="4"):
            puck4.append(line)
    puck1 = orderArray(puck1)
    puck2 = orderArray(puck2)
    puck3 = orderArray(puck3)
    puck4 = orderArray(puck4)
    #for i in orderArray(puck4):
        #print(i.split(":")[0])
    f = open("latentRepresentationsOrdered.txt","w")
    counter = 1
    for word in puck1:
        f.write(str(counter))
        f.write(" ")
        dataline = word.split(":")[1]
        dataline = multDataline(dataline)
        f.write(dataline)
        f.write("\n")
        counter+=1
    for word in puck2:
        f.write(str(counter))
        f.write(" ")
        dataline = word.split(":")[1]
        dataline = multDataline(dataline)
        f.write(dataline)
        f.write("\n")
        counter+=1
    for word in puck3:
        f.write(str(counter))
        f.write(" ")
        dataline = word.split(":")[1]
        dataline = multDataline(dataline)
        f.write(dataline)
        f.write("\n")
        counter+=1
    for word in puck4:
        f.write(str(counter))
        f.write(" ")
        dataline = word.split(":")[1]
        dataline = multDataline(dataline)
        f.write(dataline)
        f.write("\n")
        counter+=1
            
              
      
    f.close()
##############################
##TRAINING TIMES#############
##06JUL: 14:06 - ...

#pooling implementation by someone else


#LAYER NAMING: First letter: E: encoder or D: decoder
#              Second letter: C: convolutional Layer, P: pooling layer, R: reshape layer, F: fully connected layer


def multArray(array, factor):
    arrayLen = len(array)
    newArrayLen = arrayLen*factor
    val = array[0]
    for i in range(newArrayLen):
        array.append(val)
    return array

def lrelu(x, name, leak=0.2):
    
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return tf.add(f1 * x, f2*abs(x), name= name)  
    
def product(array):
    p = 1
    for i in range(len(array)):
        p = p * array[i]
    return p

def getImage(fileQueue):
    image_reader = tf.WholeFileReader()
    name, image_file = image_reader.read(fileQueue)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    image_decoded = tf.reshape(image_decoded, [120,160,3])
    image = image_decoded/255
    #image = tf.image.per_image_standardization(image)
    
    return image, name
def getImageSmall(fileQueue):
    image_reader = tf.WholeFileReader()
    name, image_file = image_reader.read(fileQueue)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    image_decoded = tf.reshape(image_decoded, [12,160,3])
    image = image_decoded/255
#input handler

def centroid(data_list):
    #list of tensors of which to compute centroid each tensor must have shape [im_num, height, width, depth]
    #returns a tesnor of shape [1,h,w,d]
    result = tf.concat(data_list,0)
    result = tf.reduce_mean(result,0)
    return result

def euclid_dist(A,B):
    #A and B are assumed to be tenors. Have to be of same shape
    dimsA = A.get_shape().as_list()
    dimsB = B.get_shape().as_list()
    if(len(dimsA)==4):
        A = A[0,:,:,:]
    if(len(dimsB)==4):
        B = B[0,:,:,:]
    result = 0
    depth = dimsA[-1]; width = dimsA[-2]; height=dimsA[-3]
    sum = 0
    for h in range(height):
        for w in range(width):
            for d in range(depth):
                sum+=(A[h,w,d]-B[h,w,d])**2
                
    dist = sum**0.5
    return dist

def batch_norm2(X, beta, scale,is_training):
    epsilon = 1e-3 
    X_mean, X_var = tf.nn.moments(X,[0])
    X_norm =  tf.nn.batch_normalization(X,X_mean,X_var,beta,scale,epsilon)
    return X_norm
def batch_norm(X, phase):
    #epsilon = 1e-3 
    #X_mean, X_var = tf.nn.moments(X,[0])
    X_norm =  batch_norm_layer(X, center=True, scale=True, is_training=phase)
    return X_norm

# def batch_norm_layer(x,is_training2):
#     return batch_norm(x, decay=0.999, center=True, scale=True,
#     updates_collections=False,
#     is_training=is_training2)
def maximum(tensor):
    shape = tensor.shape
    num_dims = len(shape)
    num_elem = 1
    max = 0.0
    for i in range(num_dims):
        num_elem *= shape[i]
    list = tf.reshape(tensor,[num_elem])   
    for j in range(num_elem):
        max = tf.maximum(max, list[j])
    return max
     
def depth2d_transpose(input, filter, output_shape,strides):
    
    tensor_list =[]
    num_out_ch = output_shape[-1] #10
    num_in_ch=input.get_shape().as_list()[3] #30
    num_unique_ch = int(num_in_ch/num_out_ch)
    #in is 30x40x30
    #out should be [batch_size,60,80,10]
    # filter  is [5,5, 10, 3]
    #
    # (ch*num_unique_ch)+num_unique_ch
    # 
    # 
    for ch in range(num_out_ch): #will be 10 unique channels of length 
        #print(ch)
        unique_list = []
        for ch_u in range(num_unique_ch):
            
            ch_index = (ch*num_unique_ch)+ch_u
            
            inPart = tf.expand_dims(input[:,:,:,ch_index],3)
            
            unique_list.append(inPart)  
        inChannel = tf.concat(3, unique_list) #this will be channel block of depth 3. Input to regular transp. convolution
        
        channelFilter = tf.expand_dims(filter[:,:,ch,:], 2) #this will have depth 3 as well. Shape [5,5,1,3]
        print(channelFilter)
        partOutputShape = [output_shape[0],output_shape[1],output_shape[2],1] #will have dim [batch,width,height,1]
        print(partOutputShape)
        singleTensor = tf.nn.conv2d_transpose(inChannel,channelFilter , partOutputShape , strides, "SAME") 
        tensor_list.append(singleTensor)
        #print("this is one tensor..")
        #print(singleTensor.shape)
    return tf.concat(3, tensor_list)
   
def concat(val,ax):
    return tf.concat(ax,val)
#get file queue working... https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue
initstddev = 1

dropout = 0.85
num_epochs=200
num_batches=31
batch_size_static = 88

filenames = tf.matching_files("./train_data/*.jpg")
#test on puck4 pls

print(filenames)

filename_queue = tf.train.string_input_producer(filenames, shuffle=False, name="DEMON_FROM_HELL")
images, names = getImage(filename_queue)
#batch, labels= tf.train.shuffle_batch([images, names], batch_size=batch_size_static, capacity=1000 + 3 * batch_size_static, allow_smaller_final_batch=True, min_after_dequeue=1000)
batch, labels= tf.train.batch([images, names], batch_size=batch_size_static, capacity=1000 + 3 * batch_size_static, allow_smaller_final_batch=True)

save_path = 'TrainedModel'

model_name = 'CAE5'#abspath(getsourcefile(lambda:0)).split('\\')[6].split('.')[0] #gives the filename without extension
print(os.path.exists('TrainedModel'))
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_full = os.path.join(save_path, model_name)
restore_file_name = save_path+"/"+model_name+".meta"
print(save_path_full)
print("Training on dataset train_data")
if not os.path.exists(restore_file_name):
    
    print("Model did not exist yet.")
    print("Generating new...")
    new_model = True
    global_step = tf.Variable(0, trainable=False, name="global_step")
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       100000, 0.96, staircase=True)
    is_training = tf.placeholder(tf.bool, name="is_training")
    time_start = timeit.default_timer()
    run_time = tf.Variable(0, name="run_time", trainable=False, dtype=tf.float32)
    X = tf.placeholder(tf.float32, [None,120,160,3], name="X")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    batch_size = tf.placeholder(tf.int32, name="batch_size")
    layer_counter = tf.Variable(0, name='layer_counter', trainable=False, dtype=tf.int32) #keeps track which layer is being pre-trained for resuming purposes
    condition = tf.Variable(0,name="condition", trainable=False, dtype=tf.int32)
    #structure numbers
    filterI1= [2,2,3];filterH1= [5,5,1];filterH2= [5,5,1];filterH3= [3,3,3];filterH4= [3,3,1]
    depthIn = 3; depth1 = 10;depth2 = 30;depth3 = 10; depth4=10; depth5=10
   
    
    weights = {
        'WCE1' : tf.Variable(tf.truncated_normal([2, 2, 3, 10], stddev=(2/(product(filterI1)))), name='WCE1'),  # 7x7 patch, 3 input channel, K output channels
        'WCE2' : tf.Variable(tf.truncated_normal([5,5, 10, 3], stddev=(2/(product(filterH1)))), name='WCE2'),  # 7x7 patch, 3 input channel, K output channels
        'WCD2' : tf.Variable(tf.truncated_normal([5,5, 1, 1], stddev=(2/(product(filterH2)))), name='WCE3'),  # 7x7 patch, 3 input channel, K output channels
        'WCD1' : tf.Variable(tf.truncated_normal([3,3, 3, 1], stddev=(2/(product(filterH3)))), name='WCE4'),  # 7x7 patch, 3 input channel, K output channels
        #'WFE1' : tf.Variable(tf.truncated_normal([heO, heI], stddev=(2/(heO+heI))), name='WFE1'),
        #'WFD2' : tf.Variable(tf.truncated_normal([hdI, hdO], stddev=(2/(heI+heO))), name='WFD2')

        }
    tiedWeights={
        'WCE1' : weights["WCE1"],
        'WCE2' : weights["WCE2"],

       #'WFE1' : tf.Variable(tf.truncated_normal([heO, heI], stddev=(2/(heO+heI))), name='WFE1'),
        #'WFD2' : tf.Variable(tf.truncated_normal([hdI, hdO], stddev=(2/(heI+heO))), name='WFD2')

        'WCD3' : weights["WCE2"],
        'WCD4' : weights["WCE1"],
        #'WCD5' : weights["WCE1"],
        
        }
    biases = {
        'BCE1' : tf.Variable(tf.zeros([depth1]), name='BCE1'), #Bias of 5 w1
        'BCE1out': tf.Variable(tf.zeros([depthIn]), name='BCE1o'),
        'BCE2' : tf.Variable(tf.zeros([depth2]), name='BCE2'),
        'BCE2out': tf.Variable(tf.zeros([depth1]), name='BCE2o'),
        'BCE3' : tf.Variable(tf.zeros([depth3]), name='BCE3'),
        'BCE3out': tf.Variable(tf.zeros([depth2]), name='BCE3o'),
        'BCE4' : tf.Variable(tf.zeros([depth4]), name='BCE4'),
        'BCE4out': tf.Variable(tf.zeros([depth3]), name='BCE4o'),
        #'BCE5' : tf.Variable(tf.zeros([depth5]), name='BCE5'),
        #'BCE5out' : tf.Variable(tf.zeros([depth4]), name='BCE5o'),
        #'BFE1' : tf.Variable(tf.zeros([heI]), name='BFE1'),

        #'BFD2' : tf.Variable(tf.zeros([hdO]), name='BFD2'),
        'BCD1' : tf.Variable(tf.zeros([1]), name='BCD1'),
        'BCD2' : tf.Variable(tf.zeros([3]), name='BCD2'),
        'BCD3' : tf.Variable(tf.zeros([depth1]), name='BCD3'),
        'BCD4' : tf.Variable(tf.zeros([depthIn]), name='BCD4'),##Bias of 5 w2
        #'BCD5' : tf.Variable(tf.zeros([depthIn]), name='BCD5'),
        }
    ###############################
    #120*160*3 INPUT          yy<-inpput dimensions
    #120*160*8 EC1            xx<-output dimensions Layer Name
    # 60*80*10 EC2   
    # 30*40*12 EC3
    # 30*40*14 EC4
    # 15*20*14 Maxpool EC3
    #  50     LatentRep
    # 15*20*14 Unpool Latent
    # 30*40*14 DC3
    # 30*40*12 DC3
    # 60*80*10 DC2
    #120*160*8 DC1
    #120*160*3 Output
    #
    #START OF ENCODER
    #input --> conv.layer with 7x7 filter --> Y1, output should be 60x80x5
    #for batch norm.
    stride= 1
    X_norm = batch_norm(X, is_training)
    #was 0.8
    EC1pre = batch_norm((tf.nn.conv2d(tf.nn.dropout(X_norm, 1),tiedWeights['WCE1'], strides=[1,2,2,1], padding='SAME') + biases['BCE1']), is_training)
    EC1 = tf.nn.elu(EC1pre, name="EC1") #introduce some noise to the input data.
    
    EC1pred = tf.nn.elu(batch_norm((tf.nn.conv2d_transpose(tf.nn.dropout(EC1, 1), tiedWeights['WCE1'], [batch_size,120,160,3], [1,2,2,1]) + biases['BCE1out']), is_training), name="EC1pred")
    
    #first conv out is 60x80x5, second conv output should be 30x40x5
    #third
    #EC1 returns a [batch,60x80x10] tensor.
    #We want the filter with first off highest var, and then lowest mean, so the highest 
    
    
    #here ch_i is the channel with the best representative power
    
    EC2 = tf.nn.elu(batch_norm((tf.nn.depthwise_conv2d(tf.nn.dropout(EC1, 1),tiedWeights['WCE2'], strides=[1,2,2,1], padding='SAME') + biases['BCE2']), is_training), name="EC2")
    EC2pred = tf.nn.elu(batch_norm((depth2d_transpose(tf.nn.dropout(EC2, 1), tiedWeights['WCE2'], [batch_size,60,80,10], [1,2,2,1]) + biases['BCE2out']), is_training), name="EC2pred")
    #depth2d_transpose(input, filter, output_shape,strides):
    best_filter = tf.expand_dims(EC2[:,:,:,condition],3, name="best_filter") 
    #best_filter has shape 30x40x1
    #condition has some value between 0 and 29. The condition can be computed from weight: (in_channel*multiplier)+mult_counter = condition
    # we know multiplier is three.So condition%3 is the mult_channel index. int(condition/3)
    mult_index = condition%3
    ch_index = tf.cast(tf.floor(condition/3),dtype=tf.int32)
    
    pooled_best_filter = tf.add(pool(best_filter),0,name="latent") #returns a [batch_size,15,20,1] tensor
    weight_DC2 = tf.expand_dims(tiedWeights['WCE2'][:,:,ch_index, mult_index],2)
    weight_DC2 = tf.expand_dims(weight_DC2,3)
    #print(weight_DC2)
    #WCD2
    #you put in 30x40x1 tensor with filter 5x5x1x1, you deconvolute it.
    DC2pre = (tf.nn.conv2d_transpose(best_filter,weight_DC2, output_shape=[batch_size,60,80,1], strides=[1,2,2,1], padding='SAME') + biases['BCD2'])
    DC2pre2 = tf.expand_dims(DC2pre[:,:,:,0],3)
    DC2 =  tf.nn.elu(batch_norm((DC2pre2), is_training), name="DC1") #output is 
    
    weight_DC1 = tf.expand_dims(tiedWeights["WCE1"][:,:,:,ch_index],3) #has shape [3,3,3,1]
    #print(weight_DC1)
    DC1pre = tf.reshape((tf.nn.conv2d_transpose(tf.nn.dropout(DC2, 1), weight_DC1, [batch_size,120,160,3], strides=[1,2,2,1], padding='SAME') + biases['BCD1']),[batch_size,120,160,3])
    DC1 =  tf.nn.elu(batch_norm(DC1pre, is_training), name="DC2")
    #DC4u = unpool(DC4)
    #DC5 = lrelu(batch_norm(tf.add(tf.nn.conv2d_transpose(tf.nn.dropout(DC4u, 1), tiedWeights['WCD5'],[batch_size,120,160,3], strides=[1,stride,stride,1], padding='SAME'),biases['BCD5']),is_training), name="prediction")
    #apply dropout
    #DP2 = tf.nn.dropout(DC1, dropout)
    
    
    #layer preds
    
    #EC4pred = lrelu(batch_norm((depth2d_transpose(tf.nn.dropout(EC4, 1), tiedWeights['WCE4'], [batch_size,15,20,1], [1,stride,stride,1]) + biases['BCE4out']),is_training), name="EC4pred")
    
    #https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py just look at this.
    #We compare the network output with the input (what an autoencoder does).
    costC1 = tf.multiply(.5, tf.reduce_sum(tf.pow(X_norm - EC1pred, 2)), name="costC1")
    costC2 = tf.multiply(.5, tf.reduce_sum(tf.pow(EC1 - EC2pred, 2)), name="costC2")
    #costC3 = tf.multiply(.5, tf.reduce_sum(tf.pow( EC2p - EC3pred, 2)), name="costC3")
    #costC4 = tf.multiply(.5, tf.reduce_sum(tf.pow(EC3p - DC1, 2)), name="costC4")
    #costC5 = tf.multiply(.5, tf.reduce_sum(tf.pow(EC4p - DC1, 2)), name="costC5")
    cost = tf.multiply(.5, tf.reduce_sum(tf.pow(X_norm - DC1, 2)), name="cost")
    
    #perpixelDiv = tf.reduce_mean(abs(X-pred))#/(160*120*3*batch_size_static)
    num_epochsCounter = 0
    saver = tf.train.Saver()
else:
    print("Resuming training...")
    new_model = False
    time_start = timeit.default_timer()
    saver = tf.train.import_meta_graph(restore_file_name)
    
    graph = tf.get_default_graph()
    
    init = tf.global_variables_initializer(), tf.local_variables_initializer()
    
    X = graph.get_tensor_by_name("X:0")
    is_training=graph.get_tensor_by_name("is_training:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    batch_size = graph.get_tensor_by_name("batch_size:0")
    cost = graph.get_tensor_by_name("cost:0")
    costC1 = graph.get_tensor_by_name("costC1:0")
    costC2 = graph.get_tensor_by_name("costC2:0")
    #costC3 = graph.get_tensor_by_name("costC3:0")
    #costC4 = graph.get_tensor_by_name("costC4:0")
    #costC5 = graph.get_tensor_by_name("costC5:0")
   # DC1 = graph.get_tensor_by_name("DC1:0")
   # DC2 = graph.get_tensor_by_name("DC2:0")
    #DC3 = graph.get_tensor_by_name("DC3:0")
    #DC4 = graph.get_tensor_by_name("DC4:0")
   # DC5 = graph.get_tensor_by_name("prediction:0")
    
    EC1 = graph.get_tensor_by_name("EC1:0")
    EC2 = graph.get_tensor_by_name("EC2:0")
    DC2 = graph.get_tensor_by_name("DC1:0")
    DC1 = graph.get_tensor_by_name("DC2:0")
    
    best_filter = graph.get_tensor_by_name("best_filter:0")
    #EC3 = graph.get_tensor_by_name("EC3:0")
    #EC4 = graph.get_tensor_by_name("EC4:0")
 #   EC5 = graph.get_tensor_by_name("EC5:0")
    WCE1 = graph.get_tensor_by_name("WCE1:0")
    WCE2 = graph.get_tensor_by_name("WCE2:0")
    WCE3 = graph.get_tensor_by_name("WCE3:0")
    WCE4 = graph.get_tensor_by_name("WCE4:0")
    BCD3 = graph.get_tensor_by_name("BCD3:0")
    BCD4 = graph.get_tensor_by_name("BCD4:0")
   # WCE5 = graph.get_tensor_by_name("WCE4_1:0")
    #latent = graph.get_tensor_by_name("latent:0")
#     j = 0
#     for node in graph.get_operations():
#         j+=1
#         if(j<200):
#             print(node.name)
    latest_checkpoint = tf.train.latest_checkpoint('TrainedModel\\temp\\')
    run_time = [v for v in tf.global_variables() if v.name=="run_time:0"][0]
    layer_counter = [v for v in tf.global_variables() if v.name=="layer_counter:0"][0]
    condition = [v for v in tf.global_variables() if v.name=="condition:0"][0]
sess = tf.Session()

if(new_model):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizerC1 = tf.train.AdamOptimizer(learning_rate).minimize(costC1, global_step=global_step)
        optimizerC2 = tf.train.AdamOptimizer(learning_rate).minimize(costC2, global_step=global_step)
        #optimizerC3 = tf.train.AdamOptimizer(learning_rate).minimize(costC3, global_step=global_step)
        #optimizerC4 = tf.train.AdamOptimizer(learning_rate).minimize(costC4, global_step=global_step)
        #optimizerC5 = tf.train.AdamOptimizer(learning_rate).minimize(costC5, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    init = tf.global_variables_initializer(), tf.local_variables_initializer()
    sess.run(init)
    tf.add_to_collection("optimizerC1", optimizerC1)
    tf.add_to_collection("optimizerC2", optimizerC2)
    #tf.add_to_collection("optimizerC3", optimizerC3)
    #tf.add_to_collection("optimizerC4", optimizerC4)
   # tf.add_to_collection("optimizerC5", optimizerC5)
    tf.add_to_collection("optimizer", optimizer)

else:
    
    sess.run(init)
    optimizerC1 = tf.get_collection("optimizerC1")[0]
    optimizerC2 = tf.get_collection("optimizerC2")[0]
    #optimizerC3 = tf.get_collection("optimizerC3")[0]
    #optimizerC4 = tf.get_collection("optimizerC4")[0]
    #optimizerC5 = tf.get_collection("optimizerC5")[0]
    optimizer = tf.get_collection("optimizer")[0]
    saver.restore(sess,save_path_full)
    run_time_carry = run_time.eval(session=sess)
   
layerGlob =0
    
step =0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord = coord)
num_encoding_layers = 3
layer1cost = 0
layer2cost = 0
totalcost = 0
#layer5cost = "not trained yet"
layer = sess.run(layer_counter)
ch_i = sess.run(condition)

best_avg = -1.0
while layer <= (num_encoding_layers): #num_encoding_layers
    min_cost = 1000000000
    layerGlob=layer
    
    
    cost_low = False
    cost_hist=[]
    prev_cost = 100000
    layer+=1
    
    if(layer ==3):
        X_data, name_data = sess.run([batch, labels])
        E2 = sess.run(EC2, {X:X_data, keep_prob: 1.0, batch_size:batch_size_static, is_training: True})
        
        
        
        for ch in range(30): #for every output channel
            print("Channel is "+str(ch))
            mean_list = []
            var_list = []
            channel = E2[:,:,:,ch]
            for im in range(batch_size_static): #for every datapoint in that channel
                #print("ch:"+str(ch)+"im:"+str(im))
                im_mean = np.mean(channel[im,:,:])
                im_var = np.var(channel[im,:,:])
                mean_list.append(im_mean)
                var_list.append(im_var)
            
            
            var_avg = np.mean(var_list)
            
            
            print(var_avg)
            print(var_avg>best_avg)
            if(var_avg>best_avg):
                print("best channel updated")
                ch_i = ch
                best_avg = var_avg
                
            print("best channel is" + str(ch_i))
            
        sess.run(condition.assign(ch_i))
        if(layer==3):
            num_epochs = 3500
        
            
    i_epoch=0
    
    while not cost_low:#num_epochs
        try:    
            i_epoch+=1
            print(i_epoch)
            for i_batch in range(num_batches): #num_batches
                step+=1
                #print(str(step)+"|", sep=' ', end='', flush=True)
                #print(step)
                
                X_data, name_data = sess.run([batch, labels])
                


            #best filter is 27                                                                                      #[batch_size, h, w, d ]
                E1 = sess.run(EC1, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:False}) #[batch_size, 60,80,10]
                E2 = sess.run(EC2, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:False}) #[batch_size, 30,40,30]
                D2 = sess.run(DC2, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:False}) #[batch_size, 60,80 1]
                D1 = sess.run(DC1, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:False}) #[batch_size, 120,160,3]
                cnt = 0
                import matplotlib.pyplot as plt
                   
                Weights1 = sess.run(WCE1)
                print(Weights1.shape)
                fig = plt.figure()
                
                for i in range(10): #per image
                    fig.add_subplot(2,5, i+1).set_title(str(i+1))
                    plt.imshow(Weights1[:,:,:,i])
                    plt.axis('off')
                fig.savefig("weights one")
                print("done")
                
#                 print("best filter is no:")
#                 print(ch_i)
#                 print("saving latent representations")
#                 f = open("latentRepresentations.txt","w")
#                 latentRep = sess.run(tf.reshape(best_pooled,[batch_size_static,6*8]))
#                 #print(name_data)
#                 for i in range(batch_size_static):
#                         #print(i)
#                         f.write(name_data[i].decode("utf-8").split('\\')[2])
#                         f.write(": ")
#                         thisLatent = str(latentRep[i])[1:-1]
#                         for word in thisLatent.split():
#                             f.write(word)
#                             f.write(" ")
#                         f.write("\n")
#                             
#   
#                 f.close()
#                 orderOutput()
#                 print("Done.")
#                 fig=plt.figure()
#                 plt.imshow(E2[3,:,:,29],cmap='gray')
#                 plt.show()
#                 fig = plt.figure()
#                 for im in range(batch_size_static): #per image
#                      
#                      
#                     fig.add_subplot(8,11,im+1)
#                     plt.imshow(best_pooled[im,:,:,0],cmap='gray')
#                     plt.axis('off')
#                      
#                 fig.savefig("poolstrain")
#                 plt.close()
#                 fig = plt.figure()
#                 for im in range(batch_size_static): #per image
#                      
#                      
#                     fig.add_subplot(8,11,im+1)
#                     plt.imshow(best[im,:,:,0],cmap='gray')
#                     plt.axis('off')
#                      
#                 fig.savefig("non-pooledtrain")
#                 plt.close()
#                 print("done")
#                 print("pools saved")
#                 for im in range(batch_size_static): #per image
#                     sitnum= name_data[im].decode("utf-8").split("\\")[2][3]
#                     fig = plt.figure()
#                           
#                     for i1 in range(10):
#                         fig.add_subplot(5,10,i1+1)
#                         plt.imshow(E1[im,:,:,i1],cmap='gray')
#                         plt.axis('off')
#                     for i2 in range(30):
#                         fig.add_subplot(5,10,i1+i2+2)
#                         plt.imshow(E2[im,:,:,i2],cmap='gray')
#                         plt.axis('off')
#                     fig.add_subplot(5,10,41)
#                     plt.imshow(best[im,:,:,0],cmap='gray')
#                     plt.axis('off')
#                     fig.add_subplot(5,10,42)
#                     plt.imshow(D2[im,:,:,0],cmap='gray')
#                     plt.axis('off')
#                     fig.add_subplot(5,10,43)
#                     plt.imshow(best_pooled[im,:,:,0],cmap='gray')
#                     plt.axis('off')
#                     fig.add_subplot(5,10,44)
#                     plt.imshow(D1[im,:,:,:])
#                     plt.axis('off')
#                     fig.add_subplot(5,10,45)
#                     plt.imshow(X_data[im,:,:,:])
#                     plt.axis('off')
#                     fig.savefig("sit"+str(sitnum)+"im"+str(im))
#                     plt.close()
                best = sess.run(best_filter, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:False}) #[batch_size, 30,40,1]
                best_pooled = sess.run(tf.nn.avg_pool(best,[1,5,5,1],[1,5,5,1],padding="SAME"))
#                 sit1=[];sit2=[];sit3=[];sit4=[];sit5=[]
#                 for j in range(batch_size_static):
#                     imname = name_data[j].decode("utf-8").split('\\')[2]
#                     im = best_pooled[j,:,:,:]
#                     im = tf.expand_dims(im,0)
#                     sitnum = int(imname[3])
#                     if(sitnum==1): sit1.append(im)
#                     elif(sitnum==2):sit2.append(im)
#                     elif(sitnum==3):sit3.append(im)
#                     elif(sitnum==4):sit4.append(im)
#                     elif(sitnum==5):sit5.append(im)
#                  
#                 len1 = len(sit1);len2 = len(sit2);len3 = len(sit3);len4 = len(sit4);len5 = len(sit5);
#                 sit1T = tf.concat(sit1,0);sit2T = tf.concat(sit2,0);sit3T = tf.concat(sit3,0);sit4T = tf.concat(sit4,0);sit5T = tf.concat(sit5,0)
#                 centroid1 = centroid(sit1);centroid2 = centroid(sit2);centroid3 = centroid(sit3);centroid4 = (centroid(sit4));centroid5 = centroid(sit5)
#                 print(centroid1.get_shape().as_list())
#                 print(sit1T.get_shape().as_list())
#                 def getDistFromCent(points, c, n):
#                     time_start = timeit.default_timer()
#                     #euclid_dist(A,B)
#                     sum = 0
#                     for i in range(n):
#                          
#                         im = points[i,:,:,:]
#                         sum+=euclid_dist(c,im)
#                         time_p = timeit.default_timer()
#                         print(str(i)+" of "+str(n)+"| ", sep='|', end='', flush=True)
#                     print("")
#                     print("")
#                     return sum/n
                 
#                 print("Sit 1 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit1T,centroid1, len1))))
#                 print("Sit 2 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit2T,centroid2, len2))))
#                 print("Sit 3 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit3T,centroid3, len3))))
#                 print("Sit 4 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit4T,centroid4, len4))))
#                 print("Sit 5 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit5T,centroid5, len5))))
#                 c_list = [centroid1, centroid2, centroid3, centroid4, centroid5]
#                 coded_list_other_avg = [0.3170996,0.297426737845,0.32827039063,0.354839861393,0.621734388173]
#                 coded_list_own_avg = [0.116071,0.173654, 0.170422, 0.191364,0.19821]
#                 
#                 rat_list_raw = [0.47364, 0.78729, 0.7119, 0.536, 0.3362]
#                 
#                 rat_list_coded = []
#                 for i in range(5):
#                     rat_list_coded.append(coded_list_own_avg[i]/coded_list_other_avg[i])
#                 
#                 N = 5
#                 ind = np.arange(N)
#                 
#                 width=0.35
#                 fig, ax= plt.subplots()
#                 rectsRaw=ax.bar(ind,rat_list_raw,width,color='gray')
#                 
#                 rectsCoded = ax.bar(ind+width, rat_list_coded, width, color="red")
#                 
#                 ax.set_ylabel("Distance ratio, R")
#                 ax.set_xticks(ind+width/2)
#                 ax.set_xticklabels(('Sit 1', 'Sit 2', 'Sit 3','Sit 4','Sit 5'))
#                 ax.legend((rectsRaw[0],rectsCoded[0]),('Raw data', 'Coded data'))
#                 plt.show()
#                 distance_num = 0
#                 for j in range(5): #for each situation
#                     c = c_list[j]
#                     sum =0
#                     for k in range(5):
#                         if(j!=k):
#                             sum+=(sess.run(euclid_dist(c_list[j],c_list[k])))
#                     avg = sum/4
#                     
#                     print("average distance centroid "+str(j)+"to other centroids is:"+str(avg))      
#                 print("Going into endless loop")
                while(True): 
                    a=1     
                 
                 
#                 #best = sess.run(best_filter, {X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:True})
#                 def trans_1d(X): #transform [im_num, h,w,d] to [im_num, l] for t_sne
#                     sh = X.get_shape().as_list()
#                     len = product(sh[1:])
#                     
#                     imnum = sh[0]
#                     l=[]
#                     for im in range(imnum):
#                         img = X[im, :,:,:]
#                         img = tf.reshape(img, [len])
#                         img =tf.expand_dims(img,0)
#                         l.append(img)
#                     return concat(l,0)
#                 
#                 
#                 sit1=[];sit2=[];sit3=[];sit4=[];sit5=[]
#                 for j in range(batch_size_static):
#                     imname = name_data[j].decode("utf-8").split('/')[2]
#                     im = data[j,:,:,:]
#                     im = tf.expand_dims(im,0)
#                     sitnum = int(imname[3])
#                     if(sitnum==1): sit1.append(im)
#                     elif(sitnum==2):sit2.append(im)
#                     elif(sitnum==3):sit3.append(im)
#                     elif(sitnum==4):sit4.append(im)
#                     elif(sitnum==5):sit5.append(im)
#                 
#                 len1 = len(sit1);len2 = len(sit2);len3 = len(sit3);len4 = len(sit4);len5 = len(sit5);
#                 sit1T = concat(sit1,0);sit2T = concat(sit2,0);sit3T = concat(sit3,0);sit4T = concat(sit4,0);sit5T = concat(sit5,0)
#                 centroid1 = centroid(sit1);centroid2 = centroid(sit2);centroid3 = centroid(sit3);centroid4 = (centroid(sit4));centroid5 = centroid(sit5)
#                 print(centroid1.get_shape().as_list())
#                 print(sit1T.get_shape().as_list())
#                 def getDistFromCent(points, c, n):
#                     time_start = timeit.default_timer()
#                     #euclid_dist(A,B)
#                     sum = 0
#                     for i in range(n):
#                         
#                         im = points[i,:,:,:]
#                         sum+=euclid_dist(c,im)
#                         time_p = timeit.default_timer()
#                         print(str(i)+" of "+str(n)+"|", sep='|', end='', flush=True)
#                     print("")
#                     print("")
#                     return sum/n
#                 
#                 print("Sit 1 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit1T,centroid1, len1))))
#                 print("Sit 2 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit2T,centroid2, len2))))
#                 print("Sit 3 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit3T,centroid3, len3))))
#                 print("Sit 4 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit4T,centroid4, len4))))
#                 print("Sit 5 avg dist from centre is: "+ str(sess.run(getDistFromCent(sit5T,centroid5, len5))))
#                 c_list = [centroid1, centroid2, centroid3, centroid4, centroid5]
#                 distance_num = 0
# #                 for j in range(5):
# #                     for k in range(5):
# #                         if not (k>=j):
# #                             distance_num+=1
# #                             print("Distance " +str(distance_num)+ "is:")
# #                             print("first centroid: "+str(j+1)+", second centroid: "+str(k+1))
# #                             print(sess.run(euclid_dist(c_list[j],c_list[k])))
#                             
#                 
#                 print("going into endless loop...")
#                 while(True): 
#                     a=1    
                
                    
                
                
                
                if not new_model:
                    print("condition is:" +str(condition))
                  
                time_step = timeit.default_timer()
                if(new_model):
                    run_time_counter = time_step-time_start
                else:
                    run_time_counter = time_step-time_start + run_time_carry
               
                
                
                #sess.run(Y_pred, {X_input:X_data, Y_true: X_data})
                if(layer ==1):
                    sess.run(optimizerC1, {X:X_data, keep_prob: 1, batch_size: batch_size_static, is_training:True})
                elif(layer==2):
                    sess.run(optimizerC2, {X:X_data, keep_prob: 1, batch_size: batch_size_static, is_training:True})
                    
                elif(layer==3):
                    sess.run(optimizer, {X:X_data, keep_prob: 1, batch_size: batch_size_static, is_training:True})
                
                if(step%10 == 0): #every 10 steps
                    print("")
                    if(layer ==1):
                        layername = "first convolutional layer"
                        train_cost = sess.run(costC1,{X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:True})
                        layer1cost = train_cost
                    elif(layer==2):
                        layername = "second convolutional layer"
                        train_cost = sess.run(costC2,{X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:True})
                        layer2cost = train_cost
                    elif(layer==3):
                        layername = "total network"
                        train_cost = sess.run(cost,{X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:True})
                        totalcost = train_cost
                        print("best filter is: "+str(ch_i))
                    elif(layer>3):
                        train_cost = sess.run(cost,{X:X_data, keep_prob:1, batch_size: batch_size_static, is_training:True})
                        print("best filter is: "+str(ch_i))
                    if(train_cost < min_cost):
                        min_cost = train_cost
                    
                    print("First cost: "+("{:,}".format(layer1cost))+", second cost: "+("{:,}".format(layer2cost))+", total cost: "+("{:,}".format(totalcost)))
                    
                    print("Lowest achieved cost on this layer: "+"{:,}".format(min_cost))
                    print("Resuming training the "+layername)
                    cost_div = train_cost - prev_cost
                    cost_hist.append(abs(cost_div))
                    prev_cost=train_cost
                    if(train_cost<200):
                        print("doneTraining")
                        cost_low = True
                        
                    print("Cost is "+ "{:,}".format(train_cost) + "and best filter no. is: "+ str(ch_i))
                    #img_sofar = sess.run(pred,{X:X_data, keep_prob:1.0, batch_size: batch_size_static})
                    
                    
                    mins = int(run_time_counter/60);secs = int(run_time_counter%60)
                    step_name = "epoch"+str(i_epoch)+"batch"+str(i_batch)+"time"+str(mins)+"m"+str(secs)+"s"
                    print(step_name)
                    #save(img_sofar[53], step_name)
                    if(i_epoch>num_epochs):
                        if(train_cost-1000<min_cost): #can abort training if train_cost is a max of 1000 higher than lowest achieved cost on this layer
                            cost_low = True
                
                   
        except KeyboardInterrupt:
            print("Layer skipped.")
            print("Saving snapshot...")
            saver.save(sess, save_path_full)
            print("Snapshot saved.")
            break
        
#saving filters

#print("Time run:"+str(run_time_counter))
#sess.run(run_time.assign(run_time_counter))
print("ran until layer:"+str(layerGlob))
sess.run(layer_counter.assign(layerGlob))
print("best filter no is:"+str(ch_i))

print("Saving trained weights to {0}", save_path_full)
saver.save(sess, save_path_full) 
print("Model saved. Joining threads...")   

coord.request_stop()
coord.join(threads)  
print("Done.")
# print("saving latent representations")
# f = open("latentRepresentations.txt","w")
# latentRep = sess.run(EF2, {X:X_data, keep_prob:1.0, batch_size: batch_size_static})
#  
# for i in range(batch_size_static):
#         f.write(str(name_data[i])[15:])
#         f.write(":")
#         thisLatent = str(latentRep[i])[1:-1]
#         for word in thisLatent.split():
#             f.write(word)
#             f.write(",")
#         f.write("\n")
#          
#  
# f.close()
# print("Done.")
    

  

#SETUP network performance graph
#check tied weights vs non-tied weights

# 
#  from sklearn.manifold import TSNE
#                
#                 from mpl_toolkits.mplot3d import Axes3D #@UnresolvedImport
#                 
#                 
#                 model = TSNE(n_components=2, learning_rate=100, init='pca',n_iter=8000)
#                 np.set_printoptions(suppress=True)
#                 sit1_pts = model.fit_transform(total);sit2_pts = model.fit_transform(total);sit3_pts = model.fit_transform(s3);sit4_pts = model.fit_transform(s4);sit5_pts = model.fit_transform(s5)
#     
#                 
#                 x1=[];y1=[];x2=[];y2=[]
#                 for p in sit1_pts:
#                     x1.append(p[0])
#                     y1.append(p[1])
#                 for p in sit2_pts:
#                     x2.append(p[0])
#                     y2.append(p[1])
#                 fig = plt.figure()
#                 
#                 ax = fig.add_subplot(111)
#                 #ax.scatter(x1, y1, c="yellow")
#                 ax.scatter(x2, y2, c="orange")
# 
#                 plt.show()




