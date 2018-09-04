from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import random
import pandas as pd
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rot90(m, k=1, axis=2):
    """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m

def rotate_oasis():
    for x in range (1, 317):
        img_data = np.load("data2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy")
        img_data = rot90(img_data, 3, 0)
        img_data = rot90(img_data, 1, 2)
        print("img" + str(x) + " rotated: ")
        np.save("data2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy", img_data)

def shape_oasis():
    for x in range (1, 317):
        img_data = np.load("data2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy")
        npad = ((5, 5), (0, 0), (0, 0))
        img_data = np.pad(img_data, pad_width=npad, mode='constant', constant_values=0)

        startz = 65//2-(55//2)
        img_data = img_data[0:65,0:65, startz:startz+55]
        print("img" + str(x) + " shaped: ")
        np.save("data2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy", img_data)

def calculate_mean():

    only_img = []
    
    for x in range (1, 1201):
        img_data = np.load("shuffled2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy")
        
        print("img" + str(x) + " appended: ")
        only_img.append(img_data)

    mean_img = np.mean(only_img, dtype=np.int, axis=0)
    np.save('mean_img2.npy', mean_img)


def shuffle():
    index = [i for j in (range(1,317), range(907, 2073)) for i in j]
    #index = [x for x in range(1,2073)]
    original = index[:]
    
    for n,i in enumerate(index):
        if i>906:
            index[n]=index[n]-590
    temp = index[:]
    random.shuffle(index)
    
    print(original)
    print("------------------------")
    print(temp)
    print("------------------------")
    print(index)
    csvfile = "index_reference2.csv"

    
    with open(csvfile, "w", newline='') as output:
        
        writer = csv.writer(output)
        for val in index:
            writer.writerows([[val]])

    for n,i in enumerate(original):
        img_data = np.load("data2\\" + str(i) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy")
        new_index = index[n]
        np.save("shuffled2\\" + str(new_index) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy", img_data)
        print ("saved " + str(i))
    
def combine_preprocess (start, end):

    combined = []
    mean_image = np.load('mean_img2.npy')

    for x in range (start, end+1):
        img_data = np.load("shuffled2\\" + str(x) + "#(" + str(IMG_SIZE_PX)+ ", " + str(IMG_SIZE_PX) + ", " +str(SLICE_COUNT)+ ").npy")
        selected = labels_file[labels_file.shuffledID == x]
        
        for index, row in selected.iterrows():
            print (str(row['ID']) + " 's age is " + str(row['AGE_AT_SCAN']))
            age = row['AGE_AT_SCAN']
            
        label = np.array([age])

        img_data -= mean_image
        
        combined.append([img_data,label])

    np.save('combined2\\' + 'batch({},{})-{}-{}-{}.npy'.format(start,end,IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), combined)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                               size of window     movement of window
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               'W_fc':tf.Variable(tf.random_normal([258944,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 258944])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):

    
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs): 
            epoch_loss = 0
            
            for current_batch in range (0, train_batch):
                batch_data = np.load('/home/lvruyi/combined/' + 'batch({},{})-{}-{}-{}.npy'.format(current_batch*batch_size+1,current_batch*batch_size+batch_size,IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT))

                for data in batch_data:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
            for current_validation in range (train_batch, train_batch+validation_batch):
                validation_data = np.load('/home/lvruyi/combined/' + 'batch({},{})-{}-{}-{}.npy'.format(current_validation*batch_size+1,current_validation*batch_size+batch_size,IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT))
                evaluated_accuracy += accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})
                print('Accuracy:',evaluated_accuracy/batch_val)


        saver.save(sess, 'fyp_model')

IMG_SIZE_PX = 65
SLICE_COUNT = 55
n_classes = 1
train_batch = 75
batch_size = 16
validation_batch = 18
labels_file = pd.read_csv('FYP_Phenotypic2.csv')


x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


#rotate_oasis ()

#shape_oasis()

#shuffle()

#calculate_mean()


for batch in range (0, train_batch+validation_batch-1):
    combine_preprocess(batch*batch_size+1, batch*batch_size+batch_size)

combine_preprocess(1473, 1482)

                                     
#train_neural_network(x)
