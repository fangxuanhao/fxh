import  tensorflow as tf
import  random
import  matplotlib.pyplot as plt
tf.set_random_seed(888)
#载入mnist数据
from tensorflow.examples.tutorials.mnist import  input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
nb_classes=10

#MNIST数据图像28 28=784
X=tf.placeholder(tf.float32,[None,784])
#0-9数字识别的10个类
Y=tf.placeholder(tf.float32,[None,nb_classes])
W=tf.Variable(tf.random_normal([784,nb_classes]))
b=tf.Variable(tf.random_normal([nb_classes]))

#模型
hypothesis=tf.nn.softmax(tf.matul(X,W)+b)

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#测试模型
is_correct=tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
#parameters
training_epochs=15 #参量
batch_size=100

TB_SUMMARY_DIR = 'D:\shenduxuexirz\logs'
sess = tf.Session()
# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batchc=int(mnist.train.num_examples / batch_size)
        for i in range(total_batchc):
            c, _=sess.run([cost,optimizer],feed_dict={
                X:batch_size,Y:batch_size
            })
            avg_cost+=c/total_batchc
        print('Epoch:','%04d'%(epoch+1),
              'cost=','{:.9f}'.format(avg_cost)
              )
    print("学习完了")

    #使用测试集测试模型

