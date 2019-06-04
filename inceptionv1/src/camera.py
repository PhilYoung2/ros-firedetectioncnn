#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os
import sys
import math

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

def construct_inceptionv1onfire (x,y):

	# Build network as per architecture in [Dunnings/Breckon, 2018]

	network = input_data(shape=[None, y, x, 3])

	conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', name = 'conv1_7_7_s2')

	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)

	conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', name='conv2_3_3')

	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')

	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

	# merge the inception_3a__
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

	#merge the inception_3b_*
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

	pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)
	pool5_7_7 = dropout(pool5_7_7, 0.4)
	loss = fully_connected(pool5_7_7, 2,activation='softmax')
	network = regression(loss, optimizer='momentum',
		         loss='categorical_crossentropy',
		         learning_rate=0.001)
	model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
		        max_checkpoints=1, tensorboard_verbose=2)

	return model


def callback(imgmsg):
	global s
	s = 0
	#k = 2
	#c = 0
	#if c % k == 0:
	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
	rows = 224
	cols = 224
	sp = img.shape
	height = sp[0]  # height(rows) of image
	width = sp[1]  # width(colums) of image
	# fps = 20
	# frame_time = round(1000/fps);
	# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
	small_frame = cv2.resize(img, (rows, cols), cv2.INTER_AREA)
	output = model.predict([small_frame])

	# label image based on prediction

	if round(output[0][0]) == 1:
		s = 1
		cv2.rectangle(img, (0,0), (width,height), (0,0,255), 50)
		cv2.putText(img,'FIRE',(int(width/16),int(height/4)),cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
	else:
		cv2.rectangle(img, (0,0), (width,height), (0,255,0), 50)
		cv2.putText(img,'CLEAR',(int(width/16),int		(height/4)),cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
	cv2.imshow("windowName", img);
	# rate = rospy.Rate(10)
	# while not rospy.is_shutdown():
	if s == 1:
		situation = "FIRE"
	else:
		situation = "CLEAR"
	#rospy.loginfo(s)
	pub.publish(situation)
	#	rate.sleep()
	# spin() simply keeps python from exiting until this node is stopped
	#rospy.spin()
	# print(s)	
	# print(height, width, frame_time)
	# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# fps = video.get(cv2.CAP_PROP_FPS)
	# cv2.imshow("listener", img)
	# cv2.imshow("smallframe", small_frame)
	#c = c + 1
	cv2.waitKey(3)

def publisherandsubscriber():
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	#s = 0
	global pub
	pub = rospy.Publisher('chatter', String, queue_size=10)
	rospy.init_node('inceptionv1', anonymous=True)
	rospy.Subscriber("/mynteye/left/image_raw", Image, callback)
	#rate = rospy.Rate(10)
	#while not rospy.is_shutdown():
		#if s == 1:
			#situation = "FIRE"
		#else:
			#situation = "CLEAR"
	#rospy.loginfo(s)
		#pub.publish(situation)
		#rate.sleep()
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()


if __name__ == '__main__':
     	bridge = CvBridge()
	model = construct_inceptionv1onfire (224, 224)
	print("Constructed InceptionV1-OnFire ...")

	model.load(os.path.join("catkin_ws/src/inceptionv1/src/models/InceptionV1-OnFire", "inceptiononv1onfire"),weights_only=True)
	print("Loaded CNN network weights ...")
	#windowName = "Live Fire Detection - InceptionV1-OnFire";
	#cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
	publisherandsubscriber()
