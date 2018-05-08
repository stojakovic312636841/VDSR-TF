# -*- coding: utf-8 -*

import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
from MODEL import model, inference
from PSNR import psnr
from TEST import test_VDSR
from tensorflow.python.client import device_lib
#from MODEL_FACTORIZED import model_factorized

DATA_PATH = "./data/train/"
TEST_DATA_PATH = "./data/test/"
TOWER_NAME = 'tower'
IMG_SIZE = (41, 41)
BATCH_SIZE = 1024
BASE_LR = 0.01
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 1

#USE_QUEUE_LOADING = False#True

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path


def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	print len(l)
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	print len(l)
	train_list = []
	for f in l:
		if os.path.exists(f): #f->gt f[:-4]->input
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
	return train_list


def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num


def get_image_batch(train_list,offset,batch_size):

	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []	

	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)

	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])

	#print(len(input_list))
	return input_list, gt_list, np.array(cbcr_list)


def loss(logits, labels):
	"""
	Add L2Loss to all the trainable variables.
  
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from get_image_batch(). 3-D tensor
              of shape [batch_size]
  
    Returns:
      Loss tensor of type float.
	"""
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(logits, labels)))
	tf.add_to_collection('losses', loss)
	
	return tf.add_n(tf.get_collection('losses'), name='total_loss')



def tower_loss(scope,images,labels):
	"""
	Calculate the total loss on a single tower running the VDSR model.

	Args:
  	scope: unique prefix string identifying the VDSR tower, e.g. 'tower_0'

	Returns:
   	Tensor of shape [] containing the total loss for a batch of data
	"""	
	'''
	print('step:%d'%(step))
	offset = step*BATCH_SIZE
	# Input images and labels.
	images, labels, _ = get_image_batch(train_list, offset, BATCH_SIZE)			
	'''
	#
	start_time = time.time()

	# Build inference Graph.
	logits = inference(images)

	# Build the portion of the Graph calculating the losses. Note that we will
	# assemble the total_loss using a custom function below.	
	loss_ = loss(logits, labels)

	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')
	
	total_loss = total_loss/BATCH_SIZE
	
	#print("%0.4f s"%(time.time()-start_time))
	
	return logits, total_loss



def average_gradients(tower_grads):
	"""Calculate average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been 
       averaged across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


'''
def average_gradients(tower_grads): 
     
    print('towerGrads:')  
    idx = 0  
    for grads in tower_grads:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple  
        print('grads---tower_%d' % idx)  
        for g_var in grads:  
            print(g_var)  
            print('\t%s\n\t%s' % (g_var[0].op.name, g_var[1].op.name))  
#             print('\t%s: %s'%(g_var[0].op.name,g_var[1].op.name))  
        idx += 1  
    
    if(len(tower_grads) == 1):  
        return tower_grads[0]  
    avgGrad_var_s = []  
    for grad_var_s in zip(*tower_grads):  
        grads = []  
        v = None  
        for g, v_ in grad_var_s:  
            g = tf.expand_dims(g, 0)  
            grads.append(g)  
            v = v_  
        all_g = tf.concat(grads,0)  
        avg_g = tf.reduce_mean(all_g, 0, keep_dims=False)  
        avgGrad_var_s.append((avg_g, v));  
    return avgGrad_var_s  
'''

def generate_towers(train_list, NUM_GPU=2, batch_size=BATCH_SIZE):
	

	train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
	train_gt  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

	train_input_tensor = tf.split(train_input, NUM_GPU, 0)
	train_gt_tensor	   = tf.split(train_gt, NUM_GPU, 0)

	# Calculate the gradients for each model tower.
	tower_grads = []

	#learning_rate = tf.train.exponential_decay(BASE_LR, 1, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
	#opt = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True,use_locking=True)
	learning_rate = BASE_LR
	opt = tf.train.AdamOptimizer(learning_rate) 

	with tf.variable_scope(tf.get_variable_scope()):	
		for i in xrange(NUM_GPU):				
			with tf.device('/gpu:%d' % i):
				with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
					print('%s_%d' % (TOWER_NAME, i))
					input_sub = train_input_tensor[i] 					
					target_sub = train_gt_tensor[i]  

	
                    # Calculate the loss for one tower of the VDSR model.
                    # This function constructs the entire VDSR model but
                    # shares the variables across all towers.
					y, t_loss = tower_loss(scope,input_sub,target_sub)

					#print("loss:%s" % t_loss.device)
 
					# Reuse variables for the next tower.
					tf.get_variable_scope().reuse_variables()

					# Retain the summaries from the final tower.
					summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,scope)
					
					# Calculate the gradients for the batch of data on this
					# VDSR tower.
					grads = opt.compute_gradients(t_loss)
					
					# Keep track of the gradients across all towers.
					tower_grads.append(grads)					

	# We must calculate the mean of each gradient. Note that this is the
	# synchronization point across all towers.
	grads = average_gradients(tower_grads)		

	train_op = opt.apply_gradients(grads)	
	
	return train_input, train_gt,learning_rate ,train_op, t_loss, y, summaries


def train():
	#get train_list
	train_list = get_train_list(DATA_PATH)
	shuffle(train_list)

	with tf.Graph().as_default(), tf.device('/cpu:0'):

		# Create a variable to count the number of train() calls. This equals
    	# the number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0),trainable=False)

		train_input, train_gt, learning_rate, train_op, t_loss, train_output, summaries = generate_towers(train_list = train_list, NUM_GPU=check_available_gpus())
		
		# Create a saver.
		saver = tf.train.Saver(tf.global_variables(),sharded=True)

		# Build the summary operation from the last tower summaries.
		#summary_op = tf.summary.merge(summaries)

		# The op for initializing the variables.
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

		# Start running operations on the Graph. allow_soft_placement must be
		# set to True to build towers on GPU, as some of the ops do not have GPU
		# implementations.
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
		sess.run(init_op)

		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		#summary_writer = tf.summary.FileWriter(DATA_PATH, sess.graph)

	try:		
		for epoch in xrange(0, MAX_EPOCH):
			for step in range(len(train_list)//BATCH_SIZE):
				offset = step*BATCH_SIZE
				images_batch, labels_batch, _ = get_image_batch(train_list, offset, BATCH_SIZE)					

				# Run one step of the model.  The return values are
				# the activations from the `train_op` (which is
				# discarded) and the `loss` op.  To inspect the values
				# of your ops or variables, you may include them in
				# the list passed to sess.run() and the value tensors
				# will be returned in the tuple from the call.
				
				feed_dict = {train_input:images_batch, train_gt:labels_batch}				

				start_time = time.time()

				_, loss_value ,debug= sess.run([train_op, t_loss, summaries], feed_dict=feed_dict)				
				print(debug)
				duration = time.time() - start_time
				print("use time:%.3f___step:%d-->loss:%.4f"%(duration,step,loss_value))
				
				assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

				# Print an overview fairly often.
				'''
				if step % 100 == 0:
					num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = duration / FLAGS.num_gpus
					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
					print(format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))
				if FLAGS.tb_logging:
					if step % 10 == 0:
						summary_str = sess.run(summary_op)
						summary_writer.add_summary(summary_str, step)
				'''
				# Save the model checkpoint periodically.
				'''
				if step % 1000 == 0 or (step + 1) == FLAGS.num_epochs * FLAGS.batch_size:
					checkpoint_path = os.path.join(FLAGS.train_dir,'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step)
				'''
				del images_batch, labels_batch

					
	except tf.errors.OutOfRangeError:
		print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()






def main(argv=None):  # pylint: disable=unused-argument
    start_time = time.time()
    train()
    duration = time.time() - start_time
    print('Total Duration (%.3f sec)' % duration)



if __name__ == '__main__':
    tf.app.run()
