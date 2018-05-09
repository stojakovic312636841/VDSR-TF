import tensorflow as tf
import numpy as np
import re

TOWER_NAME = 'tower'


def model(input_tensor):
	with tf.device("/gpu:0"):#/gpu:0
		weights = []
		tensor = None

		#conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
		conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

		for i in range(18):
			#conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
			conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
			conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
		
		#conv_w = tf.get_variable("conv_19_w", [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
		conv_w = tf.get_variable("conv_20_w", [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
		conv_b = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
		weights.append(conv_w)
		weights.append(conv_b)
		tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

		tensor = tf.add(tensor, input_tensor)
		return tensor, weights




def _variable_on_cpu(name, shape, initializer):
	"""
	Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var




def _variable_with_weight_decay(name, shape, stddev, wd):
	"""
	Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
	  name: name of the variable
	  shape: list of ints
	  stddev: standard deviation of a truncated Gaussian
	  wd: add L2Loss weight decay multiplied by this float. If None, weight
	      decay is not added for this Variable.

	Returns:
	  Variable Tensor
	"""
	
	dtype = tf.float32
	var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.    
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(images):
	"""Build the VDSR model.
	
    Args:
      images: Images returned from VDSR or inputs().
	
    Returns:
      Logits or predict
    """
	# We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training
    # runs. If we only ran this model on a single GPU, we could simplify this
    # function by replacing all instances of tf.get_variable()
    # with tf.Variable().

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

	input_tensor = tf.cast(images, tf.float32)
	weights = []
	#conv_00
	with tf.variable_scope('conv00' ,reuse = tf.AUTO_REUSE) as scope:
		tensor = None

		kernel = _variable_with_weight_decay('conv_00_w',shape=[3,3,1,64],stddev=np.sqrt(2.0/9),wd=0.0)
		biases = _variable_on_cpu('conv_00_b',[64],tf.constant_initializer(0.0))
		weights.append(kernel)
		weights.append(biases)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, kernel, strides=[1,1,1,1], padding='SAME'), biases))
		#_activation_summary(tensor)

	#conv_01 -- conv_19
	for i in range(18):
		with tf.variable_scope('conv%02d'%(i+1) ,reuse = tf.AUTO_REUSE) as scope:
			kernel = _variable_with_weight_decay('conv_%02d_w' % (i+1),shape=[3,3,64,64],stddev=np.sqrt(2.0/9/64),wd=0.0)
			biases = _variable_on_cpu('conv_%02d_b' % (i+1),[64],tf.constant_initializer(0.0))
			weights.append(kernel)
			weights.append(biases)
			tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, kernel, strides=[1,1,1,1], padding='SAME'), biases))
			#_activation_summary(tensor)
	

	#conv_20
	with tf.variable_scope('conv20' ,reuse = tf.AUTO_REUSE) as scope:		
		kernel = _variable_with_weight_decay('conv_20_w',shape=[3,3,64,1],stddev=np.sqrt(2.0/9/64),wd=0.0)
		biases = _variable_on_cpu('conv_20_b',[1],tf.constant_initializer(0.0))
		weights.append(kernel)
		weights.append(biases)
		tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, kernel, strides=[1,1,1,1], padding='SAME'), biases)
		#_activation_summary(tensor)


	tensor = tf.add(tensor, input_tensor)
	_activation_summary(tensor)

	return tensor, weights




