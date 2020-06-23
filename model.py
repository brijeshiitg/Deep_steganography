import tensorflow as tf
from tensorflow.python.layers.convolutional import conv2d


def prep_network(secret_tensor):

	with tf.variable_scope('prep_net'):

		conv11 = conv2d(inputs=secret_tensor,filters=50,kernel_size=3,padding='same',name="p11",activation=tf.nn.relu)
		conv12 = conv2d(inputs=conv11,filters=50,kernel_size=3,padding='same',name="p12",activation=tf.nn.relu)
		conv13 = conv2d(inputs=conv12,filters=50,kernel_size=3,padding='same',name="p13",activation=tf.nn.relu)
		conv14 = conv2d(inputs=conv13,filters=50,kernel_size=3,padding='same',name="p14",activation=tf.nn.relu)
		conv15 = conv2d(inputs=conv14,filters=50,kernel_size=3,padding='same',name="p1_out",activation=tf.nn.relu)

		conv21 = conv2d(inputs=secret_tensor,filters=50,kernel_size=4,padding='same',name="p21",activation=tf.nn.relu)
		conv22 = conv2d(inputs=conv21,filters=50,kernel_size=4,padding='same',name="p22",activation=tf.nn.relu)
		conv23 = conv2d(inputs=conv22,filters=50,kernel_size=4,padding='same',name="p23",activation=tf.nn.relu)
		conv24 = conv2d(inputs=conv23,filters=50,kernel_size=4,padding='same',name="p24",activation=tf.nn.relu)
		conv25 = conv2d(inputs=conv24,filters=50,kernel_size=4,padding='same',name="p2_out",activation=tf.nn.relu)

		conv31 = conv2d(inputs=secret_tensor,filters=50,kernel_size=5,padding='same',name="p31",activation=tf.nn.relu)
		conv32 = conv2d(inputs=conv31,filters=50,kernel_size=5,padding='same',name="p32",activation=tf.nn.relu)
		conv33 = conv2d(inputs=conv32,filters=50,kernel_size=5,padding='same',name="p33",activation=tf.nn.relu)
		conv34 = conv2d(inputs=conv33,filters=50,kernel_size=5,padding='same',name="p34",activation=tf.nn.relu)
		conv35 = conv2d(inputs=conv34,filters=50,kernel_size=5,padding='same',name="p3_out",activation=tf.nn.relu)
		cat = tf.concat([conv15, conv25,conv35], axis=3)
		conv4 = conv2d(inputs=cat, filters=3, kernel_size=1, padding='same', name="prep_output", activation=tf.nn.tanh)
		return conv4

def hiding_network(cover_tensor,prep_output):

	with tf.variable_scope('hide_net'):
		concat_input = tf.concat([cover_tensor,prep_output],axis=3,name='images_features_concat')

		conv11 = conv2d(inputs=concat_input,filters=50,kernel_size=3,padding='same',name="h11",activation=tf.nn.relu)
		conv12 = conv2d(inputs=conv11,filters=50,kernel_size=3,padding='same',name="h12",activation=tf.nn.relu)
		conv13 = conv2d(inputs=conv12,filters=50,kernel_size=3,padding='same',name="h13",activation=tf.nn.relu)
		conv14 = conv2d(inputs=conv13,filters=50,kernel_size=3,padding='same',name="h14",activation=tf.nn.relu)
		conv15 = conv2d(inputs=conv14,filters=50,kernel_size=3,padding='same',name="h1_out",activation=tf.nn.relu)

		conv21 = conv2d(inputs=concat_input,filters=50,kernel_size=4,padding='same',name="h21",activation=tf.nn.relu)
		conv22 = conv2d(inputs=conv21,filters=50,kernel_size=4,padding='same',name="h22",activation=tf.nn.relu)
		conv23 = conv2d(inputs=conv22,filters=50,kernel_size=4,padding='same',name="h23",activation=tf.nn.relu)
		conv24 = conv2d(inputs=conv23,filters=50,kernel_size=4,padding='same',name="h24",activation=tf.nn.relu)
		conv25 = conv2d(inputs=conv24,filters=50,kernel_size=4,padding='same',name="h2_out",activation=tf.nn.relu)

		conv31 = conv2d(inputs=concat_input,filters=50,kernel_size=5,padding='same',name="h31",activation=tf.nn.relu)
		conv32 = conv2d(inputs=conv31,filters=50,kernel_size=5,padding='same',name="h32",activation=tf.nn.relu)
		conv33 = conv2d(inputs=conv32,filters=50,kernel_size=5,padding='same',name="h33",activation=tf.nn.relu)
		conv34 = conv2d(inputs=conv33,filters=50,kernel_size=5,padding='same',name="h34",activation=tf.nn.relu)
		conv35 = conv2d(inputs=conv34,filters=50,kernel_size=5,padding='same',name="h4",activation=tf.nn.relu)
		cat = tf.concat([conv15, conv25,conv35], axis=3, name='hide_output' )
		conv4 = conv2d(inputs=cat, filters=3, kernel_size=1, padding='same', name='h_out', activation=tf.nn.tanh)
		return conv4

def reveal_network(container_tensor):

	with tf.variable_scope('reveal_net'):

		conv11 = conv2d(inputs=container_tensor,filters=50,kernel_size=3,padding='same',name="r11",activation=tf.nn.relu)
		conv12 = conv2d(inputs=conv11,filters=50,kernel_size=3,padding='same',name="r12",activation=tf.nn.relu)
		conv13 = conv2d(inputs=conv12,filters=50,kernel_size=3,padding='same',name="r13",activation=tf.nn.relu)
		conv14 = conv2d(inputs=conv13,filters=50,kernel_size=3,padding='same',name="r14",activation=tf.nn.relu)
		conv15 = conv2d(inputs=conv14,filters=50,kernel_size=3,padding='same',name="r1_out",activation=tf.nn.relu)

		conv21 = conv2d(inputs=container_tensor,filters=50,kernel_size=4,padding='same',name="r21",activation=tf.nn.relu)
		conv22 = conv2d(inputs=conv21,filters=50,kernel_size=4,padding='same',name="r22",activation=tf.nn.relu)
		conv23 = conv2d(inputs=conv22,filters=50,kernel_size=4,padding='same',name="r23",activation=tf.nn.relu)
		conv24 = conv2d(inputs=conv23,filters=50,kernel_size=4,padding='same',name="r24",activation=tf.nn.relu)
		conv25 = conv2d(inputs=conv24,filters=50,kernel_size=4,padding='same',name="r2_out",activation=tf.nn.relu)

		conv31 = conv2d(inputs=container_tensor,filters=50,kernel_size=5,padding='same',name="r31",activation=tf.nn.relu)
		conv32 = conv2d(inputs=conv31,filters=50,kernel_size=5,padding='same',name="r32",activation=tf.nn.relu)
		conv33 = conv2d(inputs=conv32,filters=50,kernel_size=5,padding='same',name="r33",activation=tf.nn.relu)
		conv34 = conv2d(inputs=conv33,filters=50,kernel_size=5,padding='same',name="r34",activation=tf.nn.relu)
		conv35 = conv2d(inputs=conv34,filters=50,kernel_size=5,padding='same',name="r3_out",activation=tf.nn.relu)
		cat = tf.concat([conv15, conv25,conv35], axis=3)
		conv4 = conv2d(inputs=cat, filters=3, kernel_size=1, padding='same', name='reveal_output', activation=tf.nn.tanh)
		return conv4

def noise_layer(tensor,std=.1):
	with tf.variable_scope("noise_layer"):
		return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32) 

def Network(secret_tensor,cover_tensor):

	prep_output = prep_network(secret_tensor)
	hiding_output = hiding_network(cover_tensor=cover_tensor,prep_output=prep_output)
	noise_add = noise_layer(hiding_output)
	reveal_output = reveal_network(noise_add)
	return prep_output, hiding_output, noise_add, reveal_output
