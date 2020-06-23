import os
import nips_model as model
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # select GPU device
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer\
('epoch_start',22, """From which epoch ?""")
tf.app.flags.DEFINE_integer\
('epoch_end',22, """To which epoch ?""")
tf.app.flags.DEFINE_string\
("cover_dir", "/home/brijesh/Rebuttal/KODAK_experiments/KODAK_selected/cover/", "The path of inputs")
tf.app.flags.DEFINE_string\
("secret_dir", "/home/brijesh/Rebuttal/KODAK_experiments/KODAK_selected/secret/", "The path of inputs")
tf.app.flags.DEFINE_string\
("stego_dir", "/home/brijesh/Rebuttal/KODAK_experiments/KODAK_selected/nips_embedded/", "The path of inputs")
tf.app.flags.DEFINE_string\
("reveal_dir", "/home/brijesh/Rebuttal/KODAK_experiments/KODAK_selected/nips_extracted/", "The path of inputs")
tf.app.flags.DEFINE_string\
("checkpoint_dir", "./checkpoints/", "The path of saving model checkpoints.")


for e in range(FLAGS.epoch_start,FLAGS.epoch_end+1):

	total_files=os.listdir(FLAGS.cover_dir)
	# print(total_files)
	for i in total_files:
		tf.reset_default_graph()
		img = cv2.imread(FLAGS.cover_dir+i)
		img= np.float32(img)/255.0
		# print(img.shape)
		h,w,c = img.shape

		msg = cv2.imread(FLAGS.secret_dir+i)
		msg= np.float32(msg)/255.0
		
		image = tf.placeholder(tf.float32, shape=(1, h, w, c))
		message = tf.placeholder(tf.float32, shape = (1,h,w,c))
		_, stego, _, ext_msg = model.Network(message, image)
		
		saver = tf.train.Saver()
	
		with tf.Session() as sess:
			saver.restore(\
				sess, FLAGS.checkpoint_dir+'model-epoch-' +str(e)) 
			#print ("load pre-trained model")
			_, stego, _, ext_msg  = sess.run([_, stego, _, ext_msg],\
			 feed_dict={message: msg.reshape((1,512,512,3)),\
			  image: img.reshape((1,512,512,3))})
			
			stego = stego.reshape((h,w,c))
			# print(stego)
			stego *= 255.0
			stego = np.uint8(stego)

			s_dir_name = FLAGS.stego_dir
			if not os.path.exists(s_dir_name):
				os.makedirs(s_dir_name)

			cv2.imwrite(s_dir_name+'/'+str(i),stego)
			print('stego image '+str(i)+' generated.')

			ext_msg = ext_msg.reshape((h,w,c))
			# print(ext_msg)
			ext_msg *= 255.0
			ext_msg = np.uint8(ext_msg)

			r_dir_name = FLAGS.reveal_dir
			if not os.path.exists(r_dir_name):
				os.makedirs(r_dir_name)

			cv2.imwrite(r_dir_name+'/'+str(i),ext_msg)
			print('revealed image '+str(i)+' generated.')

