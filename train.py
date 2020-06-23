import os
import numpy as np
import tensorflow as tf
from dataset import read_data
from arguments import *
from model import Network
from custom_loss import losses
import logging

os.environ['CUDA_VISIBLE_DEVICES']="0"
logging.basicConfig(filename='training.log',format='%(asctime)s %(message)s', level=logging.DEBUG)

if __name__ == '__main__':



	
	images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
	message = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))

	prep_output, hiding_output, noise_add, reveal_output = Network(message, images)

	loss_op,secret_loss_op,cover_loss_op = losses(message,reveal_output,images,hiding_output,beta=FLAGS.beta)
	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('secret_loss', secret_loss_op)
	tf.summary.scalar('cover_loss', cover_loss_op)


	lr_ = FLAGS.learning_rate
	lr = tf.placeholder(tf.float32, shape=[])

	optim = tf.train.AdamOptimizer(lr).minimize(loss_op)

	saver = tf.train.Saver(max_to_keep=None)

	epoch = int(FLAGS.epoch)

	with tf.Session() as sess:

		merged = tf.summary.merge_all()

		init = tf.initialize_all_variables()
		sess.run(init)
		file_writer = tf.summary.FileWriter(FLAGS.log_path, tf.get_default_graph())


		validation_image, validation_msg = read_data(FLAGS.input_path, FLAGS.train_size, FLAGS.input_size)

		if tf.train.get_checkpoint_state(FLAGS.checkpoint_path):
			ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
			saver.restore(sess, ckpt)
			start_point = int(ckpt.split('-')[-1])
			print("\nLoad success, continuing from checkpoint: epoch-%d"%(start_point+1))
		else:
			print("\nNo previous checkpoint. Training from scratch..")
			start_point = 0

		for j in range(start_point, epoch):
			if j+1 > epoch/3:  # reduce learning rate
				lr_ = FLAGS.learning_rate*0.1
			if j+1 > 2*epoch/3:
				lr_ = FLAGS.learning_rate*0.01

			Training_Loss = 0.
			Train_Secret_Loss =0.
			Train_Cover_Loss =0.

			start = 1  
			end = FLAGS.batch_size + 1 
			for i in range(int(FLAGS.train_size/FLAGS.batch_size)):
				train_data,train_msg= read_data(FLAGS.input_path,start, end) # data for training
				rand_index = np.random.permutation(FLAGS.batch_size)
				batch_data = train_data[rand_index,:,:,:]
				batch_msg = train_msg[rand_index,:,:,:]
				_,loss, secret_loss, cover_loss, summary = sess.run([optim,loss_op, secret_loss_op, cover_loss_op, merged], feed_dict={images:batch_data, message:batch_msg, lr: lr_})
				Training_Loss += loss  # training loss
				Train_Secret_Loss += secret_loss
				Train_Cover_Loss +=cover_loss

				print ('| epoch:%d | %d/%d batches finished | learning rate: %.4f | loss: %.6f | secret_loss: %.6f| cover_loss: %.6f|'\
				 % (j+1, i+1, int(FLAGS.train_size/FLAGS.batch_size), lr_, loss, secret_loss, cover_loss))
				logging.debug('| epoch:%d | %d/%d batches finished | learning rate: %.4f | loss: %.6f | secret_loss: %.6f| cover_loss: %.6f|'\
				% (j+1, i+1, int(FLAGS.train_size/FLAGS.batch_size), lr_, loss, secret_loss, cover_loss))
				start = end
				end += FLAGS.batch_size

				file_writer.add_summary(summary,j)

			# Training loss per batch on an average
			Training_Loss /=  int(FLAGS.train_size/FLAGS.batch_size)
			Train_Secret_Loss /=  int(FLAGS.train_size/FLAGS.batch_size)
			Train_Cover_Loss /=  int(FLAGS.train_size/FLAGS.batch_size)


			# Calculating validation loss
			Validation_Loss = 0.
			Val_Cov_Loss =0.
			Val_Sec_Loss =0.
			val_size = FLAGS.input_size - FLAGS.train_size
			for batch_num in range(int(val_size/FLAGS.batch_size)):
				rand_index = np.arange(FLAGS.batch_size)
				val_batch_data = validation_image[rand_index,:,:,:]
				val_msg = validation_msg[rand_index,:,:,:]
				# val_batch_label = validation_data[rand_index,:,:,:]
				val_Loss, val_sec_loss, val_cov_loss = sess.run([loss_op, secret_loss_op, cover_loss_op], feed_dict={images: val_batch_data, message:val_msg})  # validation loss
				Validation_Loss += val_Loss
				Val_Sec_Loss += val_sec_loss
				Val_Cov_Loss += val_cov_loss

			  # Validation loss per batch on an average
			Validation_Loss /= (val_size / FLAGS.batch_size)
			print ('| epoch:%d finished | Training_Loss: %.6f | Validation_Loss: %.6f | Val_Sec_Loss: %.6f | Val_Cov_Loss: %.6f |' \
				% (j+1, Training_Loss, Validation_Loss, Val_Sec_Loss, Val_Cov_Loss))
			logging.debug('| epoch:%d finished | Training_Loss: %.6f | Validation_Loss: %.6f | Val_Sec_Loss: %.6f | Val_Cov_Loss: %.6f |' \
				% (j+1, Training_Loss, Validation_Loss, Val_Sec_Loss, Val_Cov_Loss))


			model_name = 'model-epoch'   # save model
			save_path = os.path.join(FLAGS.checkpoint_path, model_name)
			saver.save(sess, save_path, global_step = j+1)
