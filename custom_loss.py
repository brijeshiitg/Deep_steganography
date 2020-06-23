import tensorflow as tf
from arguments import *

def losses(secret_true, secret_pred, cover_true, cover_pred, beta=FLAGS.beta):

	with tf.variable_scope("losses"):
		beta = tf.constant(beta,name="beta")
		secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
		cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)
		final_loss = cover_mse + beta*secret_mse
		return final_loss , secret_mse , cover_mse 
