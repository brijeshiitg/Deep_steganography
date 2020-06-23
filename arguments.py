import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('learning_rate', 0.0001, """learning rate.""")
tf.app.flags.DEFINE_float('beta', 0.5, """beta.""") #change beta according to requirements 
tf.app.flags.DEFINE_integer('epoch', 200, """epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 20, """Batch size.""")
tf.app.flags.DEFINE_integer('input_size', 50000, """Train + Val size.""")
tf.app.flags.DEFINE_integer('train_size', 49000, """Train size.""")
tf.app.flags.DEFINE_integer('num_channels', 3, """Number of the input's channels.""")
tf.app.flags.DEFINE_integer('image_size', 256, """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 256, """Size of the labels.""")
tf.app.flags.DEFINE_string("input_path", "../New_imagenet_train/cover/", "The path to input images") 
tf.app.flags.DEFINE_string("message_path", "../New_imagenet_train/secret/", "The path to secret images") 
tf.app.flags.DEFINE_string("checkpoint_path", "./checkpoints/", "The path to save the model")
tf.app.flags.DEFINE_string("log_path", "./logdir", "The path to save logs")
