from models import vgg_model
import datasets
import convnet_builder
import tensorflow as tf

phase_train = False
data_format = 'NHWC'
data_type = tf.float32
image_shape = [16, 224, 224, 3]
labels_shape = [16]
nclass = 1000
use_tf_layers = False

#with tf.device(0):
images = tf.truncated_normal(
    image_shape,
    dtype=data_type,
    mean=127,
    stddev=60,
    name='synthetic_images')

images = tf.contrib.framework.local_variable(
    images, name='gpu_cached_images')

labels = tf.random_uniform(
    labels_shape,
    minval=0,
    maxval=nclass - 1,
    dtype=tf.int32,
    name='synthetic_labels')

network = convnet_builder.ConvNetBuilder(
          images, 3, phase_train, use_tf_layers,
          data_format, data_type, data_type)

model = vgg_model.Vgg16Model()
model.add_inference(network)
logits = network.affine(nclass, activation='linear')

init_op = tf.initialize_all_variables()
init_local_op = tf.initialize_local_variables()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(init_local_op)

    sess.run(logits)