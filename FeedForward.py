import tensorflow as tf
import DatasetLoader as ds
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("E:/Workshop DeepLearning/S#06-Deeplearning/MNIST Dataset", one_hot=True)

#batch_X, batch_Y = mnist.train.next_batch(1)

sess = tf.InteractiveSession()

# First Convolution Layer Declaration
initial_wc1 = tf.truncated_normal([5, 5, 1, 128], stddev=0.1)
wc1 = tf.Variable(initial_wc1, name='kernel_5_5_1')
initial_bc1 = tf.constant(0.1, shape=[128])
bc1 = tf.Variable(initial_bc1, name='bias_kernel_5_5_1')

# Second Convolution Layer Declaration
initial_wc2 = tf.truncated_normal([5, 5, 128, 128], stddev=0.1)
wc2 = tf.Variable(initial_wc2, name='kernel_5_5_128')
initial_bc2 = tf.constant(0.1, shape=[128])
bc2 = tf.Variable(initial_bc2, name='bias_kernel_5_5_128')

# First Fully Connected Layer Declaration
initial_wl1 = tf.truncated_normal([4 * 4 * 128, 1024], stddev=0.1)
wl1 = tf.Variable(initial_wl1, name='FCWL1')
initial_bl1 = tf.constant(0.1, shape=[1024])
bl1 = tf.Variable(initial_bl1, name='FCBL1')

# Second Fully Connected Layer Declaration
initial_wl2 = tf.truncated_normal([1024, 3], stddev=0.1)
wl2 = tf.Variable(initial_wl2, name='FCWL2')
initial_bl2 = tf.constant(0.1, shape=[3])
bl2 = tf.Variable(initial_bl2, name='FCBL2')


def conv_2d(ix, iw):
    return tf.nn.conv2d(ix, iw, strides=[1, 1, 1, 1], padding='SAME', name="convolve")


def max_pool_2x2(ix):
    return tf.nn.max_pool(ix, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

number_of_images = 10


saver.restore(sess, "/Users/user/Desktop/Export/model.ckpt")

for index_of_image in range(number_of_images):

    input_data = ds.get_data_feed_forward(index_of_image=index_of_image)
    x_image = tf.reshape(input_data[0], [-1, 16, 16, 1])

    # CNN FeedForward
    convolve1 = conv_2d(x_image, wc1) + bc1
    h_conv1 = tf.nn.relu(convolve1)
    h_pool1 = max_pool_2x2(h_conv1)
    layer1 = h_pool1

    convolve2 = conv_2d(layer1, wc2) + bc2
    h_conv2 = tf.nn.relu(convolve2)
    h_pool2 = max_pool_2x2(h_conv2)
    layer2 = h_pool2

    layer2_matrix = tf.reshape(layer2, [-1, 4 * 4 * 128])
    matmul_fc1 = tf.matmul(layer2_matrix, wl1) + bl1
    h_fc1 = tf.nn.relu(matmul_fc1)
    layer3 = h_fc1

    matmul_fc2 = tf.matmul(layer3, wl2) + bl2
    y_conv = tf.nn.softmax(matmul_fc2)
    layer5 = y_conv
    result = sess.run(wc1)
    print(result)
    #print(input_data[1])
