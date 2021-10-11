import tensorflow as tf
import numpy as np
import scipy.io as sio
import DatasetLoader


mini_batch_size = 64
width = 16  # width of the image in pixels
height = 16  # height of the image in pixels
flat = width * height  # number of pixels in one image
class_output = 3  # number of possible classifications for the problem


x = tf.placeholder(tf.float32, shape=[None, flat], name="input")
y_ = tf.placeholder(tf.float32, shape=[None, class_output], name="target")

# Start interactive session
sess = tf.InteractiveSession()


# Weight Initialization
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)


# Convolution and Pooling
def conv2d_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First Convolutional Layer
number_of_filters_layer1 = 64
x_image = tf.reshape(x, [-1, 16, 16, 1])
W_conv1 = weight_variable([3, 3, 1, number_of_filters_layer1], 'kernel_3_3_1')
b_conv1 = bias_variable([number_of_filters_layer1], 'bias_kernel_3_3_1')
convolve1 = conv2d_same(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(convolve1)
h_pool1 = max_pool_2x2(h_conv1)
layer1 = h_pool1

# Second Convolutional Layer
number_of_filters_layer2 = 128
W_conv2 = weight_variable([3, 3, number_of_filters_layer1, number_of_filters_layer2], 'kernel_3_3_128')
b_conv2 = bias_variable([number_of_filters_layer2], 'bias_kernel_3_3_128')
convolve2 = conv2d_same(layer1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(convolve2)
h_pool2 = max_pool_2x2(h_conv2)
layer2 = h_pool2

# Densely Connected Layer
dimension_of_input = 4
W_fc1 = weight_variable([dimension_of_input * dimension_of_input * number_of_filters_layer2, 1024], 'FCWL1')
b_fc1 = bias_variable([1024], 'FCBL1')
layer2_matrix = tf.reshape(layer2, [-1, dimension_of_input * dimension_of_input * number_of_filters_layer2])
matmul_fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(matmul_fc1)
layer4 = h_fc1

#  Dropout
keep_prob = tf.placeholder(tf.float32)
layer4_drop = tf.nn.dropout(layer4, keep_prob)

#  Readout Layer
W_fc2 = weight_variable([1024, 3], 'FCWL2')
b_fc2 = bias_variable([3], 'FCBL2')
matmul_fc2 = tf.matmul(layer4_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(matmul_fc2)
layer5 = y_conv



writer = tf.summary.FileWriter("CNN", sess.graph)
writer.close()

# Loss Function
with tf.name_scope('Loss_Function'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer5), reduction_indices=[1]))

# Loss function using L2 Regularization
regularizer = tf.nn.l2_loss(W_fc1)
lamda = .1

# Learning Algorithm
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cross_entropy + lamda*regularizer)

# Accuracy
correct_prediction = tf.equal(tf.argmax(layer5, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def calculate_confusion_matix(number_of_fold):
    batch = DatasetLoader.get_data_test_feed(fold=number_of_fold)
    image_list = batch[0]
    number_of_images = len(image_list)
    print("\nFeedForward Evaluation on {0} images in fold {1}".format(number_of_images, number_of_fold+1))

    input_data = image_list[1:number_of_images]
    x_image = tf.reshape(input_data, [-1, 16, 16, 1])

    # CNN FeedForward
    convolve1 = conv2d_same(x_image, W_conv1) + b_conv1
    h_conv1 = tf.nn.relu(convolve1)
    h_pool1 = max_pool_2x2(h_conv1)
    layer1 = h_pool1

    convolve2 = conv2d_same(layer1, W_conv2) + b_conv2
    h_conv2 = tf.nn.relu(convolve2)
    h_pool2 = max_pool_2x2(h_conv2)
    layer2 = h_pool2

    layer2_matrix = tf.reshape(layer2, [-1, 4 * 4 * 128])
    matmul_fc1 = tf.matmul(layer2_matrix, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(matmul_fc1)
    layer3 = h_fc1

    matmul_fc2 = tf.matmul(layer3, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(matmul_fc2)
    layer5 = y_conv

    result = layer5
    label = batch[1][1:number_of_images]

    predictions = tf.argmax(layer5, 1)
    real = tf.argmax(label, 1)

    myConfusionMatrix = tf.confusion_matrix(real, predictions=predictions)
    print(sess.run(myConfusionMatrix))

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

convnet_accuracy_fold = [[]]
convnet_accuracy_run = []

for fold in range(5):
    print("---------------- Fold {0} ----------------".format(fold+1))
    test_data = DatasetLoader.get_data_test(fold=fold)
    train_data = DatasetLoader.get_data_train_complete(fold=fold)
    for run in range(1):

        best_accuracy_run = 0
        sess.run(init)
        print("--- Run {0} ---".format(run+1))
        for epoch in range(25):

            print("--Epoch {0}".format(epoch+1))
            number_of_batches = int(len(DatasetLoader.get_data_train_complete(fold=fold)[0])/mini_batch_size)

            for indexOfBatch in range(number_of_batches):

                batch = DatasetLoader.get_data_train(indexOfBatch=indexOfBatch, fold=fold)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            test_accuracy = accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0})
            print("     Test Accuracy {0:.3%}".format(test_accuracy))

            convnet_accuracy_run.append(test_accuracy)
        train_accuracy = accuracy.eval(feed_dict={x: train_data[0], y_: train_data[1], keep_prob: 1.0})
        print("     Train Accuracy {0:.3%}".format(train_accuracy))
        calculate_confusion_matix(fold)

    convnet_accuracy_fold.append(convnet_accuracy_run)
    convnet_accuracy_run = []

print("\nFinished Training Neural Network\n")

print("REPORT:\n")
sio.savemat('convnet_accuracy_fold.mat', mdict={'convnet_accuracy_fold': convnet_accuracy_fold})
best_folds_accuracy = []
number_of_fold = 1
for accuracy_array in convnet_accuracy_fold:
    if len(accuracy_array) > 0:
        print("Best Accuracy in Fold {0}: {1:.3%}".format(number_of_fold, np.max(accuracy_array)))
        best_folds_accuracy.append(np.max(accuracy_array))
        number_of_fold += 1

print("Average Accuracy: {0:.3%}".format(np.average(best_folds_accuracy)))


sess.close()













