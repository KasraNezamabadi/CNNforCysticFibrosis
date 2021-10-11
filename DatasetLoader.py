import scipy.io as sio
import numpy as np


training_set_x = []
training_set_y = []
validate_set_x = []
validate_set_y = []
test_set_x = []
test_set_y = []


mini_batch_size = 64

#base_path = "/Users/user/Documents/CF/DataSource/Ready/16/RevisedFold5"
base_path = '/Users/kasra/Documents/PyCharmWorkplace/TensorCNN/Data/NewOrder/UnbBalanced test_rndNr/Central'

trainXPath_Fold1 = base_path + "/Fold1/Training_Set_X_Fold1.mat"
trainXPath_Fold2 = base_path + "/Fold2/Training_Set_X_Fold2.mat"
trainXPath_Fold3 = base_path + "/Fold3/Training_Set_X_Fold3.mat"
trainXPath_Fold4 = base_path + "/Fold4/Training_Set_X_Fold4.mat"
trainXPath_Fold5 = base_path + "/Fold5/Training_Set_X_Fold5.mat"

trainYPath_Fold1 = base_path + "/Fold1/Training_Set_Y_Fold1.mat"
trainYPath_Fold2 = base_path + "/Fold2/Training_Set_Y_Fold2.mat"
trainYPath_Fold3 = base_path + "/Fold3/Training_Set_Y_Fold3.mat"
trainYPath_Fold4 = base_path + "/Fold4/Training_Set_Y_Fold4.mat"
trainYPath_Fold5 = base_path + "/Fold5/Training_Set_Y_Fold5.mat"

testXPath_Fold1 = base_path + "/Fold1/Test_Set_X_Fold1.mat"
testXPath_Fold2 = base_path + "/Fold2/Test_Set_X_Fold2.mat"
testXPath_Fold3 = base_path + "/Fold3/Test_Set_X_Fold3.mat"
testXPath_Fold4 = base_path + "/Fold4/Test_Set_X_Fold4.mat"
testXPath_Fold5 = base_path + "/Fold5/Test_Set_X_Fold5.mat"

testYPath_Fold1 = base_path + "/Fold1/Test_Set_Y_Fold1.mat"
testYPath_Fold2 = base_path + "/Fold2/Test_Set_Y_Fold2.mat"
testYPath_Fold3 = base_path + "/Fold3/Test_Set_Y_Fold3.mat"
testYPath_Fold4 = base_path + "/Fold4/Test_Set_Y_Fold4.mat"
testYPath_Fold5 = base_path + "/Fold5/Test_Set_Y_Fold5.mat"


# trainXPath = base_path + "/Training_Set_X.mat"
# trainYPath = base_path + "/Training_Set_Y.mat"
# testXPath = base_path + "/Test_Set_X.mat"
# testYPath = base_path + "/Test_Set_Y.mat"



training_set_x.append(sio.loadmat(trainXPath_Fold1)['Training_Set_X_Fold1'])
training_set_x.append(sio.loadmat(trainXPath_Fold2)['Training_Set_X_Fold2'])
training_set_x.append(sio.loadmat(trainXPath_Fold3)['Training_Set_X_Fold3'])
training_set_x.append(sio.loadmat(trainXPath_Fold4)['Training_Set_X_Fold4'])
training_set_x.append(sio.loadmat(trainXPath_Fold5)['Training_Set_X_Fold5'])

training_set_y.append(sio.loadmat(trainYPath_Fold1)['Training_Set_Y_Fold1'])
training_set_y.append(sio.loadmat(trainYPath_Fold2)['Training_Set_Y_Fold2'])
training_set_y.append(sio.loadmat(trainYPath_Fold3)['Training_Set_Y_Fold3'])
training_set_y.append(sio.loadmat(trainYPath_Fold4)['Training_Set_Y_Fold4'])
training_set_y.append(sio.loadmat(trainYPath_Fold5)['Training_Set_Y_Fold5'])

test_set_x.append(sio.loadmat(testXPath_Fold1)['Test_Set_X_Fold1'])
test_set_x.append(sio.loadmat(testXPath_Fold2)['Test_Set_X_Fold2'])
test_set_x.append(sio.loadmat(testXPath_Fold3)['Test_Set_X_Fold3'])
test_set_x.append(sio.loadmat(testXPath_Fold4)['Test_Set_X_Fold4'])
test_set_x.append(sio.loadmat(testXPath_Fold5)['Test_Set_X_Fold5'])

test_set_y.append(sio.loadmat(testYPath_Fold1)['Test_Set_Y_Fold1'])
test_set_y.append(sio.loadmat(testYPath_Fold2)['Test_Set_Y_Fold2'])
test_set_y.append(sio.loadmat(testYPath_Fold3)['Test_Set_Y_Fold3'])
test_set_y.append(sio.loadmat(testYPath_Fold4)['Test_Set_Y_Fold4'])
test_set_y.append(sio.loadmat(testYPath_Fold5)['Test_Set_Y_Fold5'])


def get_data_train(indexOfBatch, fold):
    batch = []
    batch.append(training_set_x[fold][indexOfBatch*mini_batch_size : (indexOfBatch+1)*mini_batch_size])
    batch.append(training_set_y[fold][indexOfBatch*mini_batch_size : (indexOfBatch+1)*mini_batch_size])
    return batch


def get_data_train_complete(fold):
    batch = []
    batch.append(training_set_x[fold])
    batch.append(training_set_y[fold])
    return batch


def get_data_validation(fold):
    batch = []
    batch.append(validate_set_x[fold])
    batch.append(validate_set_y[fold])
    return batch


def get_data_test(fold):
    batch = []
    batch.append(test_set_x[fold])
    batch.append(test_set_y[fold])
    return batch

def get_data_test_feed(fold):
    batch = []
    batch.append(np.float32(test_set_x[fold]))
    batch.append(np.float32(test_set_y[fold]))
    return batch