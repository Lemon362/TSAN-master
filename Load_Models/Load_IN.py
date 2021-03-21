# -*- coding: utf-8 -*-
# 评估已保存的模型
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from scipy.io import savemat
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from keras import backend as K
from sklearn.metrics import precision_score

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ResNeXt_TwoStepAttention


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def model_DenseNet():
    # model_dense = ssrn_SS_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)
    model_dense = ResNeXt_TwoStepAttention.ResneXt_IN((1, img_rows, img_cols, img_channels), cardinality=8, classes=16)
    # model_dense = Resnet_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)
    # model_dense = ResneXt_SPC_2_IN.ResneXt_IN((1, img_rows, img_cols, img_channels), cardinality=8, classes=16)

    RMS = RMSprop(lr=0.0003)

    # Let's train the model using RMSprop

    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)

        # data_prop = K.constant([46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93])
        # data_prop = data_prop / 10249
        # data_prop = 961 / 10249
        #
        # shape_pred = y_pred.get_shape()
        # data_prop = K.reshape(data_prop, shape_pred)
        # data_prop = K.ones_like(y_pred) * data_prop

        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)  # K.ones_like(y_pred) / nb_classes

        return (1 - e) * loss1 + e * loss2

    model_dense.compile(loss=mycrossentropy, optimizer=RMS, metrics=['accuracy'])

    return model_dense


mat_data = sio.loadmat('D:/3D-TwoStep-Network/Datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('D:/3D-TwoStep-Network/Datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']

# mat_data = sio.loadmat('D:/3D-ResNeXt-master/Datasets/UP/PaviaU.mat')
# data_IN = mat_data['paviaU']
# # 标签数据
# mat_gt = sio.loadmat('D:/3D-ResNeXt-master/Datasets/UP/PaviaU_gt.mat')
# gt_IN = mat_gt['paviaU_gt']

print(data_IN.shape)

# new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
new_gt_IN = gt_IN

batch_size = 16
nb_classes = 16  # 16
nb_epoch = 60  # 400
img_rows, img_cols = 11, 11  # 27, 27
patience = 100

INPUT_DIMENSION_CONV = 200  # 200
INPUT_DIMENSION = 200  # 200

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 10249  # 10249
VAL_SIZE = 1025  # 1025

TRAIN_SIZE = 5128  # 1031 2055 3081 4106 5128 6153

TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.5
# TRAIN_NUM = 10
# TRAIN_SIZE = TRAIN_NUM * nb_classes
# TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
# VAL_SIZE = TRAIN_SIZE


img_channels = 200  # 200
PATCH_LENGTH = 5  # Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

data = preprocessing.scale(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_

# padded_data = cv2.copyMakeBorder(whole_data, PATCH_LENGTH, PATCH_LENGTH, PATCH_LENGTH, PATCH_LENGTH, cv2.BORDER_REFLECT)

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 1
CATEGORY = 16  # 16

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

KAPPA_3D_DenseNet = []
OA_3D_DenseNet = []
AA_3D_DenseNet = []
TRAINING_TIME_3D_DenseNet = []
TESTING_TIME_3D_DenseNet = []
ELEMENT_ACC_3D_DenseNet = np.zeros((ITER, CATEGORY))

PRECISION_3D_DenseNet = []
RECALL_3D_DenseNet = []
F1SCORE_3D_DenseNet = []

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    best_weights_DenseNet_path = 'D:/3D-TwoStep-Network/models/Indian_best_ResNeXt_TwoStepAttention_5_1_4_60_group6_' + str(
        index_iter + 1) + '.hdf5'
    # best_weights_DenseNet_path = 'D:/3D-ResNeXt-master/models/UP_best_3D_ResneXt_Dual_loss_5_1_4_60_' + str(index_iter + 1) + '.hdf5'
    # best_weights_DenseNet_path = 'D:/3D-ResNeXt-master/models/Indian_best_3D_SSRN_6_1_3_50_' + str(index_iter + 2) + '.hdf5'

    np.random.seed(seeds[index_iter])
    #    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    # SS Residual Network 4 with BN
    model_densenet = model_DenseNet()

    # 直接加载训练权重
    model_densenet.load_weights(best_weights_DenseNet_path)

    pred_test = model_densenet.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)

    counter = collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

    n = len(confusion_matrix)
    for i in range(len(confusion_matrix[0])):
        rowsum, colsum = sum(confusion_matrix[i]), sum(confusion_matrix[r][i] for r in range(n))
        try:
            precision = (confusion_matrix[i][i] / float(colsum))
            recall = (confusion_matrix[i][i] / float(rowsum))
            f1score = (2*precision*recall)/(precision + recall)
            print('precision: %s' % (precision),
                  'recall: %s' % (recall),
                  'f1score: %s' % (f1score))
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' % 0)

    mcm = metrics.multilabel_confusion_matrix(pred_test, gt_test[:-VAL_SIZE])

    # precision = metrics.precision_score(pred_test, gt_test[:-VAL_SIZE])
    # recall = metrics.recall_score(pred_test, gt_test[:-VAL_SIZE])
    # f1score = metrics.f1_score(pred_test, gt_test[:-VAL_SIZE])

    KAPPA_3D_DenseNet.append(kappa)
    OA_3D_DenseNet.append(overall_acc)
    AA_3D_DenseNet.append(average_acc)
    # TRAINING_TIME_3D_DenseNet.append(toc6 - tic6)
    # TESTING_TIME_3D_DenseNet.append(toc7 - tic7)
    ELEMENT_ACC_3D_DenseNet[index_iter, :] = each_acc

    # PRECISION_3D_DenseNet.append(precision)
    # RECALL_3D_DenseNet.append(recall)
    # F1SCORE_3D_DenseNet.append(f1score)

    print("3D DenseNet  finished.")
    print("# %d Iteration" % (index_iter + 1))

    # print(counter)
modelStatsRecord.outputStats_assess(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet, CATEGORY,
                                    'D:/3D-TwoStep-Network/records/IN_test_ResNeXt_TwoStepAttention_5_1_4_60_group6_1.txt',
                                    'D:/3D-TwoStep-Network/records/IN_test_element_ResNeXt_TwoStepAttention_5_1_4_60_group6_1.txt')
# modelStatsRecord.outputStats_assess(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
#                                     CATEGORY,
#                                     'D:/3D-ResNeXt-master/records/UP_test_3D_ResneXt_Dual_loss_5_1_4_60_1.txt',
#                                     'D:/3D-ResNeXt-master/records/UP_test_element_3D_ResneXt_Dual_loss_5_1_4_60_1.txt')
# modelStatsRecord.outputStats_assess(KAPPA_3D_DenseNet, OA_3D_DenseNet, AA_3D_DenseNet, ELEMENT_ACC_3D_DenseNet,
#                                     CATEGORY,
#                                     'D:/3D-ResNeXt-master/records/IN_test_3D_SSRN_6_1_3_50_2.txt',
#                                     'D:/3D-ResNeXt-master/records/IN_test_element_3D_SSRN_6_1_3_50_2.txt')


