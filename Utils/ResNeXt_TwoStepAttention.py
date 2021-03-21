import keras.backend as K
import six
import tflearn
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, Lambda, concatenate, add, \
    GlobalAveragePooling3D, Dense, GlobalMaxPooling3D, Flatten, Dropout, MaxPooling3D, Reshape, Conv2D, MaxPooling2D, \
    multiply, dot, Add, Softmax, GlobalAvgPool2D, GlobalAvgPool3D
import keras
from keras_layer_normalization import LayerNormalization

from deform_conv.layers import ConvOffset2D
from layer import DCNN3D
from non_local import non_local_block


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4, flag=False):
    # TODO 分组卷积
    # grouped_channels=16  strides=1
    # grouped_channels=32  strides=2
    # grouped_channels=64  strides=2

    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    group_list = []

    # 标准卷积，不执行
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    if flag:
        for c in range(cardinality // 2):
            x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]

            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)

            # 分组后各自卷积
            x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                       kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

            group_list.append(x)  # 将x存放在列表里，共8个

        for c in range(cardinality // 2, cardinality):
            x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]

            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)

            # 分组后各自卷积
            x = Conv3D(grouped_channels, (5, 5, 5), padding='same', use_bias=False, strides=(strides, strides, strides),
                       kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

            group_list.append(x)  # 将x存放在列表里，共8个

    else:
        for c in range(cardinality):  # 8组
            # 根据channel维度进行分组
            x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels]
            if K.image_data_format() == 'channels_last' else
            lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :, :])(input)

            # 分组后各自卷积
            x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                       kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

            group_list.append(x)  # 将x存放在列表里，共8个

    # concat拼接
    group_merge = concatenate(group_list, axis=channel_axis)  # 将8个以channel维（最后一维）拼接

    x = BatchNormalization(axis=channel_axis)(group_merge)

    return x


def DRSN_Block(x_input, cardinality=8, filters=None, weight_decay=5e-4):
    # TODO 软阈值化
    # Absolute + GAP
    x_abs = K.abs(x_input)
    x_gap = K.mean(x_abs, axis=3, keepdims=True)
    x_gap = K.mean(x_gap, axis=2, keepdims=True)
    x_gap = K.mean(x_gap, axis=1, keepdims=True)
    # x_mean = GlobalAveragePooling3D()(K.tf.abs(x))  # 输出尺寸（？，64）
    # 左分支
    x_avg = x_gap
    # 右分支
    scales = Dense(filters // cardinality, use_bias=False, kernel_regularizer=l2(weight_decay))(x_gap)
    scales = BatchNormalization(axis=-1)(scales)
    scales = Activation('relu')(scales)
    scales = Dense(filters, use_bias=False, kernel_regularizer=l2(weight_decay))(scales)
    scales = BatchNormalization(axis=-1)(scales)
    x_sigmoid = Activation('sigmoid')(scales)
    # A*a
    thres = multiply([x_sigmoid, x_avg])
    # 软阈值化
    x_thres = multiply([K.sign(x_input), K.maximum(K.abs(x_input) - thres, 0)])

    return x_thres


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4, flag=False):
    # TODO resnext模块

    init = input
    # （9，9，37，24） 128 s=1
    # （9，9，37，128） 256 s=2
    # （5，5，19，256） 512 s=2

    # 分组卷积的个数
    grouped_channels = int(filters / cardinality)  # 128/8=16  256/8=32  512/8=64
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    # TODO 右边shortcut
    # 判断底层
    if K.image_data_format() == 'channels_first':  # 不执行
        if init._keras_shape[1] != 2 * filters:
            init = Conv3D(filters * 2, (1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:  # 底层tf，执行下面的代码
        if init._keras_shape[-1] != 2 * filters:  # 使用1x1x1卷积改变尺寸，使得shortcut的channel维度和分组卷积的滤波器相同才可以add
            init = Conv3D(filters, (1, 1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            # padding为same时，只表示图像的数据不丢失，若步长为1，则输出图像的大小不变；若步长为2，则输出图像的大小为输入图像大小的一半。
            # 1x1x1，128，s=（1，1，1），SAME  conv2d_2
            # 输入（9，9，37，24）  输出（9，9，37，128）
            # 1x1x1，256，s=（2，2，2），SAME  conv2d_33
            # 输入（9，9，37，128）  输出（5，5，19，256）
            # 1x1x1，512，s=（2，2，2），SAME
            # 输入（5，5，19，256）  输出（3，3，10，512）
            init = BatchNormalization(axis=channel_axis)(init)  # 这里不加relu是因为在后面add之后一起激活

    # TODO 左边分组卷积之前的卷积，使用1x1改变卷积核个数
    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    # 1x1x1，128，s=1，SAME  conv2d_3
    # 输入（9，9，37，24）  输出（9，9，37，128）
    # 1x1x1，256，s=1，SAME  conv2d_34
    # 输入（9，9，37，128）  输出（9，9，37，256）
    # 1x1x1，512，s=1，SAME  conv2d_34
    # 输入（5，5，19，256）  输出（5，5，19，512）
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # TODO 分组卷积
    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay, flag=flag)
    # grouped_channels=16  strides=1
    # 8组  输入（9，9，37，128）  输出（9，9，37，128）
    # grouped_channels=32  strides=2
    # 8组  输入（9，9，37，256）  输出（5，5，19，256）
    # grouped_channels=64  strides=2
    # 8组  输入（5，5，19，512）  输出（3，3，10，512）

    # （9，9，97，64）  （5，5，49，128）  （3，3，25，256）  （2，2，13，512）

    # TODO Nonlocal + SeNet + 软阈值化
    x_nonlocal_senet = NonLocal_SeNet_Block(x, filters, mode='embedded', compression=2)

    # TODO 左右连接
    x = add([init, x_nonlocal_senet])  # add_1  残差连接shortcut
    # x = add([init, x])
    # 输出（9，9，37，128）
    # 输出（5，5，19，256）
    # 输出（3，3，10，512）

    x = Activation('relu')(x)

    return x


def NonLocal_SeNet_Block(x_input, out_dims, mode=None, compression=2, reduction_ratio=4):
    residual_abs = Lambda(abs_backend, name="abs_non" + str(out_dims))(x_input)

    x = Conv3D(out_dims, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x_input)

    # NonLocal
    x_non_local = non_local_block(x, mode='embedded', compression=2)

    # SeNet
    abs_mean = GlobalAveragePooling3D()(x_non_local)

    # scales = Dense(units=out_dims // reduction_ratio, activation=None, kernel_initializer='he_normal',
    #                kernel_regularizer=l2(1e-4))(abs_mean)
    # scales = Activation('relu')(scales)
    # scales = Dense(units=out_dims)(scales)
    # scales = Activation('sigmoid')(scales)
    # scales = Reshape((1, 1, 1, out_dims))(scales)

    scales = Reshape((1, 1, 1, out_dims))(abs_mean)
    scales = Conv3D(filters=out_dims // reduction_ratio, kernel_size=1,
                    use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(scales)
    scales = Activation('relu')(scales)
    scales = Conv3D(filters=out_dims, kernel_size=1,
                    use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(scales)
    scales = Activation('sigmoid')(scales)

    thres = multiply([x, scales])

    # Soft thresholding
    sub = keras.layers.subtract([residual_abs, thres])
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros])
    residual = keras.layers.multiply([Lambda(sign_backend, name="sign_non" + str(out_dims))(x_input), n_sub])

    return residual


def SENet_Block(x_input, out_dims, reduction_ratio=4):
    residual_abs = Lambda(abs_backend, name="abs_se" + str(out_dims))(x_input)

    x = Conv3D(out_dims, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x_input)

    abs_mean = GlobalAveragePooling3D()(x)

    # scales = Dense(units=out_dims // reduction_ratio, activation=None, kernel_initializer='he_normal',
    #                kernel_regularizer=l2(5e-4))(abs_mean)
    # scales = Activation('relu')(scales)
    # scales = Dense(units=out_dims)(scales)
    # scales = Activation('sigmoid')(scales)
    # scales = Reshape((1, 1, 1, out_dims))(scales)

    scales = Reshape((1, 1, 1, out_dims))(abs_mean)
    scales = Conv3D(filters=out_dims // reduction_ratio, kernel_size=1,
                    use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(scales)
    scales = Activation('relu')(scales)
    scales = Conv3D(filters=out_dims, kernel_size=1,
                    use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(scales)
    scales = Activation('sigmoid')(scales)

    # thres = multiply([abs_mean, scales])
    thres = multiply([scales, x])

    # Soft thresholding
    sub = keras.layers.subtract([residual_abs, thres])
    zeros = keras.layers.subtract([sub, sub])
    n_sub = keras.layers.maximum([sub, zeros])
    residual = keras.layers.multiply([Lambda(sign_backend, name="sign_se" + str(out_dims))(x_input), n_sub])
    # residual = keras.layers.multiply([Lambda(sign_backend)(x_input),
    #                                   keras.layers.maximum([Lambda(abs_mean)(x_input) - thres, 0])])

    return residual


def __initial_conv_block(input, weight_decay=5e-4):
    # TODO 初始化卷积层
    # x = __initial_conv_block(img_input, weight_decay)

    # 底层tf，channel最后
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1  # -1

    # 3x3x20，24，s=（1，1，5）
    x = Conv3D(32, (3, 3, 7), strides=(1, 1, 2), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), name="init_conv")(input)
    # 输入（11，11，200，1）  输出（9，9，37，24）
    x = BatchNormalization(axis=channel_axis, name="init_BN")(x)
    # x = Activation('relu', name="init_ReLU")(x)

    return x


def _bn_relu_spc(input):  # BN + ReLU（spectral）
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu_spc(**conv_params):  # CONV + BN + ReLU（spectral） 先CONV再BN和激活
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))  # 下采样步长
    init = conv_params.setdefault("init", "he_normal")  # He正态分布初始化方法，初始化权重函数名称
    border_mode = conv_params.setdefault("border_mode", "same")  # 补零
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))  # W正则化

    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu_spc(conv)

    return f


def sign_backend(inputs):
    return K.sign(inputs)


def abs_backend(inputs):
    return K.abs(inputs)


def _bn_relu_conv_spc(**conv_params):  # BN + ReLU + CONV  先BN和激活再CONV（改进方法）
    # residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))  # 子采样步长
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      padding=border_mode)(activation)

    return f


def _shortcut_spc(input, residual):  # shortcut 残差块
    # _shortcut_spc(input, residual)
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3] + 1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]  # 通道匹配

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual._keras_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1, 1),
                          strides=(stride_dim1, stride_dim2, stride_dim3),
                          kernel_initializer="he_normal", padding="valid",
                          kernel_regularizer=l2(0.0001))(input)  # 使用1*1CONV，使得shortcut和residual通道匹配
    return add([shortcut, residual])  # 输出 shortcut + residual


def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):  # repetitions = 1 --> i = 0
            init_subsample = (1, 1, 2)  # 残差块中的CONV的S
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 1)  # 残差块前的CONV的S
            # init_subsample = (1, 1, 1)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):
        # is_first_block_of_first_layer=True
        if is_first_block_of_first_layer:  # 如果是第一层的第一个块
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=l2(0.0001),
                           filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)  # CONV2
        else:  # 不是残差块外的CONV，直接运行这一个条件
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                      subsample=init_subsample)(input)
        # spectral的CONVBN 1*1*7,24
        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)  # CONV3

        # TODO SeNet + 软阈值化
        residual = SENet_Block(residual, nb_filter)

        x = _shortcut_spc(input, residual)  # input + residual 第一个残差块的和

        x = Activation('relu')(x)

        return x

    return f


def _get_block(identifier):
    # block_fn_spc = _get_block(block_fn_spc)
    # block_fn = _get_block(block_fn)
    if isinstance(identifier, six.string_types):  # isinstance:判断是否是指定的格式
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def ResNet_SPC(input, repetitions1=None, block_fn_spc=None):
    block_fn_spc = _get_block(block_fn_spc)
    # 残差块
    block_spc = input
    nb_filter = 32
    for i, r in enumerate(repetitions1):  # i=0, r=1
        # i是索引，从0开始
        # r是元素
        block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r,
                                        is_first_layer=(i == 0))(block_spc)  # is_first_layer=True, repetitions=1
        nb_filter *= 2

    return block_spc


def __create_res_next(nb_classes, img_input, cardinality=8, weight_decay=5e-4):
    # TODO 网络搭建
    # x_dense = __create_res_next(classes, input, cardinality, weight_decay)

    # 三层模块的滤波器个数
    # filters_list = [64, 128, 256, 512]  # 64, 128, 256, 512
    if cardinality == 6:
        filters_list = [48, 96, 192, 384]
    elif cardinality == 8:
        filters_list = [64, 128, 256, 512]
    elif cardinality == 10:
        filters_list = [80, 160, 320, 640]

    # TODO ResNet_SPC_Attention
    # repetitions1: 数组[]类型，[1,1]表示每个模块重复一次
    # 第一次采样：(1, 1, 1)   第二次采样：(1, 1, 2)
    x_spc = ResNet_SPC(img_input, repetitions1=[1, 1], block_fn_spc=basic_block_spc)
    # 输入（11，11，200，1）   输出（11，11，100，64）

    # TODO 第一个模块
    x_1 = __bottleneck_block(x_spc, filters_list[0], cardinality, strides=1, weight_decay=weight_decay, flag=True)

    # TODO 第二个模块
    x_2 = __bottleneck_block(x_1, filters_list[1], cardinality, strides=2, weight_decay=weight_decay, flag=True)

    # TODO 第三个模块
    x_3 = __bottleneck_block(x_2, filters_list[2], cardinality, strides=2, weight_decay=weight_decay)

    # TODO 第四个模块
    x_4 = __bottleneck_block(x_3, filters_list[3], cardinality, strides=2, weight_decay=weight_decay)

    drop = Dropout(0.5)(x_4)

    flatten = Flatten()(drop)

    dense2 = Dense(1024, use_bias=False, kernel_regularizer=l2(weight_decay))(flatten)

    x_dense = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                    kernel_initializer='he_normal', activation='softmax')(dense2)

    return x_dense


def _handle_dim_ordering():
    # TODO 处理维度
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def A2net(input):
    channels = input._keras_shape[-1]
    intermediate_dim = channels // 2
    convA = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    convB = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    convV = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    (_, h, w, c) = K.int_shape(convB)
    feature_maps = Reshape((c, h * w))(convA)  # 对 A 进行reshape
    atten_map = Activation('softmax')(convB)
    atten_map = Reshape((h * w, c))(atten_map)# 对 B 进行reshape 生成 attention_aps
    global_descriptors = keras.layers.dot([feature_maps, atten_map], axes=[2,1], normalize=False)  # 特征图与attention_maps 相乘生成全局特征描述子
    atten_vectors = Activation('softmax')(convV)  # 生成 attention_vectors
    atten_vectors = Reshape((h*w, c))(atten_vectors)
    out = keras.layers.dot([atten_vectors, global_descriptors], axes=-1)# 注意力向量左乘全局特征描述子
    out = Reshape((c, h, w))(out)
    out = Reshape((h, w, c))(out)
    out = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(out)
    return out


def ResneXt_IN(input_shape=None, cardinality=8, weight_decay=5e-4, classes=None):
    # TODO 主函数
    # model = ResneXt_IN((1, 11, 11, 200), cardinality=8, classes=16)

    # 判断底层，tf，channel在最后
    # _handle_dim_ordering()

    # if len(input_shape) != 4:
    #     raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")
    #
    # print('original input shape:', input_shape)
    # # orignal input shape（1，11，11，200）
    #
    # if K.image_dim_ordering() == 'tf':
    #     input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
    # print('change input shape:', input_shape)

    # TODO 数据输入
    input = Input(shape=input_shape)

    # TODO 网络搭建
    x_dense = __create_res_next(classes, input, cardinality, weight_decay)
    # x_dense = A2net(input)

    # conv = DCNN3D(32, 24, (3, 3, 3), scope='deformconv1', norm=False)(input)
    # x_dense = Dense(16)(conv)

    model = Model(input, x_dense, name='resnext_IN')

    # feature_model = Model(input, outputs=model.get_layer('dense_1').output)

    return model


def main():
    # TODO 程序入口
    # TODO 可变参数3，cardinality
    model = ResneXt_IN((11, 11, 200, 1), cardinality=6, classes=16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    plot_model(model, show_shapes=True, to_file='./model_deform_cnn.png')


if __name__ == '__main__':
    main()

# 736,064
