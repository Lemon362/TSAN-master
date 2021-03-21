from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    # 如果不设置output_channels的值，则默认尺寸不变
    # 1x1x1，3x3x3，1x1x1
    # 先BN,ReLU,再CONV
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input.get_shape()[-1].value  # 获得输出通道数
    if input_channels is None:
        input_channels = output_channels // 4  # 输入通道数，也就是前两次卷积的卷积核

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    # （9，9，97，64）  （5，5，49，128）  （3，3，25，256）  （2，2，13，512）
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """
    # attention模块中的超参数，p表示残差模块个数，t表示不进行采样的残差个数，r表示上下采样的残差个数
    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value  # input_channels=128，获得输入通道数
    if output_channels is None:
        output_channels = input_channels  # 输出通道数

    # First Residual Block 上边（可省略） 先经过一次残差模块
    for i in range(p):  # p=1
        input = residual_block(input)
        # 输入（9，9，97，64）  输出（9，9，97，64）
        # 输入（5，5，49，128）  输出（5，5，49，128）
        # 输入（3，3，25，256）  输出（3，3，25，256）
        # 输入（2，2，13，512）  输出（2，2，13，512）

    # TODO Attention Block
    # TODO Trunk Branch 右边，普通的残差模块，两次
    output_trunk = input  # （9，9，97，64）  （5，5，49，128）  （3，3，25，256）  （2，2，13，512）
    for i in range(t):  # t=2
        output_trunk = residual_block(output_trunk)  # 残差
        # 输入（9，9，97，64）  输出（9，9，97，64）
        # 输入（5，5，49，128）  输出（5，5，49，128）
        # 输入（3，3，25，256）  输出（3，3，25，256）
        # 输入（2，2，13，512）  输出（2，2，13，512）

    # TODO Soft Mask Branch 左边
    # encoder
    # TODO 第一次下采样
    # first down sampling  先经过一次maxpooling池化，下采样
    output_soft_mask = MaxPooling2D(padding='same')(input)  # 32x32 第一次下采样
    # 输入（9，9，97，64）  输出（5，5，49，64）
    # 输入（5，5，49，128）  输出（5，5，49，128）
    # 输入（3，3，25，256）  输出（3，3，25，256）
    # 输入（2，2，13，512）  输出（2，2，13，512）
    for i in range(r):  # r=1
        output_soft_mask = residual_block(output_soft_mask)  # 再经过一次残差模块
        # 输入（8，8，128）  输出（8，8，128）
        # 输入（4，4，256）  输出（4，4，256）

    # TODO attention模块
    skip_connections = []
    # encoder_depth=2
    for i in range(encoder_depth - 1):  # i=0
        # TODO add_6
        output_skip_connection = residual_block(output_soft_mask)
        # 输入output_soft_mask（8，8，128）  输出add_6（8，8，128）
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        # TODO 第二次下采样
        output_soft_mask = MaxPooling2D(padding='same')(output_soft_mask)
        # 输入output_soft_mask（8，8，128）  输出（4，4，128）
        for _ in range(r):  # r=1
            # TODO add_7
            output_soft_mask = residual_block(output_soft_mask)  # 经过残差模块
            # 输入（4，4，128）  输出add_7（4，4，128）

    # decoder
    skip_connections = list(reversed(skip_connections))  # 将skip_connections中的残差模块变为可迭代的迭代器
    # TODO 第一次上采样
    for i in range(encoder_depth - 1):  # i=0
        # upsampling 上采样
        for _ in range(r):  # r=1
            output_soft_mask = residual_block(output_soft_mask)  # 先经过残差模块
            # 输入（4，4，128）  输出（4，4，128）
        output_soft_mask = UpSampling2D()(output_soft_mask)  # 第一次上采样
        # 输入（4，4，128）  输出（8，8，128）
        # TODO add_9
        # skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    # TODO 第二次上采样
    # last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)  # 先经过残差模块
        # 输入（8，8，128）  输出（8，8，128）
    output_soft_mask = UpSampling2D()(output_soft_mask)  # 第二次上采样
    # 输入（8，8，128）  输出（16，16，128）

    # Output 两次CONV
    output_soft_mask = Conv2D(output_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(output_channels, (1, 1))(output_soft_mask)
    # 输入（16，16，128）  输出（16，16，128）
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # TODO Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  # 对应元素相乘

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)  # 经过一次残差模块

    return output
