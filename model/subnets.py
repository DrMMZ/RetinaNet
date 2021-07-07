"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Classification and Regression Subnets in RetinaNet
"""

import tensorflow as tf


def cls_subnet(
        num_anchors_per_pixel, 
        num_object_classes, 
        channels_fmap=256,
        pi=0.01
        ):
    """
    Builds a classification subnet of RetinaNet.

    Parameters
    ----------
    num_anchors_per_pixel : integer
        The number of anchors to generate at different scales for every pixel;
        see anchors.anchors_from_fpn().
    num_object_classes : integer
        The number of classes containing only objects, i.e., object classes 
        denoted by positive integers while background denoted by 0.
    channels_fmap : integer, optional
        The number of filters taken from a FPN conv layer. The default is 256.
    pi : float, optional
        The bias initialization at the final conv layer of the classification
        subnet, prevents the large number of anchors from generating a large
        loss value in the first iteration of training. The default is 0.01.

    Returns
    -------
    model : tf keras
        The classification subnet.

    """
    # a FPN feature map, [batch_size, h, w, channels_fmap]
    inputs = tf.keras.Input(
            shape=(None, None, channels_fmap), name='cls_subnet_input')
    
    x = inputs
    for i in range(4):
        x = tf.keras.layers.Conv2D(
            channels_fmap, (3,3), strides=(1,1), padding='same', 
            activation='relu', name='cls_subnet_conv' + str(i+1))(x)
    
    # probs, [batch_size, h, w, num_anchors_per_pixel * num_object_classes]
    b = -tf.math.log((1-pi)/pi).numpy()
    initializer = tf.keras.initializers.Constant(b)
    outputs = tf.keras.layers.Conv2D(
        num_anchors_per_pixel * num_object_classes, (3,3), strides=(1,1), 
        padding='same', activation='sigmoid', bias_initializer=initializer, 
        name='cls_subnet_output')(x)
    
    with tf.device('/cpu:0'):
        model = tf.keras.Model(inputs, outputs, name='cls_subnet')
        return model
    
    
def reg_subnet(num_anchors_per_pixel, channels_fmap=256):
    """
    Builds a regression subnet of RetinaNet.

    Parameters
    ----------
    num_anchors_per_pixel : integer
        The number of anchors to generate at different scales for every pixel;
        see anchors.anchors_from_fpn().
    channels_fmap : integer, optional
        The number of filters taken from a FPN conv layer. The default is 256.

    Returns
    -------
    model : tf keras
        The regression subnet.

    """
    # a FPN feature map, [batch_size, h, w, channels_fmap]
    inputs = tf.keras.Input(
            shape=(None, None, channels_fmap), name='reg_subnet_input')
    
    x = inputs
    for i in range(4):
        x = tf.keras.layers.Conv2D(
            channels_fmap, (3,3), strides=(1,1), padding='same', 
            activation='relu', name='reg_subnet_conv' + str(i+1))(x)
    
    # [batch_size, h, w, 4 * num_anchors_per_pixel]
    outputs = tf.keras.layers.Conv2D(
        num_anchors_per_pixel * 4, (3,3), strides=(1,1), 
        padding='same', name='reg_subnet_output')(x)
    
    with tf.device('/cpu:0'):
        model = tf.keras.Model(inputs, outputs, name='reg_subnet')
        return model

