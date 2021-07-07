"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Feature Pyramid Networks (FPN) in RetinaNet
"""

# influenced by
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py

import tensorflow as tf


def backbone_fpn(resnet_stages, num_filters=256):
    """
    Adds a 5 stages FPN to ResNet 50/101.

    Parameters
    ----------
    resnet_stages : list
        The output feature maps [C1, C2, C3, C4, C5] from backbone_resnet() 
        where C5 is not empty.
    num_filters : integer, optional
       The number of filters in all conv layers. The default is 256.
   
    Returns
    -------
    outputs : list
        The set of feature maps [P3, P4, P5, P6, P7] at each level of the 
        second pyramid, designed for RetinaNet.

    """
    _, _, C3, C4, C5 = resnet_stages
    # assert tf.shape(C5)[1] >= 3**2 and tf.shape(C5)[2] >= 3**2, \
    #             'Image size (%d, %d) is too small to have FPN.' % (
    #                 C5.shape[1], C5.shape[2])
    
    P5 = tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c5p5')(C5)
    P4 = tf.keras.layers.Add(name='fpn_p4add')([
        tf.keras.layers.UpSampling2D((2,2), name='fpn_p5upsampled')(P5),
        tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c4p4')(C4)])
    P3 = tf.keras.layers.Add(name='fpn_p3add')([
        tf.keras.layers.UpSampling2D((2,2), name='fpn_p4upsampled')(P4),
        tf.keras.layers.Conv2D(num_filters, (1,1), name='fpn_c3p3')(C3)])
    
    # for p in [P3, P4, P5]:
    #     if p.shape[1]:
    #         assert p.shape[1] >= 2 and p.shape[2] >= 2, \
    #             'Image shape is too small to have FPN.'
    
    # 5 stages for each anchor scale
    P3 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p3')(P3)
    P4 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p4')(P4)
    P5 = tf.keras.layers.Conv2D(
        num_filters, (3,3), padding='same', name='fpn_p5')(P5)
    P6 = tf.keras.layers.Conv2D(
        num_filters, (3,3), strides=(2,2), name='fpn_p6')(C5)
    P7 = tf.keras.layers.Conv2D(
        num_filters, (3,3), strides=(2,2))(P6)
    P7 = tf.keras.layers.Activation('relu', name='fpn_p7')(P7)
    
    return [P3, P4, P5, P6, P7]
