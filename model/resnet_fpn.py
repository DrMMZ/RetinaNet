"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

ResFPN in RetinaNet
"""

import resnet, fpn
import tensorflow as tf


def resnet_fpn(
        batch_images, 
        architecture='resnet50', 
        train_bn=False, 
        channels_fmap=256
        ):
    """
    The backbone network Res-50/101-FPN for RetinaNet.

    Parameters
    ----------
    batch_images : tf tensor, [batch_size, height, width, 3]
        A batch of images.
    architecture : string
        The ResNet architecture in {'resnet50', 'resnet101'}.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.
    channels_fmap : integer, optional
        The number of filters in all FPN conv layers. The default is 256.

    Returns
    -------
    outputs : list
        FPN different level outputs for RetinaNet.

    """
    assert architecture in ['resnet50', 'resnet101'], \
        'Only support ResNet50/101.'
    
    # resnet
    C1, C2, C3, C4, C5 = resnet.backbone_resnet(
        batch_images, architecture, stage5=True, train_bn=train_bn)
    
    # fpn
    resnet_stages = [C1, C2, C3, C4, C5]
    P3, P4, P5, P6, P7 = fpn.backbone_fpn(
        resnet_stages, num_filters=channels_fmap)
    
    outputs = [P3, P4, P5, P6, P7]
    return outputs
    
    
def compute_fmap(image_shape):
    """
    Computes the feature map sizes.

    Parameters
    ----------
    image_shape : tuple
        The shape of images.

    Returns
    -------
    fmap_sizes : list
        The size of feature map at each level.

    """
    assert image_shape[0] >= 128 and image_shape[1] >= 128, \
        'Image size is too small to compute. One needs at least 128.'
    
    fmap_sizes = []
    # sizes in P3, P4, P5
    for i in range(3, 6):
        stride = 2**i
        fmap_h = int(image_shape[0] / stride)
        fmap_w = int(image_shape[1] / stride)
        fmap_size = (fmap_h, fmap_w)
        fmap_sizes.append(fmap_size)
    # sizes in P6, P7
    for i in range(2):
        fmap_h = int((fmap_h-3) / 2) + 1
        fmap_w = int((fmap_w-3) / 2) + 1
        fmap_size = (fmap_h, fmap_w)
        fmap_sizes.append(fmap_size)
        
    return fmap_sizes
