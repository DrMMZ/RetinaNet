"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Configurations
"""

# influenced by
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py

import numpy as np

import resnet_fpn


class Config(object):
    """
    Defines a custom configurations for RetinaNet.
    
    """
    
    name = None 
    
    ################
    # The model
    ################
    architecture = 'resnet50' # resnet architecture
    train_bn = False # moving average of mean and variance in BN layers?
    channels_fmap = 256 # number of channels in FPN and two subnets layers
    pi = 0.01 # initialization at the final conv layer of the cls subnet
    alpha = 0.25 # class weighting factor in focal loss
    gamma = 2.0 # focusing factor in focal loss
    confidence_threshold = 0.05 # minimum probability to select boxes
    num_top_scoring = 1000 # minimum number of boxes to select
    batch_size_per_gpu = 2    
    iou_threshold = 0.5 # used in NMS

    ################
    # Training
    ################
    num_gpus = 1 # used in multi-GPU training or inferencing
    checkpoint_path = None # previous trained model weights path
    resnet_weights_path = None # previous trained resnet weights path
    lr = 0.001 # learning rate
    momentum = 0.9 # Adam opt scalar for moving average of grads decay
    beta_2 = 0.999 # Adam opt scalar for moving average of squared grads decay
    l2 = 0.0001 # L2 regularization strength
    save_weights = True # save trained weights?
    epochs = 1 # training epochs
    validation_freq = 1 # frequence to validate
    reduce_lr = False # apply ReduceLROnPlateau?
    early_stopping = False # apply EarlyStopping?
    
    ################
    # Anchors
    ################
    scales = [32, 64, 128, 256, 512] # anchor scales for 5 FPN layers 
    ratios = [0.5, 1, 2] # height/width ratio
    fmap_strides = [2**3, 2**4, 2**5, 2**6, 2**7] # strides to get 5 FPN layers
    denser = False # add more scales?
    offsets_mean = None # offsets mean for each coordinate, optional
    offsets_std = None # offsets std for each coordinate, optional
    
    ################
    # Image
    ################
    channels_mean = None # images mean pixel per channel, optional
    channels_std = None # images std pixel per channel, optional
    shortest_side = 512 # shortest side after resized
    longest_side = 1024 # longest side after resized
    upscale_factor = 1.0 # upscale factor for resizing
    resize_mode = 'crop' # in {'crop', 'pad_square', 'pad_fpn', 'none'}
    more_scales = False # if True, randomly choose resize_mode in {'pad_square', 'crop'}
    augmenters = None # e.g., [utils.Rotate(), utils.FlipLR()]
    max_num_crops = 10 # maximum number to crop if last crop contain no object
    
    ################
    # Dataset
    ################
    num_object_classes = 1 # number of classes containing objects
    max_objects_per_class_per_img = 100 # max num of objects in a class, for all classes and images
    num_train_images = None # number of training images
    num_val_images = None # number of validation images 
    
    def __init__(self):
        assert self.shortest_side % 128 == 0, \
            'Shortest side must be a multiple of 128.'
        assert self.longest_side % 128 == 0, \
            'Longest side must be a multiple of 128.'
        # image shape
        if self.resize_mode == 'crop':
            self.image_shape = (self.shortest_side, self.shortest_side, 3)
            # FPN feature maps sizes
            self.fmap_sizes = resnet_fpn.compute_fmap(self.image_shape)
        elif self.resize_mode == 'pad_square':
            self.image_shape = (self.longest_side, self.longest_side, 3)
            # FPN feature maps sizes
            self.fmap_sizes = resnet_fpn.compute_fmap(self.image_shape)
        else:
            # need to manually adjust if resize_mode = 'pad_fpn' or 'none'
            self.image_shape = (512, 512, 3)
        
        # number of anchors per pixel
        if self.denser:
            self.num_anchors_per_pixel = len(self.ratios) * 3
        else:
            self.num_anchors_per_pixel = len(self.ratios)
        
        # global batch size
        self.batch_size_global = self.batch_size_per_gpu * self.num_gpus
        
        # number of training and validation steps per epoch
        self.steps_per_epoch = int(np.ceil(
            self.num_train_images / self.batch_size_global))
        self.validation_steps = int(np.ceil(
            self.num_val_images / self.batch_size_global))
        
        # if self.more_scales:
        #     assert self.longest_side == self.shortest_side, \
        #         'Require longest_side = shortest_side.'
        
        
    def display(self):
        print('----------Configurations----------\n')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
            
        
    
