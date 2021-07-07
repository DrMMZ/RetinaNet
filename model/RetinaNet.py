"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

RetinaNet
"""

import os, datetime, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import retinanet_model, anchors, resnet_fpn, utils


class RetinaNet(object):
    """
    Defines a class based on RetinaNet, including training (can use synchronized 
    multi-gpu training), detecting and evaluation.
    
    """
    
    def __init__(self, mode, config):
        """
        A constructor.

        Parameters
        ----------
        mode : string
            The mode of building a retinanet in {'training', 'inference'}.
        config : class
            A custom configuration, see config.Config().

        Returns
        -------
        None.

        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        
        if mode == 'inference':
            self.model = self.build_retinanet(mode, config)          
            if config.checkpoint_path is not None:
                print('\nLoading checkpoint:\n%s\n' % config.checkpoint_path)
                self.model.load_weights(config.checkpoint_path, by_name=False)
        
        
    def build_retinanet(self, mode, config):
        """
        Builds a RetinaNet.

        Parameters
        ----------
        mode : string
            The mode of building a retinanet in {'training', 'inference'}.
        config : class
            A custom configuration, see config.Config().

        Returns
        -------
        model : tf keras model
            A retinanet based on the given config.

        """
        model = retinanet_model.retinanet(
            mode,
            config.offsets_mean,
            config.offsets_std,
            config.architecture, 
            config.train_bn, 
            config.channels_fmap,
            config.num_anchors_per_pixel, 
            config.num_object_classes,
            config.pi,
            config.alpha, 
            config.gamma,
            config.confidence_threshold, 
            config.num_top_scoring,
            config.batch_size_per_gpu,
            config.max_objects_per_class_per_img,
            config.iou_threshold,
            output_top_scoring=False)
        return model
    
    
    def compile_model(
            self, 
            model, 
            lr, 
            momentum, 
            beta_2, 
            l2, 
            loss_names=['cls_loss', 'reg_loss']
            ):
        """
        Add Adam optimizer, loss(es) and L2-regularization to the model.

        Parameters
        ----------
        model : tf keras model
            The built retinanet.
        lr : float
            A learning rate.
        momentum : float
            A scalar in Adam controlling moving average of the gradients decay.
        beta_2 : float
            A scalar in Adam controlling moving average of the squared gradients 
            decay.
        l2 : float
            A scalar in L2-regularization controlling the strength of 
            regularization.
        loss_names : list, optional
            The name(s) of loss function(s) in the model. The default is 
            ['cls_loss', 'reg_loss'], i.e., focal (classification) and 
            smooth-L1 (regression) losses defined in losses.ClsLoss() and 
            losses.RegLoss(), respectively.

        Returns
        -------
        None.

        """
        # optimizer
        optimizer = tf.keras.optimizers.Adam(
            lr=lr, 
            beta_1=momentum, 
            beta_2=beta_2,
            epsilon=1e-7)
                
        # losses
        for name in loss_names:
            layer = model.get_layer(name)
            loss = layer.output
            model.add_loss(loss)
            model.add_metric(loss, name=name)        
                    
        # l2-regularization, exclude batch norm weights
        reg_losses = []
        for w in model.trainable_weights:
            if 'gamma' not in w.name and 'beta' not in w.name:
                reg_losses.append(
                    tf.math.divide(
                        tf.keras.regularizers.L2(l2)(w),
                        tf.cast(tf.size(w), w.dtype)))
        model.add_loss(lambda: tf.math.add_n(reg_losses))
        
        # compile the model            
        model.compile(
            optimizer=optimizer, 
            loss=[None] * len(model.outputs))
        
            
    def train(self, 
              train_generator, 
              val_generator=None, 
              plot_training=True
              ): 
        """
        Trains the RetinaNet.

        Parameters
        ----------
        train_generator : python generator
            Described in data_gen.data_generator().
        val_generator : python generator, optional
            Described in data_gen.data_generator(). The default is None.
        plot_training : boolean, optional
            Whether to plot the learning curves. The default is True.

        Returns
        -------
        The trained model, training log and plot if needed.

        """
        assert self.mode == 'training', \
            'Need to create an instance in training mode.'
            
        if self.config.num_gpus > 1 and \
            len(tf.config.list_physical_devices('GPU')) > 1:
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        else:
            strategy = tf.distribute.get_strategy() 
            
        with strategy.scope():
            self.model = self.build_retinanet(self.mode, self.config)
            
            if self.config.checkpoint_path is not None:
                print('\nLoading checkpoint:\n%s\n' \
                      % self.config.checkpoint_path)
                self.model.load_weights(
                    self.config.checkpoint_path, by_name=False)
                
            if self.config.resnet_weights_path is not None:
                print('\nLoading resnet:\n%s\n' \
                      % self.config.resnet_weights_path)
                self.model.load_weights(
                    self.config.resnet_weights_path, by_name=True)
                # if need to freeze resnet, uncomment the following
                # self.config.train_bn = False
                # for i in range(len(self.model.layers)):
                #     layer = self.model.layers[i]
                #     if layer.name == 'fpn_c5p5':
                #         assert self.model.layers[i-1].name == 'res5c_out'
                #         break
                #     layer.trainable = False
        
            self.compile_model(
                self.model, 
                self.config.lr, 
                self.config.momentum, 
                self.config.beta_2,
                self.config.l2)
            # assign a learning rate after loading a checkpoint; otherwise it 
            # will continue on the last learning rate in the checkpoint
            self.model.optimizer.lr.assign(self.config.lr)
            print('\nlearning rate:', self.model.optimizer.lr.numpy(), '\n')
        
        # callbacks, including CSVLogger, ModelCheckpoint, ReduceLROnPlateau,
        # and EarlyStopping
        callbacks = []
        ROOT_DIR = os.getcwd()
        log_dir = os.path.join(ROOT_DIR, 'checkpoints')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(log_dir, current_time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if self.config.save_weights:
            self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path, 
                save_weights_only=True)
            callbacks.append(cp_callback)
        if self.config.reduce_lr:
            reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=10)
            callbacks.append(reduce_lr_callback)
        if self.config.early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True)
            callbacks.append(early_stopping_callback)
        log_filename = os.path.join(checkpoint_dir, '%s.csv' % current_time)
        log_callback = tf.keras.callbacks.CSVLogger(
            log_filename, 
            append=False)
        callbacks.append(log_callback)
        
        # train
        history = self.model.fit(
            train_generator, 
            epochs=self.config.epochs, 
            steps_per_epoch=self.config.steps_per_epoch, 
            callbacks=callbacks, 
            validation_data=val_generator, 
            validation_steps=self.config.validation_steps,
            validation_freq=self.config.validation_freq)
        
        # learning curves, saved to checkpoint_dir
        if plot_training:
            loss_names = []
            for x in history.history.keys():
                if 'loss' in x:
                    loss_names.append(x)
                else:
                    print(x, history.history[x])
            train_loss_names, val_loss_names = [], []
            for x in loss_names:
                if 'val' in x:
                    val_loss_names.append(x)
                else:
                    train_loss_names.append(x)
                    
            train_losses = []
            for name in train_loss_names:
                train_losses.append(history.history[name])
            val_losses = []
            for name in val_loss_names:
                val_losses.append(history.history[name])
                
            plt.figure(figsize=(10, 10))
            for i in range(len(train_loss_names)):
                plt.subplot(3, 1, i+1)
                plt.plot(train_losses[i], label='train')
                plt.plot(val_losses[i], label='val')
                plt.title(train_loss_names[i])
                plt.legend()
            plt.savefig(os.path.join(checkpoint_dir, '%s.png' % current_time))
            plt.show()
            
            
    def detect(self, images, verbose=False):
        """
        Detects a set of images.

        Parameters
        ----------
        images : numpy array, [batch_size, height, width, 3]
            The given batch of raw images, i.e., not required normalized by 255, 
            centered (substracting mean pixel per-channel) nor standardized
            (centered and divided by standard deviation pixel per-channel).
        verbose : boolean, optional
            Whether to display the detection time.

        Returns
        -------
        boxes_batch : list
            Each element is the detected bounding boxes of an image, of shape
            [num_boxes, 4] where 4 is the corner coordinates.
        class_ids_batch : list
            Each element is the detected class ids of bounding boxes, of shape
            [num_boxes, ].
        scores_batch : list
            Each element is the detected scores of bounding boxes, of shape
            [num_boxes, ].
        t : float
            The detection time in seconds.

        """
        assert self.mode == 'inference', \
            'Need to create an instance in inference mode.'
        assert images.shape[0] == self.config.batch_size_per_gpu, \
            'The number of images has to match with the batch size per gpu.'
        
        self.config.image_shape = images.shape[1:]
        window = (0,0) + images.shape[1:3]
        self.config.fmap_sizes = resnet_fpn.compute_fmap(images.shape[1:])
        
        # generate a list of anchors, each is at different level of FPN of 
        # shape [batch_size, h_i, w_i]
        anchors_fpn = anchors.anchors_from_fpn(
                self.config.scales, 
                self.config.ratios, 
                self.config.fmap_sizes, 
                self.config.fmap_strides, 
                self.config.denser)
        anchors_fpn_batches = []
        for i in range(len(anchors_fpn)):
            a_i = np.broadcast_to(
                anchors_fpn[i], 
                (self.config.batch_size_per_gpu,) + anchors_fpn[i].shape)
            anchors_fpn_batches.append(a_i)
        # print('anchors', np.concatenate(anchors_fpn_batches, axis=1).shape)
        
        # standardize images
        if self.config.channels_mean is not None \
            and self.config.channels_std is not None:
                images = (
                    images - self.config.channels_mean
                    ) / self.config.channels_std
        elif self.config.channels_mean is not None:
            images = images - self.config.channels_mean
        elif self.config.channels_std is not None:
            images = images / self.config.channels_std
        else:
            images = images / 255.0
        window = tf.expand_dims(tf.constant(window), axis=0)
        inputs = [images, anchors_fpn_batches, window]
        t1 = time.time()
        boxes_batch, class_ids_batch, scores_batch = self.model(inputs)
        t2 = time.time()
        t = t2 - t1
        if verbose: print('\ndetection time: %fs\n' %t)
        return boxes_batch, class_ids_batch, scores_batch, t
    
    
    def evaluate(self, dataset, verbose=False):
        """
        Evaluates the trained RetinaNet for a given dataset using mAP metric,
        in particular, the images in dataset have differen shapes.

        Parameters
        ----------
        dataset : class
            Described in utils.Dataset().
        verbose : boolean, optional
            Whether to display mAP for each image. The default is False.

        Returns
        -------
        mAP : float
            The mAP result for the dataset.
        output : numpy array
            Records AP and detection time for each image, [num_images, 3] where 
            3 is (image_name, AP, time).

        """
        image_names, APs, times = [], [], []
        
        # loop if images in dataset have different shapes
        # t1 = time.time()
        for image_id in dataset.image_ids:
            image_name = dataset.get_image_info(image_id)['id']
            image_names.append(image_name)
            if verbose: print('\n---%d, %s' % (image_id, image_name))
                
            image, boxes, class_ids, cache = dataset.load_data(
                image_id,
                shortest_side=self.config.shortest_side,
                mode=self.config.resize_mode)
            image1 = image[np.newaxis,...]
            boxes_batch, class_ids_batch, scores_batch, t = self.detect(
                image1, verbose=verbose)
            times.append(t)
            
            i = 0
            pred_boxes = boxes_batch[i]
            # refine pred_boxes by clipping them to the window
            window = cache[1]
            pred_boxes = utils.clip_boxes(pred_boxes, window)
            pred_boxes = pred_boxes.numpy()
            # since detected classes are 0-base and objects only 
            # (not background 0), increase 1
            pred_class_ids = class_ids_batch[i].numpy() + 1
            pred_scores = scores_batch[i].numpy()
            x = utils.compute_mAP(
                boxes, 
                class_ids, 
                pred_boxes, 
                pred_class_ids, 
                pred_scores, 
                verbose=verbose)
            APs.append(x)       
        # t2 = time.time()
        # print('\nevaluation time: %fs\n' %(t2-t1))
        
        mAP = np.array(APs).mean()
        print('\nDataset mAP:', mAP)
        output = np.stack([image_names, APs, times], axis=1)
        return mAP, output
        
            

