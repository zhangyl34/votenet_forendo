'''Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'''
import os
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(BASE_DIR)
import tf_logger


class Visualizer():
    def __init__(self, opt, name='train'):
        # self.opt = opt
        #self.logger = tf_logger.Logger(os.path.join(opt.logging_dir, opt.name))
        #self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        self.logger = tf_logger.Logger(os.path.join(opt.log_dir, name))
        # self.log_name = os.path.join(opt.log_dir, 'tf_visualizer_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to save
    def log_images(self, visuals, step):
            for label, image_numpy in visuals.items():
                self.logger.image_summary(
                    label, [image_numpy], step)

    # scalars: dictionary of scalar labels and values
    def log_scalars(self, scalars, step):
        for label, val in scalars.items():
            self.logger.scalar_summary(label, val, step)

