from kaffe.tensorflow import Network

class ResNet50(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1_7x7_s2')
             .batch_normalization(relu=True, name='conv1_7x7_s2_bn')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(name='conv2_1_1x1_increase_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(name='conv2_1_1x1_proj_bn'))

        (self.feed('conv2_1_1x1_increase_bn', 
                   'conv2_1_1x1_proj_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1_relu', 
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2_relu', 
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(name='conv3_1_1x1_increase_bn'))

        (self.feed('conv2_3_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(name='conv3_1_1x1_proj_bn'))

        (self.feed('conv3_1_1x1_increase_bn', 
                   'conv3_1_1x1_proj_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1_relu', 
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2_relu', 
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3_relu', 
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(name='conv4_1_1x1_increase_bn'))

        (self.feed('conv3_4_relu')
             .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(name='conv4_1_1x1_proj_bn'))

        (self.feed('conv4_1_1x1_increase_bn', 
                   'conv4_1_1x1_proj_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1_relu', 
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2_relu', 
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3_relu', 
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4_relu', 
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5_relu', 
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(name='conv5_1_1x1_increase_bn'))

        (self.feed('conv4_6_relu')
             .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(name='conv5_1_1x1_proj_bn'))

        (self.feed('conv5_1_1x1_increase_bn', 
                   'conv5_1_1x1_proj_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1_relu', 
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2_relu', 
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3_relu')
             .avg_pool(3, 3, 1, 1, padding='VALID', name='pool5_3x3_s1')
             # .fc(8631, relu=False, name='classifier')
         )