import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

class ConvBlock(layers.Layer):
    def __init__(self, channels, kernel_size, strides):
        super(ConvBlock, self).__init__()
        
        initializer = tf.keras.initializers.HeNormal()
        regularizer = tf.keras.regularizers.L2(l2=1e-4)
        
        self.conv = tf.keras.Sequential([
            layers.Conv2D(channels, kernel_size=kernel_size, strides=strides, \
                          padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Conv2D(channels, kernel_size=kernel_size, strides=strides, \
                          padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer),
            layers.ELU(),
            layers.Dropout(0.5)
        ])
        
    def call(self, x, training):
        return self.conv(x, training=training)

class UnetPlusPlus(Model):
    def __init__(self):
        super(UnetPlusPlus, self).__init__()
        
        initializer = tf.keras.initializers.HeNormal()
        regularizer = tf.keras.regularizers.L2(l2=1e-4)
        
        self.conv_1_1 = ConvBlock(32, 3, 1)
        self.conv_1_2 = ConvBlock(32, 3, 1)
        self.conv_1_3 = ConvBlock(32, 3, 1)
        self.conv_1_4 = ConvBlock(32, 3, 1)
        self.conv_1_5 = ConvBlock(32, 3, 1)
        
        self.conv_2_1 = ConvBlock(64, 3, 1)
        self.conv_2_2 = ConvBlock(64, 3, 1)
        self.conv_2_3 = ConvBlock(64, 3, 1)
        self.conv_2_4 = ConvBlock(64, 3, 1)
        
        self.conv_3_1 = ConvBlock(128, 3, 1)
        self.conv_3_2 = ConvBlock(128, 3, 1)
        self.conv_3_3 = ConvBlock(128, 3, 1)
        
        self.conv_4_1 = ConvBlock(256, 3, 1)
        self.conv_4_2 = ConvBlock(256, 3, 1)
        
        self.conv_5_1 = ConvBlock(512, 3, 1)
        
        self.up_conv_1_2 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_1_3 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_1_4 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_1_5 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')
        
        self.up_conv_2_2 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_2_3 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_2_4 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')
        
        self.up_conv_3_2 = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.up_conv_3_3 = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')
        
        self.up_conv_4_2 = layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same')
        
        self.pool_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.pool_2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.pool_3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.pool_4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.nest_conv1 = layers.Conv2D(2, 1, 1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.nest_conv2 = layers.Conv2D(2, 1, 1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.nest_conv3 = layers.Conv2D(2, 1, 1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        self.nest_conv4 = layers.Conv2D(2, 1, 1, kernel_initializer=initializer, kernel_regularizer=regularizer)
    
    def call(self, x, training=True):
        conv1_1 = self.conv_1_1(x, training=training)
        pool1 = self.pool_1(conv1_1, training=training)
        
        conv2_1 = self.conv_2_1(pool1, training=training)
        pool2 = self.pool_2(conv2_1, training=training)
        
        # Output 1
        up1_2 = self.up_conv_1_2(conv2_1, training=training)
        conv1_2 = tf.concat([up1_2, conv1_1], axis=-1)
        conv1_2 = self.conv_1_2(conv1_2, training=training)
        
        conv3_1 = self.conv_3_1(pool2, training=training)
        pool3 = self.pool_3(conv3_1, training=training)
        
        up2_2 = self.up_conv_2_2(conv3_1, training=training)
        conv2_2 = tf.concat([up2_2, conv2_1], axis=-1)
        conv2_2 = self.conv_2_2(conv2_2, training=training)
        
        # Output 2
        up1_3 = self.up_conv_1_3(conv2_2, training=training)
        conv1_3 = tf.concat([up1_3, conv1_1, conv1_2], axis=-1)
        conv1_3 = self.conv_1_3(conv1_3, training=training)
        
        conv4_1 = self.conv_4_1(pool3, training=training)
        pool4 = self.pool_4(conv4_1, training=training)
        
        up3_2 = self.up_conv_3_2(conv4_1, training=training)
        conv3_2 = tf.concat([up3_2, conv3_1], axis=-1)
        conv3_2 = self.conv_3_2(conv3_2, training=training)
        
        up2_3 = self.up_conv_2_3(conv3_2, training=training)
        conv2_3 = tf.concat([up2_3, conv2_1, conv2_2], axis=-1)
        conv2_3 = self.conv_2_3(conv2_3, training=training)
        
        # Output 3
        up1_4 = self.up_conv_1_4(conv2_3, training=training)
        conv1_4 = tf.concat([up1_4, conv1_1, conv1_2, conv1_3], axis=-1)
        conv1_4 = self.conv_1_4(conv1_4, training=training)
        
        conv5_1 = self.conv_5_1(pool4, training=training)
        
        up4_2 = self.up_conv_4_2(conv5_1, training=training)
        conv4_2 = tf.concat([up4_2, conv4_1], axis=-1)
        conv4_2 = self.conv_4_2(conv4_2, training=training)
        
        up3_3 = self.up_conv_3_3(conv4_2, training=training)
        conv3_3 = tf.concat([up3_3, conv3_1, conv3_2], axis=-1)
        conv3_3 = self.conv_3_3(conv3_3, training=training)
        
        up2_4 = self.up_conv_2_4(conv3_3, training=training)
        conv2_4 = tf.concat([up2_4, conv2_1, conv2_2, conv2_3], axis=-1)
        conv2_4 = self.conv_2_4(conv2_4, training=training)
        
        # Output 4
        up1_5 = self.up_conv_1_5(conv2_4, training=training)
        conv1_5 = tf.concat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
        conv1_5 = self.conv_1_5(conv1_5, training=training)
        
        logits1_2 = self.nest_conv1(conv1_2, training=training)
        logits1_3 = self.nest_conv2(conv1_3, training=training)
        logits1_4 = self.nest_conv3(conv1_4, training=training)
        logits1_5 = self.nest_conv4(conv1_5, training=training)
        
        # Softmax or Sigmoid
        output_1 = tf.nn.softmax(logits1_2) # tf.math.sigmoid
        output_2 = tf.nn.softmax(logits1_3)
        output_3 = tf.nn.softmax(logits1_4)
        output_4 = tf.nn.softmax(logits1_5)
        
        return output_1, output_2, output_3, output_4
    
if __name__ == '__main__':
    batch_size = 32
    height = 64
    width = 64
    channel = 3
    
    test_data = tf.random.normal([batch_size, height, width, channel])
    
    model = UnetPlusPlus()
    
    test_output1, test_output2, test_output3, test_output4 = model(test_data, training=False)
    
    print(test_output1.shape)
    print('Done !')
