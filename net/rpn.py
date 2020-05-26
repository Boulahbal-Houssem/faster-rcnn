import tensorflow as tf
from tensorflow.keras.layers import *

class RPN(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.intermediate = Conv2D(filters=512,kernel_size=[3,3])
        self.cls_conv = Conv2D(filters=2,kernel_size=[1,1])
        self.box_reg_conv = Conv2D(filters=4,kernel_size=[1,1])       
    def call(self, input_tensor, training=False):
        x = self.intermediate(input_tensor)
        out_cls = self.cls_conv(x)
        out_box_reg = self.box_reg_conv(x)
        return [out_cls,out_box_reg]

if __name__ == '__main__':
    inputs = Input(shape=(222,222,3))
    rpn_out = RPN()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=rpn_out, name="RPN TEST")
    model.summary()
