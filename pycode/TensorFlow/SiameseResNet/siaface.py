import tensorflow as tf
import tensorflow.keras.applications.ResNet50 as resnet50

class siafacenet:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 64*64])
        self.x2 = tf.placeholder(tf.float32, [None, 64*64])
        
        with tf.variable_scope("siaface") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variable()
            self.o2 = slef.network(self.x2)
        
        # 计算损失
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
    
    def network(self):
        model = resnet50
    
    def batch_norm_relu(inputs, )
        pass
