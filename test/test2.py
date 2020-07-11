import tf_api as tf
import numpy as np

tf.Graph().as_default()

w0 = tf.Constant(2)
x0 = tf.Constant(-1)
w1 = tf.Constant(-3)
x1 = tf.Constant(-2)
w2 = tf.Constant(-3)

mul1 = tf.multiply(w0, x0)
mul2 = tf.multiply(w1, x1)

sum1 = tf.add(mul1, mul2)
sum2 = tf.add(sum1, w2)

res = tf.sigmoid(sum2)


session = tf.Session()
out = session.run(res)
print(out)

out = session.backward(res)
print(out)




