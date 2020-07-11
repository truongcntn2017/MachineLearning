from sources import tf_api as tf
import numpy as np

tf.Graph().as_default()

a = tf.Constant(-2)
b = tf.Constant(5)
c = tf.Constant(-4)

sum = tf.add(a, b)
prod = tf.multiply(sum, c)
res = prod

session = tf.Session()
out = session.run(res)
print(out)

out = session.backward(res)
print(out)



