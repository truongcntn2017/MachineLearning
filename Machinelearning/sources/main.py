import tf_api as tf
import numpy as np

# X = np.array([0.4])
# w = np.array([[0.4, 0.1, 0.2]])
# b = np.array([[0.4,0.1,0.2]]).T

# print(X.shape, w.shape)
# print(X.dot(w))

# create default graph
tf.Graph().as_default()

# construct computational graph by creating some nodes
a = tf.Constant(-2)
b = tf.Constant(5)
c = tf.Constant(-4)

sum = tf.add(a, b)
prod = tf.multiply(sum, c)
res = prod

session = tf.Session()
# run computational graph to compute the output for 'res'
out = session.run(res)
print(out)

out = session.backward(res)
print(out)



