import numpy as np

class Graph():
  def __init__(self):
    self.operations = []
    self.placeholders = []
    self.variables = []
    self.constants = []

  def as_default(self):
    global _default_graph
    _default_graph = self

class Operation():
  def __init__(self, input_nodes=None):
    self.input_nodes = input_nodes
    self.output = None
    
    # Append operation to the list of operations of the default graph
    _default_graph.operations.append(self)

  def forward(self):
    pass

  def backward(self):
    pass

class BinaryOperation(Operation):
  def __init__(self, a, b):
    super().__init__([a, b])

## Binary operation area 
class add(BinaryOperation):
  """
  Computes a + b element-wise
  """
  def forward(self, a, b):
    return a + b

  def backward(self, upstream_grad):
    return upstream_grad, upstream_grad

class multiply(BinaryOperation):
  """
  Computes a * b, element-wise
  """
  def forward(self, a, b):
    return a * b

  def backward(self, upstream_grad):
    output = [self.input_nodes[0].output, self.input_nodes[1].output]
    return output[1]*upstream_grad, output[0]*upstream_grad

class divide(BinaryOperation):
  """
  Returns the true division of the inputs, element-wise
  """
  def forward(self, a, b):
    return np.true_divide(a, b)

  def backward(self, upstream_grad):
    output = [self.input_nodes[0].output, self.input_nodes[1].output]
    return upstream_grad/output[1], -output[0]*upstream_grad/(output[1]*output[1])

class minus(BinaryOperation):
  """
  Returns the true division of the inputs, element-wise
  """
  def forward(self, a, b):
    return a - b

  def backward(self, upstream_grad):
    return upstream_grad, - upstream_grad

# class Matmul(BinaryOperation):
#   """
#   Multiplies matrix a by matrix b, producing a * b
#   """
#   def forward(self, a, b):
#     return a.dot(b)

#   def backward(self, upstream_grad):
#     raise NotImplementedError

## Operator
class Power(BinaryOperation):
    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, upstream_grad):
        output = [self.input_nodes[0].output, self.input_nodes[1].output]
        return np.power(output[0], output[1]-1)*upstream_grad

class Operator(Operation):
  def __init__(self, a):
    super().__init__([a])

class exp(Operator):
    def forward(self, a):
        return np.exp(a)

    def backward(self, upstream_grad):
        return np.exp(self.output)*upstream_grad

class sigmoid(Operator):
    def forward(self, a):
        return 1 / (1 + np.exp(-a))

    def backward(self, upstream_grad):
        return [upstream_grad*((1 - 1/ (1 + np.exp(- self.output)))/(1 + np.exp(- self.output)))]

class sin(Operator):
    def forward(self, a):
        return np.sin(a)
    
    def backward(self, upstream_grad):
        output = [self.input_nodes[0].output]
        return [np.cos(output[0])*upstream_grad]


class cos(Operator):
    def forward(self, a):
        return np.cos(a)
    
    def backward(self, upstream_grad):
        output = [self.input_nodes[0].output]
        return [- np.sin(output[0])*upstream_grad]

class tan(Operator):
    def forward(self, a):
        return np.tan(a)

    def backward(self, upstream_grad):
        output = [self.input_nodes[0].output]
        return [1/np.power(np.cos(output[0]), 2)*upstream_grad]


class Placeholder():
  def __init__(self):
    self.value = None
    _default_graph.placeholders.append(self)

class Constant():
  def __init__(self, value=None):
    self.__value = value
    _default_graph.constants.append(self)

  @property
  def value(self):
    return self.__value

  @value.setter
  def value(self, value):
    raise ValueError("Cannot reassign value.")

class Variable():
  def __init__(self, initial_value=None):
    self.value = initial_value
    _default_graph.variables.append(self)

def topology_sort(operation):
    ordering = []
    visited_nodes = set()

    def recursive_helper(node):
      if isinstance(node, Operation):
        for input_node in node.input_nodes:
          if input_node not in visited_nodes:
            recursive_helper(input_node)

      visited_nodes.add(node)
      ordering.append(node)

    # start recursive depth-first search
    recursive_helper(operation)

    return ordering

class Session():
  def run(self, operation, feed_dict={}):
    nodes_sorted = topology_sort(operation)

    for node in nodes_sorted:
      if type(node) == Placeholder:
        node.output = feed_dict[node]
      elif type(node) == Variable or type(node) == Constant:
        node.output = node.value
      else:
        inputs = [node.output for node in node.input_nodes]
        node.output = node.forward(*inputs)

    return operation.output

  def backward(self, operation, feed_dict={}):
    res = []

    def recursive_helper_backward(node, upstream_grad):
      if type(node) == Placeholder:
        node.output = feed_dict[node]
      elif type(node) == Variable or type(node) == Constant:
        node.output = upstream_grad
        res.append(node.output)
      else:
        upstream_grad_list = node.backward(upstream_grad)
        # print(upstream_grad_list)
        for i in range(len(node.input_nodes)):
            recursive_helper_backward(node.input_nodes[i], upstream_grad_list[i])

    recursive_helper_backward(operation, 1.0)

    return res