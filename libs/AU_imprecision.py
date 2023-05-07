import tensorflow as tf

""" If the label is one-hot label, it must be converted to a numerical label """
def average_utility(utility_matrix, inputs, labels, act_set):
    utility = 0
    # print(len(inputs))
    for i in range(len(inputs)):
      x = inputs[i]
      y = labels[i]
      utility += utility_matrix[x,y]
    # print(utility)
    average_utility = utility/len(inputs)
    return average_utility
