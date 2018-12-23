#!/usr/bin/env python
# coding: utf-8

# ### Write TensorFlow code to provide the solutions to the following simple problems: 

# 1. Declare two constant tensors that have the values of 15 and 45. Add these two tensors and print out the results. 

# In[4]:


import tensorflow as tf

a= tf.constant(15)
b= tf.constant(45)
addOp_1=a+b
with tf.Session() as sess:
    result=sess.run(addOp_1)
    print (result)


# 2. Declare two variable tensors, a and b, that are initialized with scalar values of 2.75 and 8.5. Find their product and print out the result.

# In[2]:


import tensorflow as tf

# Declare variables
a=tf.Variable(2.75,tf.float32)
b=tf.Variable(8.5,tf.float32)

# Initialize all variables
init_all_op = tf.global_variables_initializer()

# This statement performs the operation to initialize and prints the result out
with tf.Session() as sess:
    sess.run(init_all_op)
    results=sess.run(a*b)
    print (results)


# 3. Create a constant tensor that is a matrix of the shape (8, 8). The matrix is initialized with all ones (1). Create constant variable tensor that is also a matrix of the shape (8, 8) and initialized with random values between 0 and 99. Add these two matrices and display the results.

# In[3]:



# Import tensorflow and numpy libraries
import tensorflow as tf
import numpy as np

# declare a constant matrix with the shape (8,8) and all ones and display the properties of the matrix
x = tf.constant(1.0, shape=[8, 8])

# Create a random numpy matrix with a range 0,99 and size (8,8)
data = np.random.randint(low=0,high=100,size=(8,8))

# convert numpy to tensor and display the properties of the matrix
y = tf.convert_to_tensor(data,dtype=tf.float32)

# Create a Python list and run this in the tensorflow session

list_ops=[x,y]
with tf.Session()as sess:
    for op in list_ops:
        print(sess.run(op))
        print("\n")


# Initialize all the variable
init_all_op = tf.global_variables_initializer()

#This statement performs the operation to initialize, calculate the result and prints it

with tf.Session() as sess:
    sess.run(init_all_op)
    results=sess.run(x+y)
    print ("Final Result :", results)


# 4. Create two placeholders: x and y - that are both scalars of 32-bit floats. Assign 5.25 to x and 12.6 to y, multiply them together, and print out the results.
# 

# In[5]:


import tensorflow as tf
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)


# feed the scalars 5.25 and 12.6 into the placeholders x and y respectively
# then print it out

with tf.Session()as sess:
    resultx=sess.run(x,feed_dict={x: [5.25]})
    resulty=sess.run(y,feed_dict={y: [12.6]})
    prodxy=resultx*resulty
    print (prodxy)


# 5. Create one placeholder: z - that is a 2D-array that can have any shape (shape = None). Feed this vector [1, 3, 5, 7, 9] into z and multiply it by 3. Display the results.
# 

# In[6]:


z=tf.placeholder(tf.float32,shape=[None])
with tf.Session()as sess:
    z_data=[1,3,5,7,9]
    result=sess.run(z,feed_dict={z:z_data})
    result3=result*3
    print (result3)


# In[ ]:





# In[ ]:




