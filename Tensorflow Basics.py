#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# ## Hello Tensorflow World

# In[2]:


# First, declare two string constant tensors

hello =tf.constant("Hello")
world=tf.constant("World")


# In[3]:


type(hello)


# In[4]:


print(hello)


# In[6]:


# If we want to concatenate these two tensors of strings
# Must run the operation in a TF session to get the result

with tf.Session() as sess:
    result=sess.run(hello+world)


# In[7]:


# Then print the result

print(result)


# ## Add two constanr integer Tensors

# In[9]:


import tensorflow as tf

a= tf.constant(10)
b= tf.constant(20)


# In[10]:


type(a)


# In[11]:


# the result is a tensor
a+b


# In[12]:


# addOp_1 is a tensor

addOp_1=a+b

addOp_1


# In[15]:


# if we want to actually perform the operation
# Must run the operation in a TF session to get the real result

with tf.Session() as sess:
    result=sess.run(addOp_1)
    print (result)    
    


# ## Matrix Tensors

# In[16]:


import tensorflow as tf
const=tf.constant(10)
int_mat=tf.fill((3,3),10)


# In[17]:


myzeros=tf.zeros((3,3))
myones=tf.ones((3,3))


# In[18]:


# This is a matrix of (3,3) of random values following the normal dist

myrandn=tf.random_normal((3,3),mean=0,stddev=1.0)


# In[19]:


# This is a matrix of (3,3) of random values between 0 and 1 following the normal dist

myrandu=tf.random_uniform((3,3),minval=0,maxval=1)


# In[20]:


myzeros


# In[21]:


# Declare a Python list consists of all the above constant and matrices
list_ops=[const,int_mat,myzeros,myones,myrandn,myrandu]


# In[22]:


# print out the values of these constants
# We have to run the in a tensorFlow session

with tf.Session()as sess:
    for op in list_ops:
        print(sess.run(op))
        print("\n")


# ## Use op.eval()

# In[24]:


import tensorflow as tf

# Running an operation: can use op.eval() instead of tf.Session().run()

with tf.Session()as sess:
    for op in list_ops:
        print(op.eval())
        print("\n")


# ## Shape of Tensors

# In[25]:


import tensorflow as tf


# In[26]:


# Declare a constant tensor that is a metrix of two columns
# First row: 1,2;2nd row:3,4

# VIP notes:

# A matrix :[]
# ROWS: --> each row inside this matrix is an embedded array:
# 1 row : [[]]
# 2 rows : [[],[]]

# Columns : --> Each column is one value inside the row array
# --> 2 rows, 1 col [[1],[2]]

#--> 2 rows, 2 cols [[1,3],[2,4]]

# --> 2 rows, 3 cols [[1,3,5],[2,4,6]]

a=tf.constant([[1,2],[3,4]])


# In[27]:


a


# In[29]:


a.get_shape()


# In[30]:


# Display the info of the tensor a

print(a)


# ## Another example of matrices operations in TensorFlow

# In[31]:


import tensorflow as tf


# In[32]:


# declare a constanr tensor that is a matrix of 2 rows an 2 cols
# first row :1,2; 2nd row: 3,4

a=tf.constant([[1,2],
               [3,4]])


# In[37]:


# declare a constant tensor
# -> matrix of 2 rows and 1 col = a vector

b=tf.constant([ [10],[100] ])


# In[38]:


b


# In[39]:


b.get_shape()


# In[40]:


# Perform the matrix on a and b
mat_product=tf.matmul(a,b)


# In[41]:


# display the info about the matrix product
# it should be also a matrix tensor with the shape [2,1]of the data type int

mat_product


# In[42]:


# display thw result
# must run it in a TF session

# first : run the operation
with tf.Session() as sess:
    results=sess.run(mat_product)
    print(results)


# ## TensorFlow: Variables

# ### Declare tf.Variable()

# In[44]:


import tensorflow as tf

aTensor=tf.random_uniform((3,3),0,1)

aTensor


# In[45]:


a_tf_var=tf.Variable(initial_value=aTensor)


# In[46]:


a_tf_var


# In[48]:


print(a_tf_var)


# In[49]:


# Cause an error: Variable must be initialized first
"""
with tf.Session()as sess:
    result=sess.run(a_tf_var)
    
"""    
  


# ### Initialize a tf.Variable

# In[52]:


# Get initializer, the operation to initialize the variable

initVar=tf.global_variables_initializer()

initVar


# In[55]:


# This statement performs the oepration to initialize the variable

with tf.Session()as sess:
    sess.run(initVar)
    results=sess.run(a_tf_var)
    print (results)


# ### Another way

# In[56]:


with tf.Session()as sess:
    sess.run(initVar)
    print(sess.run(a_tf_var))
   


# ## Tensor Flow: Placeholders & feed_dict

# In[58]:


import tensorflow as tf
x1=tf.placeholder(tf.float32)


# In[61]:


# feed the scalar 111 into the placeholder x1
# then print it out

with tf.Session()as sess:
    result=sess.run(x1,feed_dict={x1: [111]})
    print (result)


# In[64]:


x2=tf.placeholder(tf.float32,None)
y=x2*2


# In[65]:


# Run  a tf.Session(), feed data into tf.placeholder and print the results

with tf.Session()as sess:
    result=sess.run(y,feed_dict={x2:[1,2,3]})
    print(result)


# ### Multi-dimensional placeholders

# In[68]:


# Parameters: shape=[None, 3]
# -> The placeholder x3 has a shape of Nx3 matrix
#-> where N can be any number => 1

x3=tf.placeholder(tf.float32,shape=[None,3])
y=x3*2


# In[70]:


with tf.Session()as sess:
    x_data=[[1,2,3],
           [4,5,6],]
    result=sess.run(y,feed_dict={x3:x_data})
    print (result)


# ## tf.Graph() and tf.Session

# ### Construct Dataflow graph of the computation
# 
# 

# In[71]:


import tensorflow as tf
# Declare two tf.constanr: n1=1, n2=2

n1=tf.constant(1)
n2=tf.constant(2)


# In[75]:


# Construct a data graph of the computation
# This expression represents a graph
# ->Two input nodes: n1 and n2
# -> One operation: addition
# -> Output: n3

n3=n1+n2

# n3 is a tensor output(type: int32) of the operation +

n3


    


# In[76]:


print (n3)
    


# In[79]:


#Run a tf.Session() to execute the Computation
with tf.Session() as sess:
    result=sess.run(n3)
    print(result)


# ### Default Graphs

# In[80]:


print(tf.get_default_graph())


# In[82]:


g=tf.Graph()


# In[83]:


g


# In[84]:


# Display the info of the graph

print(g)


# In[85]:


graph_one=tf.get_default_graph()


# In[86]:


print(graph_one)


# In[87]:


graph_two=tf.Graph()


# In[88]:


print(graph_two)


# In[89]:


# swt the graph_two as the default graph temporarily
# print out the address of the default gtaph

with graph_two.as_default():
    print(graph_two)


# In[91]:


with graph_two.as_default():
    print(graph_two is tf.get_default_graph())


# In[92]:


# graph_two is still a normal graph

print(graph_two is tf.get_default_graph())


# In[ ]:




