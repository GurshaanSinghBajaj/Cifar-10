
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import keras


# In[ ]:


import cifar10
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


cifar10.data_path = "data/CIFAR-10/"


# In[5]:


cifar10.maybe_download_and_extract()


# In[6]:


class_names = cifar10.load_class_names()
class_names


# In[7]:


images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


# In[8]:


print(images_train)


# In[9]:


print(cls_train)


# In[10]:


images_train.shape


# In[11]:


images_test.shape


# In[12]:


x_train = images_train.reshape(images_train.shape[0],-1)
x_test = images_test.reshape(images_test.shape[0],-1)
print(x_train.shape)
print(x_test.shape)


# In[ ]:


y_train = cls_train
y_test = cls_test


# In[14]:


fig = plt.figure(figsize=(8,8))
for i in range(64):
    ax = fig.add_subplot(8,8,i+1)
    ax.imshow(images_train[i], cmap = plt.cm.bone)
plt.show()


# In[58]:


pca = PCA()
pca.fit(x_train)


# In[ ]:


pca.components_.shape


# In[20]:


k = 0
total = sum(pca.explained_variance_)
currentsum = 0
while currentsum/total < 0.99 :
    currentsum += pca.explained_variance_[k]
    k = k+1
print(k)    


# In[21]:


pca = PCA(n_components=k,whiten=True)
transformed_training_data = pca.fit_transform(x_train)
print(transformed_training_data.shape)


# In[22]:


approx_training_data = pca.inverse_transform(transformed_training_data)
print(approx_training_data.shape)


# In[ ]:


approx_data = approx_training_data.reshape((50000,32,32,3))


# In[24]:


fig = plt.figure(figsize=(8,8))
for i in range(64):
    ax = fig.add_subplot(8,8,i+1)
    ax.imshow(approx_data[i], cmap = plt.cm.bone)
plt.show()


# In[ ]:


transformed_testing_data = pca.transform(x_test)


# In[32]:


rf = RandomForestClassifier()
rf.fit(transformed_training_data,y_train)
y_predict_rf = rf.predict(transformed_testing_data)
print(classification_report(y_test,y_predict_rf))
print(confusion_matrix(y_test,y_predict_rf))
print(accuracy_score(y_test,y_predict_rf))


# In[33]:


knn = KNeighborsClassifier()
knn.fit(transformed_training_data,y_train)
y_predict_knn = knn.predict(transformed_testing_data)
print(classification_report(y_test,y_predict_knn))
print(confusion_matrix(y_test,y_predict_knn))
print(accuracy_score(y_test,y_predict_knn))


# In[34]:


lr = LogisticRegression()
lr.fit(transformed_training_data,y_train)
y_predict_lr = lr.predict(transformed_testing_data)
print(classification_report(y_test,y_predict_lr))
print(confusion_matrix(y_test,y_predict_lr))
print(accuracy_score(y_test,y_predict_lr))


# In[35]:


xgb = XGBClassifier()
xgb.fit(transformed_training_data,y_train)
y_predict_xgb = xgb.predict(transformed_testing_data)
print(classification_report(y_test,y_predict_xgb))
print(confusion_matrix(y_test,y_predict_xgb))
print(accuracy_score(y_test,y_predict_xgb))


# In[ ]:


svc = svm.SVC()
svc.fit(transformed_training_data,y_train)
y_predict_svc = svc.predict(transformed_testing_data)
print(classification_report(y_test,y_predict_svc))
print(confusion_matrix(y_test,y_predict_svc))
print(accuracy_score(y_test,y_predict_svc))


# In[ ]:


import tensorflow as tf


# In[ ]:


n_input = 658
n_hidden_1 = 300
n_hidden_2 = 300
n_classes = 10

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[28]:


from numpy import array
from numpy import argmax
from keras.utils import to_categorical
data_train = y_train
data_test = y_test
data_train = array(data_train)
data_test = array(data_test)
encoded_train = to_categorical(data_train)
encoded_test = to_categorical(data_test)
print(encoded_train.shape)
print(encoded_train[0])


# In[ ]:


def forward_propagation(x, weights, biases):
    in_layer1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    out_layer1 = tf.nn.relu(in_layer1)
    
    in_layer2 = tf.add(tf.matmul(out_layer1, weights['h2']), biases['h2'])
    out_layer2 = tf.nn.relu(in_layer2)
    
    output = tf.add(tf.matmul(out_layer2, weights['out']), biases['out'])
    return output


# In[ ]:


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder(tf.int32, [None, n_classes])
pred = forward_propagation(x, weights, biases)


# In[31]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimize = optimizer.minimize(cost)


# In[ ]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[33]:


for i in range(100):
    c, _ = sess.run([cost,optimize], feed_dict={x:transformed_training_data , y:encoded_train})
    print(c)


# In[34]:


predictions = tf.argmax(pred, 1)
predictions_test = tf.argmax(pred, 1)
correct_labels = tf.argmax(y, 1)
correct_labels_test = tf.argmax(y, 1)
correct_predictions = tf.equal(predictions, correct_labels)
correct_predictions_test = tf.equal(predictions_test, correct_labels_test)
predictions, correct_predictions  = sess.run([predictions, correct_predictions], feed_dict={x:transformed_training_data , y:encoded_train})
print("Training score: ", correct_predictions.sum())
predictions_test, correct_predictions_test  = sess.run([predictions_test, correct_predictions_test], feed_dict={x:transformed_testing_data , y:encoded_test})
print("Testing score: ",correct_predictions_test.sum())


# In[ ]:


predictions_decoded = []
predictions = y_predict_svc
labels = []
for i in range(len(predictions)):
    labels.append(class_names[predictions[i]])
np.savetxt("predictions.csv", labels, fmt = '%s')


# In[15]:


print(images_train.shape)
print(images_test.shape)


# In[16]:


from keras.utils import to_categorical
y_train = to_categorical(cls_train)
y_test = to_categorical(cls_test)
print(y_train[56])
print(y_test[109])


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D


# In[ ]:


model = Sequential()
model.add(Conv2D(48, (3, 3), padding = 'same', input_shape = (32, 32, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding = 'same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(192, (3, 3) , padding = 'same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:


# Train the model
model.fit(images_train, y_train, batch_size=128, epochs = 200, validation_data = (images_test, y_test), verbose=2)


# In[30]:


scores = model.evaluate(images_test, y_test, batch_size=128, verbose=1)
print(scores[0])
print(scores[1])


# In[31]:


print(soultion[4])


# In[ ]:


from google.colab import files
predictions = soultion
labels = []
for i in range(len(predictions)):
    index = np.argmax(predictions[i])
    labels.append(class_names[index])
np.savetxt("prediction.csv", labels, fmt = '%s')
files.download('prediction.csv')


# In[ ]:


predictions = cls_test
labels = []
for i in range(len(predictions)):
    index = predictions[i]
    labels.append(class_names[index])
np.savetxt("prediction_1.csv", labels, fmt = '%s')
files.download('prediction_1.csv')

