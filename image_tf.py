import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import collections
import tensorflow as tf

x_train = []
x_test = []
y_train = []
y_test = []
inf = list(os.walk('3D'))
print(len(inf))
inf = inf[0][1]
i=0
angles= {}
angles_graph = {}
print(inf)
for folder in inf:
    n_mes = int(folder.split('_')[0])

    angles[n_mes] = folder
angles = collections.OrderedDict(sorted(angles.items()))
print(dict(angles))

for n_folder in list(angles)[:10]:

    folder = angles[n_folder]
    print(folder)
    i+=1
    files = list(os.listdir('3D' + '\\' + folder))
    # Train

    for file in files[:13]:
        path = '3D' + '\\' + folder + '\\' + file
        df = pd.read_csv(path, sep=';', header=None)
        A =np.array(df.values[200:300, 0:-1], copy=True)# Create new copy with no vinculation

        # Select portion of picture with no ropes

        # Create array where temperatures will be stored
        temp = []
        for a in range(220,450,2):
            # Get a column of pixels
            col = A[:,a]
            # Get the averate of the temperatures on the column to avoid noise
            mean = np.mean(col)

            # Append average
            temp.append(mean)
        mx = max(temp)
        mn = min(temp)
        for t in range(0,len(temp)):
            temp[t] =(temp[t]-mn)/(mx-mn)

        x_train.append(np.array(temp))
        y_train.append(n_folder)
    angles_graph[n_folder] = temp

    # Test
    for file in files[13:]:
        path = '3D' + '\\' + folder + '\\' + file
        df = pd.read_csv(path, sep=';', header=None)
        A =np.array(df.values[200:300, 0:-1], copy=True)
        # Select portion of picture with no ropes

        # Create array where temperatures will be stored
        temp = []
        for a in range(220,450,2):
            # Get a column of pixels
            col = A[:, a]
            # Get the averate of the temperatures on the column to avoid noise
            mean = np.mean(col)

            # Append average
            temp.append(mean)
        temp = np.array(temp)
        mx = max(temp)
        mn = min(temp)
        for t in range(0,len(temp)):
            temp[t] =(temp[t]-mn)/(mx-mn)

        x_test.append(np.array(temp))
        y_test.append(n_folder)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
print(y_train)
y_test = np.array(y_test)
print(x_test.shape)
randomize = np.arange(len(x_train))
np.random.shuffle(randomize)
y_train = y_train[randomize]
x_train = x_train[randomize]
randomize = np.arange(len(x_test))
np.random.shuffle(randomize)
y_test = y_test[randomize]
x_test = x_test[randomize]
print(x_train[0])
print(y_train[0])


print(x_train[0])


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1000,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(i+1,activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30)

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)
#model.save('tf_number_predictor.model')
predictions = model.predict([x_test])
print(predictions[0])
for i in range(0,100):

    fig = plt.figure()
    fig.suptitle("Results", fontsize=16)
    ax = plt.subplot("311")
    ax.set_title("Test"+ angles[y_test[i]])
    ax.plot(x_test[i])

    ax = plt.subplot("312")
    ax.set_title("Prediction" + angles[np.argmax(predictions[i])])
    ax.plot(angles_graph[np.argmax(predictions[i])])

    ax = plt.subplot("313")
    ax.set_title("Real" + angles[y_test[i]])
    ax.plot(angles_graph[y_test[i]])

    plt.show()


