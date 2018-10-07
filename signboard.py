import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.data
import skimage.transform
from PIL import Image
NUM_CLASSES=43
path=''

def get_data(path):
    images=[]
    labels=[]
    dummy_file=open('train.p','rb')
    file=pickle.load(dummy_file)
    feature=file['features']
    label=file['labels']
   
        
    #one hot encoding
    # one=np.zeros((label.shape[0],NUM_CLASSES))
    # for i, hot in enumerate(one):
    #     hot[label[i]]=1.
    # print(one[4700])
    #  print(label[4700])
    unique_labels=np.unique(label)
    labels=list(label)
    dummy_file.close()
    print("Unique label: {0} and Total Images: {1}".format(len(unique_labels),len(images)))
    return feature,labels

def display_images_and_label(images,labels):
    unique=set(label)
    image_list=[]
    plt.figure(figsize=(15,15))
    i=1
    for f in range(len(images)):
        img=Image.fromarray(images[f])
        #img.save("file"+ str(f) +".jpeg")
        image_list.append(img)
    for l in unique:
        image=image_list[label.index(l)]
        plt.subplot(6,8,i)
        plt.axis('off')
        plt.title("Label: {0} ({1})".format(l,label.count(l)))
        i+=1
        _=plt.imshow(image)
    plt.show()

    
images,label=get_data(path)
display_images_and_label(images,label)

labels_a=np.array(label)
print("labels: ", labels_a.shape, "\nimages: ", images.shape)

graph=tf.Graph()
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph=tf.placeholder(tf.float32,[None,32,32,3])
    labels_ph=tf.placeholder(tf.int32,[None])
    
    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat=tf.contrib.layers.flatten(images_ph)
    print(images_flat)
    
    # Fully connected layer. 
    # Generates logits of size [None, 62]
    logits=tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)
    print(logits)
    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)
    print(predicted_labels)
    
    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
    print(loss)

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])

for i in range(201):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)

# Pick 10 random images
sample_indexes = random.sample(range(len(images)), 10)
sample_images = [images[i] for i in sample_indexes]
sample_labels = [label[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])











