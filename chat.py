from gettext import translation
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer= LancasterStemmer()


import numpy
import tflearn 
import tensorflow as tf
import random
import json

with open("/home/sai/assistantAI/json/intents.json") as f:
    data = json.load(f)
# print(data["intents"])    

words=[]

labels=[]

docs=[]

docs_x=[]

docs_y=[]

for i in data["intents"]:
    for p in i["patterns"]:
        w=nltk.word_tokenize(p)
        words.extend(w)
        docs_x.append(w)
        docs_y.append(i["tag"])


    if i["tag"] not in labels:
        labels.append(i["tag"])

words=[stemmer.stem(i.lower()) for i in words if i != "?"]

Words=sorted(list(set(words)))

labels=sorted(labels)


# print(Words)


training=[]

output=[]


out_emty=[0 for i in  range(len(labels))]

# print(out_emty)
# print(docs_x)



for x,doc in enumerate(docs_x):
    bag=[]

    wrds=[stemmer.stem(w) for w in doc]

    for w in Words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row=out_emty[:]
    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row)

print(output_row)

#change into numpy arary


training = numpy.array(training)

output = numpy.array(output)




# net= tflearn.input_data(shape=[None,len(training[0])])

# net=tflearn.fully_connected(net,8)
# net=tflearn.fully_connected(net,8)

# net= tflearn.fully_connected(net,len(output[0]),activation="softmax")
# net=tflearn.regression(net)

# model =tflearn.DNN(net)

# model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
# model.save("chatbot.tflearn")
t=tf.expand_dims(training,axis=0)
print(t)

o=tf.expand_dims(output,axis=0)
print(o)
# print(training.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
            
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(t, o, epochs=10,step_p)
