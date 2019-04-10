import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from PIL import Image
import glob


#def getData(numinstancesperclass):
    #A=np.random.rand(numinstancesperclass,2)+1
    #B=np.random.rand(numinstancesperclass,2)+1*0.01
    #C=np.random.rand(numinstancesperclass,2)-1
    #A=np.append(A,np.ones([len(A),1])*1,1)
    #B=np.append(B,np.ones([len(B),1])*2,1)
    #C=np.append(C,np.ones([len(C),1])*3,1)
   # data=np.concatenate([A,B])
  #  data=np.concatenate([data,C])
    #return data


train_images = []
for filename in glob.glob('train/*.png'): #assuming png
    im=Image.open(filename)
    train_images.append(im)
    
    
#test_images = []
#for filename in glob.glob('test/*.png'): #assuming png
#    im=Image.open(filename)
#    test_images.append(im)

#import gzip
#f = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 50

#f.read(16)
#buf = f.read(image_size * image_size * num_images)
#data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#data = data.reshape(num_images, image_size, image_size, 1)

#import matplotlib.pyplot as plt
#image = np.asarray(data[3]).squeeze()
#plt.imshow(image)


#f = gzip.open('train-labels-idx1-ubyte.gz','r')
#for i in range(0,100):
   # f.read(8)
  #  buf = f.read(1 * 32)
  #  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  #  print(labels)
    
    
#train=getData(data)
#test=getData(100)
#classifier=neighbors.KNeighborsClassifier(3)
#classifier.fit(train[:,:2],train[:,2])
#predictions=classifier.predict(test[:,:2])
#print"Accuracy(%)=",accuracy_score(test[:,2],predictions)*100
#plt.figure(1)
#plt.subplot(131)
#plt.scatter(train[:,0],train[:,1],s=50,c=train[:,2],cmap=plt.cm.Paired)
#plt.title("TrainingData")
#plt.subplot(132)
#plt.scatter(test[:,0],test[:,1],s=50,c=predictions,cmap=plt.cm.Paired)
#plt.title("TestData")
#plt.subplot(133)
#plt.scatter(test[:,0],test[:,1],s=50,c=test[:,2],cmap=plt.cm.Paired)
#plt.title("Ground-TruthData")
#plt.show()
