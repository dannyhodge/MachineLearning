print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from numpy import array
# The digits dataset
#digits = datasets.load_digits()

train_images = []
train_labels = []
for filename in glob.glob('train/*.png'): #assuming png
    
    firstChar = int(filename[6])
    train_labels.append(firstChar)
    #print(firstChar)
    im=Image.open(filename).convert('LA')
    im=array(im)
    train_images.append(im)


for images 


test_images = []
for filename in glob.glob('test/*.png'): #assuming png
    im=Image.open(filename).convert('LA')
   # print(im)
    test_images.append(im)


# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(train_images, train_labels))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(train_images)
data = train_images

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], train_labels[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = train_labels[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(train_images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()