# Hands on with Apache MXNet

## Requirements

- AWS Account
- Basic understanding of ML and NDArrays
- SSH Client and key
- Linux Skills

## Credits

This workshop is adapted from Julien Simons blog posts: [https://medium.com/@julsimon/getting-started-with-deep-learning-and-apache-mxnet-34a978a854b4](https://medium.com/@julsimon/getting-started-with-deep-learning-and-apache-mxnet-34a978a854b4)

## Workshop Overview

In this workshop we are going to take a look at running Apache MXNet on the Amazon Linux Deep Learning AMI. We'll take a pre-trained image reckognition model and use this to predict the contents of images we feed into the model.

## Running MXNet on AWS

AWS provides you with the Deep Learning AMI, available both for Amazon Linux and Ubuntu. This AMI comes pre-installed with many Deep Learning frameworks (MXNet included), as well as all the Nvidia tools and more. No plumbing needed.

```
====================================================================
       __|  __|_  )
       _|  (     /   Deep Learning AMI for Amazon Linux
      ___|\___|___|
====================================================================

[ec2-user@ip-172-31-42-173 ~]$ nvidia-smi -L
GPU 0: GRID K520 (UUID: GPU-d470337d-b59b-ca2a-fe6d-718f0faf2153)

[ec2-user@ip-172-31-42-173 ~]$ source activate mxnet_p27

[ec2-user@ip-172-31-42-173 ~]$ python
>>> import mxnet as mx
>>> mx.__version__
'1.0.0'
```

You can run this AMI either on a standard instance or on a GPU instance. If you want to train a model and don’t have a NVidia GPU on your machine your most inexpensive option will be to use a g2.2xlarge instance at $0.65 per hour.

However in these labs we are using pre-trained models for speed so a standard instance of Amazon Linux is fine. This will allow us to get going with the lab without installing any special tools as the Deep Learning AMI comes with those pre-baked. Just remeber you need to source the environment after you SSH into the instance you create. In our case we want to use MXNet and python 2:

```bash
source activate mxnet_p27
```

## Using a pre-trained model

### The MXNet model zoo

In this first part of the lab we are going to recognising images with Inception v3, published in December 2015, Inception v3 is an evolution of the GoogleNet model (which won the 2014 ImageNet challenge). We won’t go into the details of the research paper, but paraphrasing its conclusion, Inception v3 is 15–25% more accurate than the best models available at the time, while being six times cheaper computationally and using at least five times less parameters (i.e. less RAM is required to use the model).

The model zoo is a collection of pre-trained models ready for use. You’ll find the model definition, the model parameters (i.e. the neuron weights) and instructions.

Let’s download the definition and the parameters. Feel free to open the first file you’ll see the definition of all the layers. The second one is a binary file, so don’t try and open that.

```bash
$ wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json

$ wget -O Inception-BN-0000.params http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params
```

Since this model has been trained on the ImageNet data set, we also need to download the corresponding list of image categories which contains the 1000 categories, that way we can see the human readable prediction output. You can take a look at this file also.

```bash
$ wget http://data.dmlc.ml/models/imagenet/synset.txt

$ wc -l synset.txt
    1000 synset.txt

$ head -5 synset.txt
n01440764 tench, Tinca tinca
n01443537 goldfish, Carassius auratus
n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
n01491361 tiger shark, Galeocerdo cuvieri
n01494475 hammerhead, hammerhead shark
```

Now we are also going to need some sample images to test the model against. I'm going to suggest two images and if you are feeling adventurous, feel free to add some of your own images and have a look at the outputs

```bash
wget -O image0.jpeg https://cdn-images-1.medium.com/max/1600/1*sPdrfGtDd_6RQfYvD5qcyg.jpeg

wget -O image1.jpeg https://www.google.co.uk/imgres?imgurl=http%3A%2F%2Fkidszoo.org%2Fwp-content%2Fuploads%2F2015%2F02%2Fclownfish3-1500x630.jpg&imgrefurl=http%3A%2F%2Fkidszoo.org%2Four-animals%2Fclownfish%2F&docid=ck2VMetqCvgH7M&tbnid=1w7pqpZbQtPrLM%3A&vet=10ahUKEwiNvbDD0-vYAhXSFewKHQZiClMQMwjtASgjMCM..i&w=1500&h=630&safe=active&client=firefox-b-ab&bih=752&biw=1440&q=clown%20fish&ved=0ahUKEwiNvbDD0-vYAhXSFewKHQZiClMQMwjtASgjMCM&iact=mrc&uact=8
```

### Loading the model for use

Open your python shell,

```bash
python
>>>
```

Load the model from its saved state. MXNet calls this a checkpoint. In return, we get the input Symbol and the model parameters.

```python
import mxnet as mx

sym, arg_params, aux_params = mx.model.load_checkpoint('Inception-BN', 0)
```

Create a new Module and assign it the input Symbol. We could also use a context parameter indicating where we want to run the model. By default the module uses the value cpu(0), but we could also use gpu(0) to run this on a GPU.

```python
mod = mx.mod.Module(symbol=sym)
```

Bind the input Symbol to input data. We’ll call it ‘data’ because that’s its name in the input layer of the network (look at the first few lines of the JSON file).

Define the shape of ‘data’ as 1 x 3 x 224 x 224.

```python
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
```

‘224 x 224’ is the image resolution, that’s how the model was trained. ‘3’ is the number of channels : red, green and blue (in this order). ‘1’ is the batch size: we’ll predict one image at a time.

set the model parameters.

```python
mod.set_params(arg_params, aux_params)
```

That’s all it takes. Four lines of code! Now it’s time to push some data in there and see what happens. 

### Data preparation

Before we get some predictions out of our model we'll need to prep the data (the images you downloaded)

Remember that the model expects a 4-dimension NDArray holding the red, green and blue channels of a single 224 x 224 image. We’re going to use the popular OpenCV library to build this NDArray from our input image. This is already installed on the Amazon Deep Learning AMI.

First lets load some libaries we'll need.

```python
import numpy as np
import cv2
```

Now we read the image, this will return a numpy array shaped as (image height, image width, 3), with the three channels in BGR order (blue, green and red).

```python
img = cv2.imread('<YOUR_IMAGE_FILE_NAME>')
```

Let’s convert the image to RGB, so we have the correct order (RGB) for the pre-trained model we are using.

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

Now resize the image to 224 x 224.

```python
img = cv2.resize(img, (224, 224,))
```

reshape the array from (image height, image width, 3) to (3, image height, image width).

```python
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
```

Add a fourth dimension and build the NDArray

```python
img = img[np.newaxis, :]
array = mx.nd.array(img)

>>> print array.shape
(1L, 3L, 224L, 224L)
```

Here’s our input picture.

![Input picture 448x336 (Source: metaltraveller.com)](image0.jpeg)

Its imput size is 448x336 and its in full colour. Remember our model needs images of 224x224 and in RGB.

Once processed, this picture has been resized and split into RGB channels stored in array[0] 

![array[0][0] : 224x224 red channel](image0-red.jpeg)

![array[0][1] : 224x224 green channel](image0-green.jpeg)

![array[0][2] : 224x224 blue channel](image0-blue.jpeg)

If batch size was higher than 1, then we would have a second image in array[1], a third in array[2] and so on.

Now your data is prepared let’s predict!

### Predicting

Normally we'd use a module object and we must feed data to a model in batches, the common way to do this is to use a data iterator (specifically, we used an NDArrayIter object).

Here, we’d like to predict a single image, so although we could use data iterator, it’d probably be overkill. Instead, we’re going to create a named tuple, called Batch, which will act as a fake iterator by returning our input NDArray when its data attribute is referenced.

```python
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
```

Now we can pass this “batch” to the model and let it predict.

```python
mod.forward(Batch([array]))
```

The model will output an NDArray holding the 1000 probabilities, corresponding to the 1000 categories. It has only one line since batch size is equal to 1.

```python
prob = mod.get_outputs()[0].asnumpy()

>>> prob.shape
(1, 1000)
```

Let’s turn this into an array with squeeze(). Then, using argsort(), we’re creating a second array holding the index of these probabilities sorted in descending order.

```python
prob = np.squeeze(prob)

>>> prob.shape
(1000,)
>> prob
[  4.14978594e-08   1.31608676e-05   2.51907986e-05   2.24045834e-05
   2.30327873e-06   3.40798979e-05   7.41563645e-06   3.04062659e-08 etc.

sortedprob = np.argsort(prob)[::-1]

>> sortedprob.shape
(1000,)
```

According to the model, the most likely category for this picture is #546 (if you are using image0.jpeg), with a probability of 58%.

```python
>> sortedprob
[546 819 862 818 542 402 650 420 983 632 733 644 513 875 776 917 795
etc.
>> prob[546]
0.58039135
```

Let’s find the name of this category. Using the synset.txt file, we can build a list of categories and find the one at index 546.

```python
synsetfile = open('synset.txt', 'r')
categorylist = []
for line in synsetfile:
  categorylist.append(line.rstrip())

>>> categorylist[546]
'n03272010 electric guitar'
```

The model has correctly identified there is an electric guitar in the image, pretty impressive.

What about the second highest category?

```python
>>> prob[819]
0.27168664
>>> categorylist[819]
'n04296562 stage'
```

Now you know how to use a pre-trained, state of the art model for image classification. All it took was a few lines of code and the rest was just data preparation. You can now try this with the other images (image1.jpeg and your own images) by starting at the data preparation stage again.


