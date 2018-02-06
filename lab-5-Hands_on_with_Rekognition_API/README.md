## Hands on with the Amazon Rekognition API

### Requirements

- AWS Account
- AWS CLI tools installed
- python + boto3 + some python skills

### Why Amazon Rekognition?

Earlier we looked at image recognition using Apache MXNet and we analysed an image of a rock guitarist. In order to do that we had to prepare the data, this takes time when dealing with thousands or millions of images. Rekognition does this for you automatically. The screen shot below shows the same image processed by rekognition.

![demo0.png](demo0.png)

As you can see AWS has already done the heavy lifting of data preparation for you. Also it extends much further tom object detection. It can detect faces, guess the age of the person, compare faces and even process video streams in the same way.

### Using Amazon Rekognition

To start with lets look at the AWS CLI for Rekognition, we'll want a few sample images, ones you can use easily are from this repository, but feel free to subsitute with your own.

Create a public S3 bucket and upload the sample images.

First of all lets scan a picture to find a face:

```bash
aws rekognition detect-faces --image "S3Object={Bucket="image-demo-lab", Name="ric_harvey.jpeg"}"
```
The output of this shows there is indeed a face detected, and you can see details on the Landmarks it used to detect the face and its confidence.

```json
{
    "FaceDetails": [
        {
            "BoundingBox": {
                "Width": 0.45778611302375793,
                "Height": 0.3050000071525574,
                "Left": 0.2739211916923523,
                "Top": 0.1693750023841858
            },
            "Landmarks": [
                {
                    "Type": "eyeLeft",
                    "X": 0.42861831188201904,
                    "Y": 0.28512176871299744
                },
                {
                    "Type": "eyeRight",
                    "X": 0.572612464427948,
                    "Y": 0.28403761982917786
                },
                {
                    "Type": "nose",
                    "X": 0.5114457607269287,
                    "Y": 0.3355158865451813
                },
                {
                    "Type": "mouthLeft",
                    "X": 0.44752469658851624,
                    "Y": 0.39533865451812744
                },
                {
                    "Type": "mouthRight",
                    "X": 0.5687991976737976,
                    "Y": 0.39315375685691833
                }
            ],
            "Pose": {
                "Roll": -1.3944040536880493,
                "Yaw": 4.856456279754639,
                "Pitch": 7.715858459472656
            },
            "Quality": {
                "Brightness": 33.98454666137695,
                "Sharpness": 99.9945297241211
            },
            "Confidence": 99.9999771118164
        }
    ],
    "OrientationCorrection": "ROTATE_0"
}
```
However in most cases you don't want to just find a face you want some information about that face, gender and defining features, maybe the sentiment. We can do this by calling detect-labels.

```bash
aws rekognition detect-labels --image "S3Object={Bucket="image-demo-lab", Name="ric_harvey.jpeg"}"
```

The resulting output determins I am indeed human.

```json
{
    "Labels": [
        {
            "Name": "Human",
            "Confidence": 99.31964874267578
        },
        {
            "Name": "People",
            "Confidence": 99.31964874267578
        },
        {
            "Name": "Person",
            "Confidence": 99.31964874267578
        },
        {
            "Name": "Man",
            "Confidence": 75.68144226074219
        },
        {
            "Name": "Face",
            "Confidence": 61.79734420776367
        },
        {
            "Name": "Portrait",
            "Confidence": 56.22377395629883
        },
        {
            "Name": "Dimples",
            "Confidence": 51.63166046142578
        }
    ],
    "OrientationCorrection": "ROTATE_0"
}
```

So if we can find faces in images and identify key objects about that face we should be able to compare faces and find the same person in multiple photos. Heres an example using compare-faces.

```bash
aws rekognition compare-faces --source-image '{"S3Object":{"Bucket":"image-demo-lab","Name":"ric_harvey.jpeg"}}' --target-image '{"S3Object":{"Bucket":"image-demo-lab","Name":"ric.jpg"}}'
```

In the first example we have a match.

```json
{
    "SourceImageFace": {
        "BoundingBox": {
            "Width": 0.45778611302375793,
            "Height": 0.3050000071525574,
            "Left": 0.2739211916923523,
            "Top": 0.1693750023841858
        },
        "Confidence": 99.9999771118164
    },
    "FaceMatches": [
        {
            "Similarity": 88.0,
            "Face": {
                "BoundingBox": {
                    "Width": 0.5841526985168457,
                    "Height": 0.5870445370674133,
                    "Left": 0.22267206013202667,
                    "Top": 0.1069982647895813
                },
                "Confidence": 100.0,
                "Landmarks": [
                    {
                        "Type": "eyeLeft",
                        "X": 0.4296815097332001,
                        "Y": 0.3233932554721832
                    },
                    {
                        "Type": "eyeRight",
                        "X": 0.6151635646820068,
                        "Y": 0.3312254250049591
                    },
                    {
                        "Type": "nose",
                        "X": 0.5194208025932312,
                        "Y": 0.4376829266548157
                    },
                    {
                        "Type": "mouthLeft",
                        "X": 0.4202229976654053,
                        "Y": 0.5286720395088196
                    },
                    {
                        "Type": "mouthRight",
                        "X": 0.6185688972473145,
                        "Y": 0.533376157283783
                    }
                ],
                "Pose": {
                    "Roll": 2.0403640270233154,
                    "Yaw": -0.9170446395874023,
                    "Pitch": 7.417059421539307
                },
                "Quality": {
                    "Brightness": 31.274736404418945,
                    "Sharpness": 99.99880981445312
                }
            }
        }
    ],
    "UnmatchedFaces": [],
    "SourceImageOrientationCorrection": "ROTATE_0",
    "TargetImageOrientationCorrection": "ROTATE_0"
}
```

However lets look at a picture with more than one persojn in it.

```bash
aws rekognition compare-faces --source-image '{"S3Object":{"Bucket":"image-demo-lab","Name":"ric_harvey.jpeg"}}' --target-image '{"S3Object":{"Bucket":"image-demo-lab","Name":"ric_crowd1.jpg"}}'
```

the output fromt his shows 1 matched face and several unmatched faces.

```json
{
    "SourceImageFace": {
        "BoundingBox": {
            "Width": 0.45778611302375793,
            "Height": 0.3050000071525574,
            "Left": 0.2739211916923523,
            "Top": 0.1693750023841858
        },
        "Confidence": 99.9999771118164
    },
    "FaceMatches": [
        {
            "Similarity": 90.0,
            "Face": {
                "BoundingBox": {
                    "Width": 0.22687500715255737,
                    "Height": 0.30250000953674316,
                    "Left": 0.2524999976158142,
                    "Top": 0.3891666531562805
                },
                "Confidence": 99.95853424072266,
                "Landmarks": [
                    {
                        "Type": "eyeLeft",
                        "X": 0.3539683222770691,
                        "Y": 0.4969490170478821
                    },
                    {
                        "Type": "eyeRight",
                        "X": 0.4162258207798004,
                        "Y": 0.5343121886253357
                    },
                    {
                        "Type": "nose",
                        "X": 0.3986102342605591,
                        "Y": 0.5785834193229675
                    },
                    {
                        "Type": "mouthLeft",
                        "X": 0.32881030440330505,
                        "Y": 0.616106390953064
                    },
                    {
                        "Type": "mouthRight",
                        "X": 0.37624654173851013,
                        "Y": 0.6475170850753784
                    }
                ],
                "Pose": {
                    "Roll": 21.32535171508789,
                    "Yaw": 33.83676528930664,
                    "Pitch": 12.081502914428711
                },
                "Quality": {
                    "Brightness": 22.441940307617188,
                    "Sharpness": 99.95819854736328
                }
            }
        }
    ],
    "UnmatchedFaces": [
        {
            "BoundingBox": {
                "Width": 0.3474999964237213,
                "Height": 0.4633333384990692,
                "Left": 0.71875,
                "Top": 0.5108333230018616
            },
            "Confidence": 99.98457336425781,
            "Landmarks": [
                {
                    "Type": "eyeLeft",
                    "X": 0.8321326971054077,
                    "Y": 0.694943368434906
                },
                {
                    "Type": "eyeRight",
                    "X": 0.9426370859146118,
                    "Y": 0.7011836767196655
                },
                {
                    "Type": "nose",
                    "X": 0.8815482258796692,
                    "Y": 0.767633855342865
                },
                {
                    "Type": "mouthLeft",
                    "X": 0.8442646265029907,
                    "Y": 0.8384411334991455
                },
                {
                    "Type": "mouthRight",
                    "X": 0.9513602256774902,
                    "Y": 0.8438958525657654
                }
            ],
            "Pose": {
                "Roll": 2.4253525733947754,
                "Yaw": -8.93567180633545,
                "Pitch": 5.809824466705322
            },
            "Quality": {
                "Brightness": 15.722269058227539,
                "Sharpness": 99.95819854736328
            }
        },
        {
            "BoundingBox": {
                "Width": 0.48750001192092896,
                "Height": 0.6499999761581421,
                "Left": -0.10125000029802322,
                "Top": 0.3499999940395355
            },
            "Confidence": 96.41242980957031,
            "Landmarks": [
                {
                    "Type": "eyeLeft",
                    "X": 0.08621729910373688,
                    "Y": 0.6087729334831238
                },
                {
                    "Type": "eyeRight",
                    "X": 0.23481208086013794,
                    "Y": 0.6346983313560486
                },
                {
                    "Type": "nose",
                    "X": 0.2088991105556488,
                    "Y": 0.7378349304199219
                },
                {
                    "Type": "mouthLeft",
                    "X": 0.08538326621055603,
                    "Y": 0.8273465037345886
                },
                {
                    "Type": "mouthRight",
                    "X": 0.21713919937610626,
                    "Y": 0.842521071434021
                }
            ],
            "Pose": {
                "Roll": 5.360462188720703,
                "Yaw": 30.094144821166992,
                "Pitch": 5.545968055725098
            },
            "Quality": {
                "Brightness": 14.225797653198242,
                "Sharpness": 99.99090576171875
            }
        },
        {
            "BoundingBox": {
                "Width": 0.18687500059604645,
                "Height": 0.24916666746139526,
                "Left": 0.6043750047683716,
                "Top": 0.34833332896232605
            },
            "Confidence": 99.91377258300781,
            "Landmarks": [
                {
                    "Type": "eyeLeft",
                    "X": 0.6566815972328186,
                    "Y": 0.4502323269844055
                },
                {
                    "Type": "eyeRight",
                    "X": 0.7224865555763245,
                    "Y": 0.43844547867774963
                },
                {
                    "Type": "nose",
                    "X": 0.6805922985076904,
                    "Y": 0.4983518421649933
                },
                {
                    "Type": "mouthLeft",
                    "X": 0.6659459471702576,
                    "Y": 0.532056987285614
                },
                {
                    "Type": "mouthRight",
                    "X": 0.7296943664550781,
                    "Y": 0.5243272185325623
                }
            ],
            "Pose": {
                "Roll": -7.166214466094971,
                "Yaw": -19.44483757019043,
                "Pitch": -3.4779880046844482
            },
            "Quality": {
                "Brightness": 30.67827606201172,
                "Sharpness": 99.884521484375
            }
        },
        {
            "BoundingBox": {
                "Width": 0.16062499582767487,
                "Height": 0.21416667103767395,
                "Left": 0.47062501311302185,
                "Top": 0.42750000953674316
            },
            "Confidence": 99.99253845214844,
            "Landmarks": [
                {
                    "Type": "eyeLeft",
                    "X": 0.5277988314628601,
                    "Y": 0.5116181969642639
                },
                {
                    "Type": "eyeRight",
                    "X": 0.5793790817260742,
                    "Y": 0.5157355070114136
                },
                {
                    "Type": "nose",
                    "X": 0.5589439868927002,
                    "Y": 0.5528774261474609
                },
                {
                    "Type": "mouthLeft",
                    "X": 0.5237618088722229,
                    "Y": 0.5906259417533875
                },
                {
                    "Type": "mouthRight",
                    "X": 0.5729554891586304,
                    "Y": 0.5956164598464966
                }
            ],
            "Pose": {
                "Roll": 2.085719585418701,
                "Yaw": 13.388442993164062,
                "Pitch": 8.00314712524414
            },
            "Quality": {
                "Brightness": 29.39847183227539,
                "Sharpness": 99.9305191040039
            }
        }
    ],
    "SourceImageOrientationCorrection": "ROTATE_0"
}
```

Repeating this with ric_crowd0.jpg will show no results.

### Doing this from python

Using the CLI is fine but if you want to embed this into you system you'll need to make these calls from code. We'll use Python to do this and we'll need boto3 installed for accessing the AWS API:

```bash
pip install boto3
```

__Note:__ OSX may need to run ```sudo -H pip install boto3```

#### Sample code

Lets look at some same code that allows you to detect faces and the labels for each face. Try running this on a few of the sample images.

TO TIDY UP AND ADD CLI OPTIONS!!!!!

```python
#!/usr/bin/env python

import sys
import boto3

defaultRegion = 'eu-west-1'
defaultUrl = 'https://rekognition.eu-west-1.amazonaws.com'

def connectToRekognitionService(regionName=defaultRegion, endpointUrl=defaultUrl):
    return boto3.client('rekognition', region_name=regionName, endpoint_url=endpointUrl)

def detectFaces(rekognition, imageBucket, imageFilename, attributes='ALL'):
    resp = rekognition.detect_faces(
            Image = {"S3Object" : {'Bucket' : imageBucket, 'Name' : imageFilename}},
            Attributes=[attributes])
    return resp['FaceDetails']

def detectLabels(rekognition, imageBucket, imageFilename, maxLabels=100, minConfidence=0):
    resp = rekognition.detect_labels(
        Image = {"S3Object" : {'Bucket' : imageBucket, 'Name' : imageFilename}},
        MaxLabels = maxLabels, MinConfidence = minConfidence)
    return resp['Labels']

def printFaceInformation(face, faceCounter):
    print('*** Face ' + str(faceCounter) + ' detected, confidence: ')+str(face['Confidence'])
    print('Gender: ')+face['Gender']['Value']
    # You need boto3>=1.4.4 for AgeRange
    print('Age: ')+str(face['AgeRange']['Low'])+"-"+str(face['AgeRange']['High'])
    if (face['Beard']['Value']):
        print ('Beard')
    if (face['Mustache']['Value']):
        print ('Mustache')
    if (face['Eyeglasses']['Value']):
        print ('Eyeglasses')
    if (face['Sunglasses']['Value']):
        print ('Sunglasses')
    for e in face['Emotions']:
        print e['Type']+' '+str(e['Confidence'])

def printLabelsInformation(labels):
    for l in labels:
        print('Label ' + l['Name'] + ', confidence: ' + str(l['Confidence']))

def usage():
    print('\nrekognitionDetect <S3BucketName> <image>\n')
    print('S3BucketName  : the S3 bucket where Rekognition will find the image')
    print('image         : the image to process')
    print('Output        :  labels & face information (stdout)\n')

if (len(sys.argv) != 3):
    usage()
    sys.exit()

imageBucket = str(sys.argv[1])
image       = str(sys.argv[2])

reko = connectToRekognitionService()

labels = detectLabels(reko, imageBucket, image, maxLabels=10, minConfidence=70.0)
printLabelsInformation(labels)

faceList = detectFaces(reko, imageBucket, image)
faceCounter = 0
for face in faceList:
    printFaceInformation(face, faceCounter)
    faceCounter=faceCounter+1

labelText = ''
for l in labels:
    if (l['Confidence'] > 80.0):
        labelText = labelText + l['Name'] + ", "
```

### Challenge 1 (Where's Ric)

Durring registration I took some photos and uploaded them to a public S3 bucket s3://image-demo-lab there is also an array of asorted images in the bucket. Your challenge is to:

- a) find how many pictures in the bucket contain a photo of me
- b) find your self in the images

You'll need to create a compare-faces function and also get a list of all the objects in the S3 bucket (warning they may not all be images!) Extra points for the fastest way of doing this.

#### Resources

[http://boto3.readthedocs.io/en/latest/reference/services/rekognition.html](http://boto3.readthedocs.io/en/latest/reference/services/rekognition.html)

Let an instructor know when you've completed this there may be a prize.