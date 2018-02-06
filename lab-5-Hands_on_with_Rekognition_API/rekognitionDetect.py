#!/usr/bin/env python

import sys
import boto3
import pprint

pp = pprint.PrettyPrinter(indent=4)

defaultRegion = 'eu-west-1'
defaultUrl = 'https://rekognition.eu-west-1.amazonaws.com'

def connectToRekognitionService(regionName=defaultRegion, endpointUrl=defaultUrl):
    return boto3.client('rekognition', region_name=regionName, endpoint_url=endpointUrl)

def detectFaces(rekognition, imageBucket, imageFilename, attributes='ALL'):
    resp = rekognition.detect_faces(
            Image = {"S3Object" : {'Bucket' : imageBucket, 'Name' : imageFilename}},
            Attributes=[attributes])
    return resp['FaceDetails']

def compareFaces(rekognition, imageBucket, imageSourceFilename, imageTargetFilename):
    resp = rekognition.compare_faces(
            SourceImage = {"S3Object" : {'Bucket' : imageBucket, 'Name' : imageSourceFilename}},
            TargetImage = {"S3Object" : {'Bucket' : imageBucket, 'Name' : imageTargetFilename}})
    return resp['FaceMatches']

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
    pp.pprint(face)
#    if (face['Beard']['Value']):
#        print ('Beard')
#    if (face['Mustache']['Value']):
#        print ('Mustache')
#    if (face['Eyeglasses']['Value']):
#        print ('Eyeglasses')
#    if (face['Sunglasses']['Value']):
#        print ('Sunglasses')
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

