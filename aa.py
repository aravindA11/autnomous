#!/usr/bin/env python
from threading import *
import rospy
import time
import serial
import math
import struct
import autocalc as c
import Queue
from jetson.msg import DriveNeuron
from sensor_msgs.msg import NavSatFix
from jetson.msg import ImuNeuron
from jetson.msg import AutoPointNeuron
from std_msgs.msg import String
from jetson.msg import *

import numpy as np
from sklearn.preprocessing import normalize
import cv2

pubDrive =  rospy.Publisher('DriveData',DriveNeuron, queue_size=10)
dataDrive = DriveNeuron()
"""cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,10)
cap.set(3,1280)
cap.set(4,480)
"""
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS,10)
width = 320
height = 240  #  240
cap.set(3,width*2) #1280
cap.set(4,height) #480

ratio = float(float(width)/640)**2
midLimit  = (ratio) * 20000
sideLimit = (ratio) * 5000





qlat=Queue.Queue(maxsize=10)
qlon=Queue.Queue(maxsize=10)
error = 20            #degrees
writeOffset = 0       #unknown
fwdConstant = 0.65    #unknown
linearConst = 50      #for linear mode
powerOffset = 1        #0.6
accuracy = 3        #metres
# l = -1
#dataDrive = DriveNeuron()
averages = 4
counter = averages/2   #midway

def fit(value):
    if(value>1):
        return 1.0
    elif (value<-1):
        return -1.0
    else:
        return value


def updateDest(destData):

    if destData.data[0] == -1:
        auto.l=-1
    else :
        #print("points stored")
        #for i in range(0,len(destData.data)):
         #  print(destData.data[i])

        qlon.queue.clear()
        qlat.queue.clear()
        for i in range(0, len(destData.data)/2):
          if (destData.data[i] != 0):
              qlat.put(destData.data[i])
        #      print("latitudes:",qlat.get())
        for i in range(len(destData.data)/2,len(destData.data)):
          if (destData.data[i] != 0):
              qlon.put(destData.data[i])
         #     print("Longitudes:",qlon.get())
        auto.destlat = qlat.get()
        auto.destlon = qlon.get()
        auto.l = qlon.qsize()
       # print(auto.destlat,auto.destlon,auto.l)
#initCount = 0
def detectObs():
    auto.initCount +=1
    if auto.initCount <5:
    	publishDrive(0,0)
    ret,img = cap.read()
    #cv2.imshow("img",img)
    imgL = img[0:height , 0:width]
    imgR = img[0:height, width:2*width]
    imgR = cv2.cvtColor (imgR, cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor (imgL, cv2.COLOR_BGR2GRAY)
    window_size = 7                                                #32            7
    left_matcher = cv2.StereoSGBM_create(minDisparity=0,numDisparities=32,blockSize=7,P1=8 * 3 * window_size *window_size,P2=32 * 3 * window_size*window_size,
        disp12MaxDiff=-25,uniquenessRatio=15,speckleWindowSize=0, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  #     .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)     # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    _,filteredImg= cv2.threshold(filteredImg, 127, 255, cv2.THRESH_BINARY)
    left  = filteredImg[0:int(0.7*height) ,15:80]     #set to 30:80 on 1280x640
    mid   = filteredImg[0:int(0.7*height), 80:width-50]
    right = filteredImg[0:int(0.7*height), width-50:width]

    regionMaskL  = (left==255)
    FilledAreaL  = np.sum(regionMaskL)
    regionMaskR  = (right==255)
    FilledAreaR  = np.sum(regionMaskR)
    regionMaskMid  = (mid==255)
    FilledAreaMid  = np.sum(regionMaskMid)
    print(FilledAreaMid)
    obsMid = 0
    obsL = 0
    obsR = 0
    if (FilledAreaMid >=midLimit):
        obsMid = 1
    if (FilledAreaR >=sideLimit):
        obsR = 1
    if (FilledAreaL >=sideLimit):
        obsL = 1

    #cv2.imshow("Mid",mid)
#    cv2.imshow("leftRaw",imgL)
#    cv2.imshow("rightRaw",imgR)

    return obsL,obsMid,obsR
obsMid = 0
obsL = 0
obsR = 0
def updateData(gpsData):
    auto.lon = gpsData.longitude
    auto.lat = gpsData.latitude

def updateComp(imuData):
    auto.heading = imuData.heading
    auto.heading = 180 + auto.heading
    auto.heading = auto.heading%360

def publishDrive(leftVal,rightVal):
    dataDrive = DriveNeuron()
    dataDrive.channel1 = float(leftVal)
    dataDrive.channel2 = float(rightVal)
    print ("VALUE  "+str(leftVal)+", "+str(rightVal))
    #print("--------------------_",dataDrive)
    pubDrive.publish(dataDrive)
    #print(auto.destlon,auto.destlat)
def fwd(difference):
    obsL,obsMid,obsR = detectObs()
    if (auto.count>0 and obsR and obsMid and obsL):
            difference=10
            auto.count-=1

    if difference>90:
        difference=90
    elif difference<-90:
        difference=-90
    sinVal = abs(math.sin(math.radians(difference))) 
    if difference >error and difference < -error:
        leftVal = fit(sinVal+0.15)
        rightVal = fit(-1*sinVal -0.15)                       #  3CHANGE OFFSETS
    elif difference <error and difference > -error:
        if(obsMid):
            difference=40
        elif(obsL):
            difference=30
        elif(obsR):
            difference=-30
        if(obsR or obsMid or obsL):
            fwd(difference)
            auto.count=50
        leftVal=(difference<0)and 1-(sinVal**powerOffset) or 1
        rightVal=(difference>0)and 1-(sinVal**powerOffset) or 1

    publishDrive(leftVal,rightVal)


"""
def fwd(difference):
    #print("in fwd function")
    if difference>90:
        difference=90
    elif difference<-90:
        difference=-90
    sinVal = abs(math.sin(math.radians(difference)))     #  why (difference - error?)  change this  .....
    #print ( str(difference)+"----------------" )
    if (difference <error and difference > -error):
        print("FORWard")
        obsL,obsMid,obsR = detectObs()
        print(obsL,obsMid,obsR,"----")
        #obsMid = 0
        if (obsMid == 0):                        #1 is fwd, 0 is left, 2 is right
            print("forward")
            leftVal=(difference<0)and 1-(sinVal**powerOffset) or 1
            rightVal=(difference>0)and 1-(sinVal**powerOffset) or 1
            publishDrive (leftVal,rightVal)

        else:                                    #fwd not clear

            diffPos = 1 + error
            diffNeg = -1 - error
            if (obsR == 0):
                if(difference>0):
                    fwd( diffPos )
                elif (difference <-10):
                    fwd( diffNeg )
            elif (obsL == 0):
                if(difference>10):
                    fwd( diffPos )
                elif (difference <0):
                    fwd( diffNeg )
            if (obsR==1 and obsL==1):
                if (difference>0 ):
                    fwd( diffPos )
                else : fwd( diffNeg )
    else:
        print("Turning")
        leftVal = sinVal+0.15
        rightVal = -1*sinVal -0.15                       #  3CHANGE OFFSETS
        leftVal = fit(leftVal)
        rightVal = fit(rightVal)
        publishDrive(leftVal,rightVal)
"""

class autoDriveData:
    def __init__(self):
        self.destHeading = 0
        self.heading = 0
        self.difference = 0
        self.distance = 0.0
        self.lat = -1.0
        self.lon = -1.0
        self.destlat = -1.0
        self.destlon = -1.0
        self.l =0
        self.obsPoint = 0
        self.initCount = 0
        self.count=0
	#4 Prateeek phone
        #self.destlat = 12.821330573
	#self.destlon = 80.039393088
	#3 Lekha phone
	#self.destlat = 12.8213791
	#self.destlon = 80.039185313
        #mid....
	#self.destlat = 12.821444731
        #self.destlon = 80.038990096
	#1
	#self.destlat = 12.821446675
 	#self.destlon = 80.038980441
	#self.destlat = -1.0
        #self.destlon = -1.0

    def run(self):
 #       rospy.init_node('autonomous',anonymous = False)
        rospy.Subscriber('Autonomous',AutoPointNeuron, updateDest)
        rospy.Subscriber('fix', NavSatFix, updateData)          #gpsdata to be defined in msg
        rospy.Subscriber('imu_data', ImuNeuron, updateComp)     #compassData to be defined in msg 
        rospy.Subscriber('destData', String, updateDest)
        print("setting up subscribers")
        rospy.spin()

#rospy.Subscriber()

class driver(Thread):
    def run(self):
        # l = -1
        while not rospy.is_shutdown():
            #print ("SIZE OF QUEUE",auto.l)
            if auto.l>=0 and auto.destlat!=-1 and auto.destlon!=-1:
                #print(auto.destlat,auto.destlon,auto.l)
                auto.destHeading = c.calBea(auto.lat, auto.lon, auto.destlat, auto.destlon)
                auto.difference = c.calDiff(auto.heading,auto.destHeading)
                auto.distance =6 #c.calDis(auto.lat, auto.lon, auto.destlat, auto.destlon)
                print("DISTANCE: "+str(auto.distance)+"    DIFFERENCE: "+str(auto.difference)+ "    POSITION:"+str(auto.lat)+", "+str(auto.lon) +"  HEADING: "+ str(auto.heading))
                #print("condition next")
                if auto.distance>accuracy:
                    fwd(auto.difference)
                else:
                    auto.l = qlon.qsize()-1
                    print ("POINT REACHED")
                    publishDrive(0,0)
                    auto.destlat = qlat.get()
                    auto.destlon = qlon.get()
            else:
                    publishDrive(0,0)
                    print("NO TARGET")
            time.sleep(0.02)



rospy.init_node('autonomous',anonymous = False)
auto = autoDriveData()
current = driver()
#pubDrive = rospy.Publisher('DriveData',DriveNeuron, queue_size=10)
#pubDrive =  rospy.Publisher('DriveData',DriveNeuron, queue_size=10)
#dataDrive = DriveNeuron()
loc = AutoPointNeuron()


if __name__ == '__main__':
    current.start()     #starting Differntial Drive Thread
    auto.run()          #Recieve data....Main Thread
