# cameraDetection
This project is DroneDeploy's coding challenge.

Environment: 
The python I use is 2.7.14 |Anaconda, Inc.
The openCV I use is version: 3.3.0

Algorithm:
The code canbe devided into 4 parts
1. convert the image taken by iphone and then convert into gray image
2. used findContours function to find contours and then find the convex hull of the QR code.
3. solve the PnP problem using OpenCV function and get the rotation and translation matrix
4. compute the rotation and translation matrix to find the distance between camera and phote and rotation angle of camera.

At last
I use Axes3D and matplotlib to implement the visulization of the position of photo and camera
