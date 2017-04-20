# horizon
Sample of using OpenCV for skyline edge detecting.

This is a smal programm for detecting skyline edge in provided video files.
Sample files can be found in test_video folder.
Used Microsoft Visual Studio 2015 and OpenCV 3.2
To compile on Linux (tested on Ubuntu) find skyline.cpp and execute:

g++ skyline.cpp -Wall -o skyline \`pkg-config --libs opencv\`
