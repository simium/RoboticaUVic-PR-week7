#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

/** Global variables */
std::string face_cascade_name = "haarcascade_frontalface_default.xml";
cv::CascadeClassifier face_detect;

int main(int argc, char *argv[])
{
  int i;
  // OpenCV video capture object
  cv::VideoCapture capture;

  // OpenCV image objects
  cv::Mat frame, gray, img;

  // Rectangles to be drawn
  std::vector<cv::Rect> rects;

  // Camera ID associated to device number in /dev/videoX
  int cam_id = 0;

  // Advertising to the user
  std::cout << "Opening video device " << cam_id << std::endl;

  // Open the video stream and make sure it's opened
  if( !capture.open(cam_id) )
  {
    std::cout << "Error opening the capture. May be invalid device id. EXIT program." << std::endl;
    return -1;
  }

  // Train the face classifier
  face_detect = cv::CascadeClassifier(face_cascade_name);

  // Capture loop until user presses a key
  while(true)
  {
    // Read image and check it
    if(!capture.read(frame))
    {
      std::cout << "No frame" << std::endl;
      cv::waitKey();
    }

    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Detect faces and generate the rectangles associated
    face_detect.detectMultiScale(gray, rects, 1.3, 4, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30), cv::Size(200,200));

    // Safe copy
    img = frame.clone();

    // Draw the rectangles
    for (i=0; i<rects.size(); i++)
    {
      cv::rectangle(img, rects[i], cv::Scalar(0,255,0), 2);
    }

    // Show image in a window
    cv::imshow("Output Window", img);

    // Waits 1 millisecond to check if a key has been pressed. If so, breaks the loop. Otherwise continues.
    if(cv::waitKey(1) >= 0) break;
  }
}
