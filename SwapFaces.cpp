#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Load the cascade classifier model
    cv::CascadeClassifier face_cascade;
    face_cascade.load("C:/Users/Toni/Documents/Workspace/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml");

    cv::CascadeClassifier eye_cascade;
    eye_cascade.load("C:/Users/Toni/Documents/Workspace/opencv/sources/data/haarcascades/haarcascade_eye.xml");

    // Open the webcam
    cv::VideoCapture cap(0);

    int blink_counter = 0;
    double ear_threshold = 0.5;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        // Loop through the detected faces
        for (const auto& face : faces)
        {
            // Detect eyes in the face region
            std::vector<cv::Rect> eyes;
            eye_cascade.detectMultiScale(gray(face), eyes);

            for (const auto& eye : eyes)
            {
                // Create a rectangle for the eye
                cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);

                // Extract the eye region
                cv::Mat eye_region = gray(eye_rect);

                // Detect the eyes' landmarks
                vector<Point2f> landmarks;
                cv::goodFeaturesToTrack(eye_region, landmarks, 8, 0.1, 10);

                // Calculate the Eye Aspect Ratio (EAR)
                if(landmarks.size() >= 6) {
                  double a = cv::norm(landmarks[1] - landmarks[5]);
                  double b = cv::norm(landmarks[2] - landmarks[4]);
                  double c = cv::norm(landmarks[0] - landmarks[3]);
                  double ear = (a + b) / (2 * c);

                  cout << "ear: " << ear << " ";

                  // Check if the eye is closed (EAR is low)
                  if (ear < ear_threshold)
                  {
                      // Increment the blink counter
                      blink_counter++;

                      // Draw a rectangle around the closed eye
                      cv::rectangle(frame, eye_rect, cv::Scalar(0, 0, 255), 2);
                  }
                  else
                  {
                      // Draw a rectangle around the open eye
                      cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
                  }
                }
            }
        }

        // Display the blink counter on the frame
        cv::putText(frame, "Blinks: " + std::to_string(blink_counter), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

        // Show the frame
        cv::imshow("Frame", frame);

        // Exit if the user presses the 'Esc' key
        if (cv::waitKey(1) == 27)
            break;
    }

    // Release the webcam and close the window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}