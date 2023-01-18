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

int main(int argc, char** argv) {

    // Load the cascades
    CascadeClassifier face_cascade;
    face_cascade.load("C:/Users/Toni/Documents/Workspace/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml");
    CascadeClassifier eyes_cascade;
    eyes_cascade.load("C:/Users/Toni/Documents/Workspace/opencv/sources/data/haarcascades/haarcascade_eye.xml");

    // Open the video stream
    VideoCapture stream(0);
    if (!stream.isOpened()) {
        cout << "Error : Unable to open camera" << endl;
        return -1;
    }

    // Variables to track eye blinks
    int blink_count1 = 0;
    int blink_count2 = 0;
    double prev_norm1 = 0;
    double prev_norm2 = 0;
    double curr_norm1 = 0;
    double curr_norm2 = 0;
    double threshold = 0.02;
    bool face1_detected = false;
    bool face2_detected = false;

    while (true) {
        Mat frame;
        stream >> frame;

        // Convert to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        // Draw rectangles around the faces and eyes
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);

            Mat faceROI = gray(faces[i]);
            vector<Rect> eyes;

            //-- In each face, detect eyes
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (size_t j = 0; j < eyes.size(); j++) {
                Point center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                circle(frame, center, radius, Scalar(255, 0, 0), 2);

                // Count eye blinks for the first face
                if (i == 0) {
                    Mat eyeROI = gray(Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height));
                    Mat eyeROI_norm;
                    cv::normalize(eyeROI, eyeROI_norm, 0, 255, cv::NORM_MINMAX);
                    curr_norm1 = cv::norm(eyeROI_norm, NORM_L1);
                    if (prev_norm1 == 0) prev_norm1 = curr_norm1;
                    else {
                        double norm_diff = abs(prev_norm1 - curr_norm1);
                        if (norm_diff > threshold) {
                            blink_count1++;
                        }
                        prev_norm1 = curr_norm1;
                    }
                }

                // Count eye blinks for the second face
                else if (i == 1) {
                    Mat eyeROI = gray(Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height));
                    Mat eyeROI_norm;
                    cv::normalize(eyeROI, eyeROI_norm, 0, 255, cv::NORM_MINMAX);
                    curr_norm2 = cv::norm(eyeROI_norm, NORM_L1);
                    if (prev_norm2 == 0) prev_norm2 = curr_norm2;
                    else {
                        double norm_diff = abs(prev_norm2 - curr_norm2);
                        if (norm_diff > threshold) {
                            blink_count2++;
                        }
                        prev_norm2 = curr_norm2;
                    }
                }
            }
        }

        // Display blink count for each face
        putText(frame, "Blink count (face 1): " + to_string(blink_count1), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        putText(frame, "Blink count (face 2): " + to_string(blink_count2), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        // Show the frame
        imshow("Frame", frame);
        if (waitKey(1) == 27) break;
    }

    return 0;
}