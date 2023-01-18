#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//This header includes definition of 'rectangle()' function//
#include<opencv2/objdetect/objdetect.hpp>
//This header includes the definition of Cascade Classifier//
#include<string>
#include <opencv2/calib3d.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;
int main(int argc, char** argv) {
    Mat video_stream;//Declaring a matrix hold frames from video stream//
    VideoCapture real_time(0);//capturing video from default webcam//
    namedWindow("Face Detection");//Declaring an window to show the result//
    string trained_classifier_location = "D:/UserData/z0047zpj/ComputerVisionProjekt/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml";//Defining the location our XML Trained Classifier in a string// 
    CascadeClassifier faceDetector;//Declaring an object named 'face detector' of CascadeClassifier class//
    faceDetector.load(trained_classifier_location);//loading the XML trained classifier in the object//
    vector<Rect>faces;//Declaring a rectangular vector named faces//
    Mat faceROIToChange;
    while (true) {
        faceDetector.detectMultiScale(video_stream, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));//Detecting the faces in 'image_with_humanfaces' matrix//
        real_time.read(video_stream);// reading frames from camera and loading them in 'video_stream' Matrix//
       



        for (int i = 0; i < faces.size(); i++) { //for locating the face
            Mat faceROI = video_stream(faces[i]);//Storing face in the matrix//
            int x = faces[i].x;//Getting the initial row value of face rectangle's starting point//
            int y = faces[i].y;//Getting the initial column value of face rectangle's starting point//
            int h = y + faces[i].height;//Calculating the height of the rectangle//
            int w = x + faces[i].width;//Calculating the width of the rectangle//
            rectangle(video_stream, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8,0);//Drawing a rectangle using around the faces//
            //Detect the mouth
            string trained_classifier_location_smile = "D:/UserData/z0047zpj/ComputerVisionProjekt/opencv/sources/data/haarcascades/haarcascade_smile.xml";//Defining the location our XML Trained Classifier in a string// 
            CascadeClassifier smileDetector;//Declaring an object named 'face detector' of CascadeClassifier class//
            smileDetector.load(trained_classifier_location_smile);//loading the XML trained classifier in the object//
            vector<Rect>smiles;//Declaring a rectangular vector named faces//
            smileDetector.detectMultiScale(faceROI, smiles, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));//Detecting the faces in 'image_with_humanfaces' matrix//
            for (int i = 0; i < smiles.size(); i++) { //for locating the smile
                Mat smileROI = faceROI(smiles[i]);//Storing smile in the matrix//
                int x = smiles[i].x;//Getting the initial row value of face rectangle's starting point//
                int y = smiles[i].y;//Getting the initial column value of face rectangle's starting point//
                int h = y + smiles[i].height;//Calculating the height of the rectangle//
                int w = x + smiles[i].width;//Calculating the width of the rectangle//
                rectangle(faceROI, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);//Drawing a rectangle using around the mouths//

            }

            
        }
        Mat display;
        video_stream.copyTo(display);
        if (faces.size() == 2) {
            // copy videoStream to display mat
            Mat faceROI1 = display(faces[0]);
            Mat faceROI2 = display(faces[1]);//Storing face in the matrix//

    

            // Define corner points of faces in cevtor
            faceROI2.cols;
            faceROI2.rows;

            Point2f p1_1(0, 0);
            Point2f p1_2(faceROI1.cols, 0);
            Point2f p1_3(0, faceROI1.rows);
            Point2f p1_4(faceROI1.cols, faceROI1.rows);

            Point2f p2_1(0,0);
            Point2f p2_2(faceROI2.cols, 0);
            Point2f p2_3(0, faceROI2.rows);
            Point2f p2_4(faceROI2.cols, faceROI2.rows);

            vector<Point2f> points1{ p1_1,p1_2,p1_3,p1_4 };
            vector<Point2f> points2{ p2_1,p2_2,p2_3,p2_4 };


            //find Homography matrix that is the homography from face 2 to face 1
            Mat h1 = findHomography(points2, points1);

            //find Homography matrix that is the homography from face 1 to face 2
            Mat h2 = findHomography(points1, points2);

            //Output image face 2 warped
            Mat im_out1;

            //Output image face 1 warped
            Mat im_out2;

            //Warp faceROI2 and save it in im_out1
            warpPerspective(faceROI2, im_out1, h1, faceROI1.size());
            //Warp faceROI1 and save it in im_out2
            warpPerspective(faceROI1, im_out2, h2, faceROI2.size());

           
            cout << faceROI1.size()<<endl;
            cout << im_out1.size()<<endl;
            cout << "these numbers should be equal " << endl;
            

            //copyTo for adding the image im_out to the videstream
            im_out1.copyTo(faceROI1);
            //copyTo for adding the image im_out to the videstream
            im_out2.copyTo(faceROI2);

           
        }
       
        
        imshow("Face Detection", display);
        //Showing the detected face//
        if (waitKey(10) == 27) { //wait time for each frame is 10 milliseconds//
            break;
        }
     
    }
    return 0;
}
