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
            //TODO: FaceSwap evtl. mit ellyptischer Form umbauen
            //ellipse(video_stream, Point(x , y ), Size(h/2, w/2), 0, 0, 360, Scalar(255, 0, 255), 8,0);
        }
        Mat display;
        video_stream.copyTo(display);
        if (faces.size() == 2) {
            // copy videoStream to display mat
            Mat faceROI1 = display(faces[0]);
            Mat faceROI2 = display(faces[1]);//Storing face in the matrix//

            //Mat faceROI1 = video_stream(faces[0]);
            //Mat faceROI2 = video_stream(faces[1]);//Storing face in the matrix//
    

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

            //Show results 
            //imshow("Source Image", faceROI2);
            //imshow("Destination Image", faceROI1);
            //imshow("Warped Source Image", im_out1);
            cout << faceROI1.size()<<endl;
            cout << im_out1.size()<<endl;
            cout << "these numbers should be equal " << endl;
            

            //TODO: position new image im_out at the right place in the videostream
            // 
            

            //copyTo for adding the image im_out to the videstream
            im_out1.copyTo(faceROI1);
            //copyTo for adding the image im_out to the videstream
            im_out2.copyTo(faceROI2);

            
            //Display images


            /*cv::resize(faceROI1, faceROI1temp, cv::Size(faceROI2.size().width, faceROI2.size().height));
            cv::resize(faceROI2, faceROI2temp, cv::Size(faceROI1.size().width, faceROI1.size().height));
            faceROI2temp.copyTo(faceROI1);
            faceROI1temp.copyTo(faceROI2);*/




            /*
            // Create a rough mask around the airplane.
            Mat faceROI1_mask = Mat::zeros(faceROI1temp.rows, faceROI1temp.cols, faceROI1temp.depth());

            // Define the mask as a closed polygon
            Point poly1[1][7];
            poly1[0][0] = Point(4, 80);
            poly1[0][1] = Point(30, 54);
            poly1[0][2] = Point(151, 63);
            poly1[0][3] = Point(254, 37);
            poly1[0][4] = Point(298, 90);
            poly1[0][5] = Point(272, 134);
            poly1[0][6] = Point(43, 122);

            const Point* polygons1[1] = { poly1[0] };
            int num_points1[] = { 7 };

            // Create mask by filling the polygon

            fillPoly(faceROI1_mask, polygons1, num_points1, 1, Scalar(255, 255, 255));

            // Create a rough mask around the airplane.
            Mat faceROI2_mask = Mat::zeros(faceROI2temp.rows, faceROI2temp.cols, faceROI2temp.depth());

            // Define the mask as a closed polygon
            Point poly2[1][7];
            poly2[0][0] = Point(4, 80);
            poly2[0][1] = Point(30, 54);
            poly2[0][2] = Point(151, 63);
            poly2[0][3] = Point(254, 37);
            poly2[0][4] = Point(298, 90);
            poly2[0][5] = Point(272, 134);
            poly2[0][6] = Point(43, 122);

            const Point* polygons2[1] = { poly2[0] };
            int num_points2[] = { 7 };

            // Create mask by filling the polygon

            fillPoly(faceROI2_mask, polygons2, num_points2, 1, Scalar(255, 255, 255));

            cv::Mat output;
            Point2f center(faceROI1temp.cols/2, faceROI1temp.rows/ 2);
            cv::seamlessClone(faceROI1temp, faceROI2, faceROI1_mask, center, output,NORMAL_CLONE);
            imshow("Faces",output);


            /*
            Point2f p1_1(faces[0].x, faces[0].y);
            Point2f p1_2(faces[0].x + faces[0].width, faces[0].y);
            Point2f p1_3(faces[0].x, faces[0].y +  faces[0].height);
            Point2f p1_4(faces[0].x + faces[0].width, faces[0].y + faces[0].height);

            Point2f p2_1(faces[1].x, faces[1].y);
            Point2f p2_2(faces[1].x + faces[1].width, faces[1].y);
            Point2f p2_3(faces[1].x, faces[1].y + faces[1].height);
            Point2f p2_4(faces[1].x + faces[1].width, faces[1].y + faces[1].height);

            vector<Point2f> points1{p1_1,p1_2,p1_3,p1_4};
            vector<Point2f> points2{p2_1,p2_2,p2_3,p2_4};
            Mat h = cv::findHomography(points1, points2, cv::RANSAC);
            cout << "H:\n" << h << endl;
            warpPerspective(faceROI1src, faceROI2src, h, faceROI2src.size());

            Mat img_draw_matches;
            hconcat(faceROI1src, faceROI2src, img_draw_matches);
            for (size_t i = 0; i < points1.size(); i++)
            {
                Mat pt1 = (Mat_<double>(3, 1) << points1[i].x, points1[i].y, 1);
                Mat pt2 = h * pt1;
                pt2 /= pt2.at<double>(2);
                Point end((int)(faceROI1src.cols + pt2.at<double>(0)), (int)pt2.at<double>(1));
                line(img_draw_matches, points1[i], end, 23, 2);
            }
            imshow("Draw matches", img_draw_matches);

           */

           
        }
       

        imshow("Face Detection", display);
        //Showing the detected face//
        if (waitKey(10) == 27) { //wait time for each frame is 10 milliseconds//
            break;
        }
     
    }
    return 0;
}
