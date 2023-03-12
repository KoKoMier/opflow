#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    
    VideoCapture cap(0);

    int maxCorners = 100;
    double qualityLevel = 0.03;
    double minDistance = 7;
    Size winSize(10, 10);
    int maxLevel = 2;
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    vector<uchar> status;
    vector<float> err;
    vector<Point2f> good_new, good_old;
    double x_last = 0;
    double y_last = 0;
    Mat currimg,currgrayimg;
    Mat prev_img, prev_img_gray;
    int x_err,y_err,x,y;
    vector<Point2f> prev_points,curr_points;
    Mat img;
    Size size(640, 480);

    cap >> prev_img;
    resize(prev_img, prev_img, size);
    cvtColor(prev_img, prev_img_gray, COLOR_BGR2GRAY);
    Mat mask_img = Mat::zeros(prev_img.size(), prev_img.type());

    goodFeaturesToTrack(prev_img_gray, prev_points, maxCorners, qualityLevel, minDistance);

    while (true){
        cap >> currimg;
        resize(currimg, currimg, size);
        cvtColor(currimg,currgrayimg,COLOR_BGR2GRAY);

        calcOpticalFlowPyrLK(prev_img_gray, currgrayimg, prev_points, curr_points, status, err, winSize, maxLevel, criteria);    


        for (int i = 0; i < prev_points.size(); i++) {
            if (status[i] == 1) {
            Point2f prev_pt = prev_points[i];
            Point2f curr_pt = curr_points[i];

            line(currimg, prev_points[i], curr_points[i], Scalar(0, 0, 255), 3);

            x_err = curr_pt.x - prev_pt.x;
            y_err = curr_pt.y - prev_pt.y;

            x = x_err + x_last;
            y = y_err + y_last;

            x_last = x;
            y_last = y;

            prev_points[i] = curr_pt;
                    }
                }
        int count = 0;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
            count++;
            }
        }
        putText(currimg, "x: " + to_string(static_cast<int>(x/count)) + " y: " + to_string(static_cast<int>(y/count)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);

        imshow("img1", currimg);
        swap(prev_points, curr_points);
        swap(prev_img_gray, currgrayimg);   

        if ((waitKey(30) & 0xff) == 'q') break;
        
        cout << "Detected " << count << " feature points." << std::endl;

        if (count <10)
        {
            goodFeaturesToTrack(prev_img_gray, prev_points, maxCorners, qualityLevel, minDistance);
        }

    }
    destroyAllWindows();
    cap.release();

    

}