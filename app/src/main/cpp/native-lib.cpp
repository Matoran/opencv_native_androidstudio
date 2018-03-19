#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

extern "C"
{
    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_salt(JNIEnv *env, jobject instance,
                                                                               jlong matAddrGray,
                                                                               jint nbrElem) {
        Mat &mGr = *(Mat *) matAddrGray;
        for (int k = 0; k < nbrElem; k++) {
            int i = rand() % mGr.cols;
            int j = rand() % mGr.rows;
            mGr.at<uchar>(j, i) = 255;
        }
    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_reduce(JNIEnv *env, jobject instance,
                                                                           jlong matAddrGray,
                                                                            jint n) {
        Mat &I = *(Mat *) matAddrGray;
        const int channels = I.channels();
        switch(channels)
        {
            case 1:
            {
                MatIterator_<uchar> it, end;
                for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                    *it = *it / n * n + n / 2;
                break;
            }
            case 4:
            {
                MatIterator_<Vec4b> it, end;
                for( it = I.begin<Vec4b>(), end = I.end<Vec4b>(); it != end; ++it)
                {
                    (*it)[0] = (*it)[0] / n * n + n / 2;
                    (*it)[1] = (*it)[1] / n * n + n / 2;
                    (*it)[2] = (*it)[2] / n * n + n / 2;
                }
            }
        }

    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_sharpen(JNIEnv *env, jobject instance,
                                                                                 jlong matAddrGray,
                                                                                 jlong matResult) {
        Mat &I = *(Mat *) matAddrGray;
        Mat &J = *(Mat *) matResult;
        int channels = I.channels();

        int nRows = I.rows;
        int nCols = I.cols * channels;


        int i,j;
        uchar *previous, *current, *next, *newCurrent;
        for( i = 1; i < nRows-1; ++i) {
            previous = I.ptr<uchar>(i-1);
            current = I.ptr<uchar>(i);
            next = I.ptr<uchar>(i+1);
            newCurrent = J.ptr<uchar>(i);
            for ( j = channels; j < nCols-channels; ++j) {
                newCurrent[j] = saturate_cast<uchar>(5*current[j] - current[j-channels] - current[j+channels] - previous[j] - next[j]);
            }
        }
    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_median(JNIEnv *env, jobject instance,
                                                                                  jlong matAddrGray,
                                                                                  jlong matResult) {
        Mat &I = *(Mat *) matAddrGray;
        Mat &J = *(Mat *) matResult;
        Mat kernel = Mat::ones( 15, 15, CV_32F )/ (float)(15*15);
        filter2D(I, J, -1, kernel);
    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_laplacien(JNIEnv *env, jobject instance,
                                                                                 jlong matAddrGray,
                                                                                 jlong matResult) {
        Mat &I = *(Mat *) matAddrGray;
        Mat &J = *(Mat *) matResult;
        Mat kernel = Mat::ones( 3, 3, CV_32F );
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                kernel.at<uchar>(i, j) = (i == 1 && j == 1) ? 8 : -1;
            }
        }
        filter2D(I, J, -1, kernel);

    }

    void JNICALL Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_binary(JNIEnv *env, jobject instance,
                                                                                    jlong matAddrGray,
                                                                                    jlong matResult) {

    }
}
extern "C"
JNIEXPORT void JNICALL
Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_canny(JNIEnv *env, jobject instance,
                                                               jlong base, jlong result, jint a, jint b) {

    Mat &src_gray = *(Mat *) base;
    Mat &detected_edges = *(Mat *) result;
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );

    /// Canny detector
    Canny( detected_edges, detected_edges, a, b);

    /// Using Canny's output as a mask, we display our result
    Mat dst;
    dst = Scalar::all(0);
    src_gray.copyTo( dst, detected_edges);

}

extern "C"
 JNIEXPORT void JNICALL
 Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_houghLinesP(JNIEnv *env, jobject instance,
                                                                      jlong base, jlong result) {
     Mat &dst = *(Mat *) base;
     Mat &r = *(Mat *) result;
     Mat color_dst;
     cvtColor( dst, color_dst, CV_GRAY2BGR );
     vector<Vec4i> lines;
     HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
     for( size_t i = 0; i < lines.size(); i++ )
     {
         Vec4i l = lines[i];
         line( color_dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
     }
    r = Scalar::all(0);
     color_dst.copyTo(r);

 }extern "C"
 JNIEXPORT void JNICALL
 Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_houghCircle(JNIEnv *env, jobject instance,
                                                                      jlong base) {
     Mat &src = *(Mat *) base;

     /// Reduce the noise so we avoid false circle detection
     //GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

     vector<Vec3f> circles;

     /// Apply the Hough Transform to find the circles
     HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 200, 100, 0, 0 );

     /// Draw the circles detected
     for( size_t i = 0; i < circles.size(); i++ )
     {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // circle center
         circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // circle outline
         circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
     }
 }extern "C"
 JNIEXPORT void JNICALL
 Java_ch_hepia_iti_opencvnativeandroidstudio_MainActivity_red(JNIEnv *env, jobject instance,
                                                              jlong base,
                                                              jint amin, jint bmin, jint cmin,
                                                              jint a, jint b, jint c) {
     Mat &src = *(Mat *) base;
     Mat frame;
     cvtColor(src, src,CV_RGB2HSV);
     inRange(src, Scalar(amin, bmin, cmin), Scalar(a, b, c), src);
     //frame.copyTo(src);
     //cvtColor(src, src,CV_HSV2RGB);
 }