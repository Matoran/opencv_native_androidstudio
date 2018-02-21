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
}
