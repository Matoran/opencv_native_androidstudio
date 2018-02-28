package ch.hepia.iti.opencvnativeandroidstudio;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.CvType.CV_32F;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase _cameraBridgeViewBase;
    private int step = 0;
    private float scaledColumns;
    private float scaledRows;
    private float rows = -1;
    private float columns;
    private float border;
    private Mat save;

    private BaseLoaderCallback _baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load ndk built module, as specified in moduleName in build.gradle
                    // after opencv initialization
                    System.loadLibrary("native-lib");
                    _cameraBridgeViewBase.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{Manifest.permission.CAMERA},
                1);

        _cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.main_surface);
        _cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        _cameraBridgeViewBase.setCvCameraViewListener(this);
        _cameraBridgeViewBase.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                step = (step + 1) % 4;
            }
        });

        _cameraBridgeViewBase.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                int x = (int) ((event.getX() - border) / scaledColumns * columns);
                int y = (int) (event.getY() / scaledRows * rows);
                return false;
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        disableCamera();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, _baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            _baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(MainActivity.this, "Permission denied to read your External storage", Toast.LENGTH_SHORT).show();
                }
                return;
            }
            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    public void onDestroy() {
        super.onDestroy();
        disableCamera();
    }

    public void disableCamera() {
        if (_cameraBridgeViewBase != null)
            _cameraBridgeViewBase.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        System.out.println(step);
        if(step == 3){
            return save;
        }
        Mat base = inputFrame.gray();
        if(rows == -1){
            scaledColumns = 1f * base.cols() / base.rows() * _cameraBridgeViewBase.getHeight();
            scaledRows = _cameraBridgeViewBase.getHeight();
            rows = base.rows();
            columns = base.cols();
            border = (_cameraBridgeViewBase.getWidth() - scaledColumns)/2;
        }

        Mat result = new Mat(base.rows(), base.cols(), base.type());
        Mat kernel = new Mat(3, 3, CV_32F);
        //blur
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                kernel.put(i, j, 1f / (3*3));
            }
        }
        Imgproc.filter2D(base, result, -1, kernel);
        if(step == 0){
            return result;
        }
        //laplacien
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                kernel.put(i, j, (i == 1 && j == 1) ? 8 : -1);
            }
        }
        Imgproc.filter2D(result, base, -1, kernel);
        if(step == 1){
            return base;
        }
        //binary
        Imgproc.threshold(base, result, 32, 255, 1);

        /*for (int row = 0; row < base.rows(); row++) {
            for (int column = 0; column < base.cols(); column++) {
                base.put(row, column, (base.get(row, column)[0] > 127) ? 0 : 255);
            }
        }
        salt(base.getNativeObjAddr(), 2000);*/
        save = result;
        return result;
    }

    public native void salt(long matAddrGray, int nbrElem);
    public native void reduce(long reduce, int n);
    public native void sharpen(long base, long result);
    public native void median(long base, long result);
    public native void laplacien(long base, long result);
    public native void binary(long base, long result);
}

