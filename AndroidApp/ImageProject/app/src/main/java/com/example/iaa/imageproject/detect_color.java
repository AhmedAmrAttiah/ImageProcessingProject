

package com.example.iaa.imageproject;

        import android.graphics.Color;
        import android.speech.tts.TextToSpeech;
        import android.support.v7.app.AppCompatActivity;
        import android.os.Bundle;
        import android.view.MotionEvent;
        import android.view.SurfaceView;
        import android.view.View;
        import android.view.WindowManager;
        import android.widget.TextView;

        import org.json.JSONArray;
        import org.json.JSONException;
        import org.json.JSONObject;
        import org.opencv.android.BaseLoaderCallback;
        import org.opencv.android.CameraBridgeViewBase;
        import org.opencv.android.LoaderCallbackInterface;
        import org.opencv.android.OpenCVLoader;
        import org.opencv.core.Core;
        import org.opencv.core.CvType;
        import org.opencv.core.Mat;
        import org.opencv.core.Rect;
        import org.opencv.core.Scalar;
        import org.opencv.imgproc.Imgproc;

        import java.util.Locale;

public class detect_color extends AppCompatActivity implements View.OnTouchListener, CameraBridgeViewBase.CvCameraViewListener2 {

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;
    private Scalar mBlobColorHsv;
    private Scalar mBlobColorRgba;

    TextView touch_coordinates;
    TextView touch_color;
    TextView color_name;

    TextToSpeech speakText;
    double x = -1;
    double y = -1;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(detect_color.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detect_color);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        touch_coordinates = (TextView) findViewById(R.id.touch_coordinates);
        touch_color = (TextView) findViewById(R.id.touch_color);
        color_name = (TextView) findViewById(R.id.textView);

        speakText = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR)
                    speakText.setLanguage(Locale.US);

            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_tutorial_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        return mRgba;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        double yLow = (double)mOpenCvCameraView.getHeight() * 0.2401961;
        double yHigh = (double)mOpenCvCameraView.getHeight() * 0.7696078;

        double xScale = (double)cols / (double)mOpenCvCameraView.getWidth();
        double yScale = (double)rows / (yHigh - yLow);

        x = event.getX();
        y = event.getY();

        y = y - yLow;

        x = x * xScale;
        y = y * yScale;

        if((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        //touch_coordinates.setText("X: " + Double.valueOf(x) + ", Y: " + Double.valueOf(y));

        Rect touchedRect = new Rect();

        touchedRect.x = (int)x;
        touchedRect.y = (int)y;

        touchedRect.width = 8;
        touchedRect.height = 8;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width * touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);

        String ColorHex = "#" + String.format("%02X", (int)mBlobColorRgba.val[0])
                + String.format("%02X", (int)mBlobColorRgba.val[1])
                + String.format("%02X", (int)mBlobColorRgba.val[2]);

        touch_color.setText("Color: " + ColorHex);

        String colorName = getClosestColor((int) mBlobColorRgba.val[0],
                (int) mBlobColorRgba.val[1],
                (int) mBlobColorRgba.val[2]);

        color_name.setText("Name: " + colorName);

        speakText.speak(colorName, TextToSpeech.QUEUE_FLUSH, null);

        touch_color.setTextColor(Color.rgb((int) mBlobColorRgba.val[0],
                (int) mBlobColorRgba.val[1],
                (int) mBlobColorRgba.val[2]));
        /*touch_coordinates.setTextColor(Color.rgb((int)mBlobColorRgba.val[0],
                (int)mBlobColorRgba.val[1],
                (int)mBlobColorRgba.val[2]));*/

        return false;
    }

    public static String getClosestColor(int red, int green, int blue) {

        int redMultiple = 1;
        int greenMultiple = 1;
        int blueMultiple = 1;
        if(red>=green && red>= blue)
            redMultiple = 3;
        else if(green>=red && green >=blue)
            greenMultiple = 3;
        else
            blueMultiple = 3;
        double currDifference = Double.MAX_VALUE;
        String ret = "";
        try {
            JSONArray d = new JSONArray(Data.getAllColors());

            for(int i = 0; i< d.length(); i++)
            {

                JSONObject curr = d.optJSONObject(i);

                double avg = calculateDifferenceWithHue(red, green, blue, curr);
                if(avg<currDifference)
                {
                    currDifference = avg;
                    ret = curr.optString("label");
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return ret;
    }

    private static double calculateDifferenceWithHue(int red, int green, int blue, JSONObject curr){
        double actualHue = getHueFromRGB(red, green, blue);
        double currHue = getHueFromRGB(curr.optInt("x"), curr.optInt("y"), curr.optInt("z"));
        System.out.println(actualHue);
        double avg = Math.abs(actualHue-currHue);
        return avg;
    }

    public static double getHueFromRGB(int red, int green, int blue) {
        float[] HSV = new float[3];
        Color.RGBToHSV(red, green, blue, HSV);
        return HSV[0];

    }


    private Scalar convertScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
