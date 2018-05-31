package vn.com.tma.hdnghia.hairrecognite;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Camera;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedWriter;
import java.io.Console;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static  final String TAG = "OCV_Camera::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private int render=0;

    private Mat mRgba;
    private Mat mRgbaF;
    private Mat mRgbaT;

    private Mat mGray;
    private Mat dst;
    private Button btnCapture;

    private final int  MY_PERMISSIONS_REQUEST_CAMERA = 1;
    private boolean isPermissionGranted = false;
    private int mScreenWidth = 640, mScreenHeight = 480;

    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");
//    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

//        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.KITKAT) {
//            requestPermissions(new String[]{Manifest.permission.CAMERA},100);
//        }

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.opencvCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setMaxFrameSize(mScreenWidth,mScreenHeight);
        mOpenCvCameraView.setCvCameraViewListener(this);

        textureView = (TextureView)findViewById(R.id.textureView);
        //From Java 1.4 , you can use keyword 'assert' to check expression true or false
        assert textureView != null;
        textureView.setSurfaceTextureListener(textureListener);
        btnCapture = (Button)findViewById(R.id.btnCapture);
        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takePicture();
            }
        });
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case BaseLoaderCallback.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG,"Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this, mLoaderCallback);
        }else{
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mOpenCvCameraView!=null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
//        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
//        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
        mGray = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        mRgba = inputFrame.rgba();
//        mGray = inputFrame.gray();
//        dst = new Mat();
//        Imgproc.resize(mRgba, dst, new Size(mScreenWidth,mScreenHeight));

//        grabCut();
//        skinSegmentation();
//        setQuantizedImages();
//        findImageDifference();
//        performErosion_Dilution();
//        findContours();
////        return predict_hair();
//        return matrix9_finalOutput;
        return  null;
    }

    Mat matrix2_grabcut;
    Mat matrix3_skindetection;
    Mat matrix5_grabcut_quantized;
    Mat matrix6_skin_quantized;
    Mat matrix7_output;
    Mat erosion_dilutionMatrix;
    Mat matrix8_max_contour;
    Mat matrix9_finalOutput;

    /**
     * Paso uno
     * @return
     */
    private void grabCut(Mat srImage){
        Mat sourceImage = srImage;

        Mat result = new Mat(sourceImage.size(),sourceImage.type());
        Mat bgModel = new Mat();    //background model
        Mat fgModel = new Mat();    //foreground model

        //draw a rectangle
        Rect rectangle = new Rect(1,1,sourceImage.cols()-1,sourceImage.rows()-1);

        Imgproc.grabCut(sourceImage, result,rectangle, bgModel,fgModel,10,Imgproc.GC_INIT_WITH_RECT);
        Core.compare(result,new Scalar(3,3,3),result,Core.CMP_EQ);
        matrix2_grabcut = new Mat(sourceImage.size(),CvType.CV_8UC3,new Scalar(255,255,255));
        sourceImage.copyTo(matrix2_grabcut, result);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "abc" +".jpg",matrix2_grabcut);
    }

    private void skinSegmentation(){
        matrix3_skindetection = new Mat(matrix2_grabcut.size(), matrix2_grabcut.type());
        matrix3_skindetection.setTo(new Scalar(0,0,255));

        Mat skinMask = new Mat();
        Mat hsvMatrix = new Mat();

        Scalar lower = new Scalar(0,48,80);
        Scalar upper = new Scalar(20,255,255);

        Imgproc.cvtColor(matrix2_grabcut,hsvMatrix,Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvMatrix,lower,upper,skinMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11,11));
        Imgproc.erode(skinMask, skinMask, kernel);
        Imgproc.dilate(skinMask, skinMask, kernel);

        Imgproc.GaussianBlur(skinMask,skinMask, new Size(3,3),0);

        Core.bitwise_and(matrix2_grabcut, matrix2_grabcut, matrix3_skindetection, skinMask);
    }

    private void setQuantizedImages() {
        matrix5_grabcut_quantized = this.quantizeImage(matrix2_grabcut);
        matrix6_skin_quantized = this.quantizeImage(matrix3_skindetection);
    }

    private  Mat quantizeImage(Mat image) {
        //Mat image  = testGrabCut(imageFilePath);
        int rows = image.rows();
        int cols = image.cols();
        Mat newImage = new Mat(image.size(),image.type());
        for(int r = 0 ; r < rows ; r++)
        {
            for(int c =0; c< cols; c++)
            {
                double [] pixel_val = image.get(r, c);
                double [] pixel_data = new double[3];
                pixel_val[0] = reduceVal(pixel_val[0]);
                pixel_val[1] = reduceVal(pixel_val[1]);
                pixel_val[2] = reduceVal(pixel_val[2]);

                newImage.put(r, c, pixel_val);
            }
            //  System.out.println();
        }
        /*
        MatOfInt params= new MatOfInt();
        int arr[] = new int[2];
        arr[0]= Imgcodecs.CV_IMWRITE_JPEG_QUALITY;
        arr[1]= 100;
        params.fromArray(arr);
        */
        return newImage;
    }

    private double reduceVal(double val){
        if(val >=0.00 && val <64.00) return 0.00;
        else if(val>=64.00 && val <128.00) return 64.00;
        else if (val>= 128.00 && val < 192.00) return 128.00;
        else return 255.00;

    }

    private void findImageDifference() {
        matrix7_output = new Mat(dst.size(),dst.type());
        matrix7_output.setTo(new Scalar(255,255,255));      //white colored image
        int rows = dst.rows();
        int cols = dst.cols();

        for(int r=0;r <rows ; r++)
        {
            for(int c =0; c < cols; c++)
            {
                //  double grabcut_pixel_val[] =matrix2_grabcut.get(r, c);
                //  double skin_pixel_val[] = newMask.get(r, c);
                double grabcut_pixel_val[] =matrix5_grabcut_quantized.get(r,c);
                double skin_pixel_val[] =  matrix6_skin_quantized.get(r,c);
                //extract those pixels which are non blue in 1st image and red in 2nd image
                if(  ( (grabcut_pixel_val[0] != 255 ) && (grabcut_pixel_val[1]!=255 ) && (grabcut_pixel_val[2] !=255) )  && ( (skin_pixel_val[0]== 0) && (skin_pixel_val[1]== 0) &&(skin_pixel_val[2]== 255) ) )
                {
                    double orgImage_pixel_val[] = dst.get(r, c);
                    //double orgImage_pixel_val[] = new double[]{0,0,0};
                    //double pixel_val[] = new double[3];
                    //pixel_val[0]=pixel_val[1]=pixel_val[2]=0;
                    matrix7_output.put(r, c, orgImage_pixel_val);
                }
            }
        }

    }

    private  void performErosion_Dilution() {
        erosion_dilutionMatrix = new Mat(this.matrix7_output.size(),this.matrix7_output.type());
        int erosion_size=2;

        //erosion
        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
        Imgproc.erode(matrix7_output, erosion_dilutionMatrix, element1);

            /*
            //dilation
            Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
            Imgproc.dilate(erosion_dilutionMatrix, erosion_dilutionMatrix, element2);
            */
    }

    private void findContours(){

        //Mat orgImage = Imgcodecs.imread(imageFilePath); //load image
        Mat grayImage = new Mat();
        Mat cannyImage= new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.cvtColor(this.erosion_dilutionMatrix, grayImage, Imgproc.COLOR_BGR2GRAY);      //bgr to gray scale image conversion
        Imgproc.Canny(grayImage, cannyImage, 100, 200);     //get edges of image


        //morph edge detected image to improve egde connectivity
        Mat element= Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(5,5));
        Imgproc.morphologyEx(cannyImage, cannyImage, Imgproc.MORPH_CLOSE, element);


        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);    //find all contours

        matrix8_max_contour = Mat.zeros(cannyImage.size(),CvType.CV_8UC3);
        matrix8_max_contour.setTo(new Scalar(255,255,255));
        double maxArea = 0.0;
        if (contours.size() > 0) {
            Imgproc.contourArea(contours.get(0));
        }
        // double maxArea = 0;
        int maxAreaIndex = 0;
        MatOfPoint temp_contour;
        if (contours.size() > 0) {
            for (int i = 1; i < contours.size(); i++) {
                temp_contour = contours.get(i);
                double curr_cont_area = Imgproc.contourArea(temp_contour);
                if (maxArea < curr_cont_area) {
                    maxArea = curr_cont_area;
                    maxAreaIndex = i;
                }
            }
        }
        //Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),Core.FILLED);
        //   Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),1);
        Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),-1);
        //Imgproc.watershed();


        matrix9_finalOutput = new Mat(dst.size(),dst.type());
        matrix9_finalOutput.setTo(new Scalar(255,255,255));

        for( int r =0 ;r < matrix8_max_contour.rows() ; r++)
        {
            for( int c=0; c < matrix8_max_contour.cols() ; c++)
            {
                double[] pixel_val = matrix8_max_contour.get(r, c);
                if(pixel_val[0] == 0 && pixel_val[1] == 0 && pixel_val[2] == 0)
                {
                    double[] orginal_pixel_val = dst.get(r, c);
                    matrix9_finalOutput.put(r,c,orginal_pixel_val);
                }
            }
        }

        //dilution on final image
        int erosion_size=2;
        Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
        Imgproc.dilate(matrix9_finalOutput,matrix9_finalOutput, element2);

        // Imgproc.drawContours(mask, contours, maxAreaIndex, new Scalar(255,255,255));


        /*
        //--------------code for copying original image with mask--------------------------
        //Mat croppedImage = new Mat(orgImage.size(),CvType.CV_8UC3);
        Mat croppedImage = new Mat(orgImage.size(),orgImage.type());
        //Mat croppedImage = Mat.ones(orgImage.size(), orgImage.type());
        croppedImage.setTo(new Scalar(0,0,0));
        String destinationPath = "/home/sujit25/Pictures/croppedImage.png";
        orgImage.copyTo(croppedImage, mask);              // copy original image with mask
        Imgcodecs.imwrite(destinationPath, croppedImage);
        return destinationPath;
        */
        /* //normalize and save mask
        //Core.normalize(mask, mask,0,255,Core.NORM_MINMAX,CvType.CV_8UC3);
         // write mask
        String destinationPath2 = "/home/sujit25/Pictures/maskImage.png";
        Imgcodecs.imwrite(destinationPath2,mask);
        return destinationPath2;
        */

         /*
        //---------------code for grabcut with final mask------------------------------/
        Mat bgModel = new Mat();        //background model
        Mat fgModel = new Mat();        //foreground model
        Rect rectangle = Imgproc.boundingRect(contours.get(maxAreaIndex));  //draw a rectangle around maximum area contour
        Mat result = new Mat();

        Imgproc.grabCut(orgImage, result, rectangle, bgModel, fgModel,1,Imgproc.GC_INIT_WITH_RECT);
        Core.compare(result,new Scalar(3,3,3),result,Core.CMP_EQ);
        Mat foreground = new Mat(orgImage.size(),CvType.CV_8UC3,new Scalar(0,0,255));
        orgImage.copyTo(foreground, result);
        String destination = "/home/sujit25/Pictures/Results/face4_contourImage.png";
        Imgcodecs.imwrite(destination, foreground);
        return destination;
        */
        //  return this.skinSegmentation_WithThreshold(destination);
    }

    public Mat predict_hair() {
        Mat hsv_input = matrix9_finalOutput.clone();
        List<Mat> channels= new ArrayList<>();
        Mat hsv_histogram = new Mat();
        MatOfFloat ranges = new MatOfFloat(0,180);
        MatOfInt histSize = new MatOfInt(255);
        Imgproc.cvtColor(hsv_input, hsv_input, Imgproc.COLOR_BGR2HSV);
        Core.split(hsv_input, channels);
        Imgproc.calcHist(channels.subList(0,1), new MatOfInt(0), new Mat(), hsv_histogram, histSize, ranges);
        int hist_w =256;
        int hist_h = 150;

        int bin_w =(int)Math.round(hist_w/histSize.get(0,0)[0]);
        Mat histImage= new Mat(hist_h,hist_w,CvType.CV_8UC3,new Scalar(0,0,0));

        for(int i=1;i < histSize.get(0,0)[0];i++)
        {
            Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hsv_histogram.get(i - 1, 0)[0])), new Point(bin_w * (i), hist_h - Math.round(hsv_histogram.get(i, 0)[0])), new Scalar(255,0,0),2);
        }

        return histImage;
    }

    /**
     * Devuelve una imagen en blanco y negro donde se distingen los contornos de las cosas
     * @param gray imagen inicial en escala de grises
     * @return imagen en blando y negro con contornos
     */
    private Mat contoursWithImageinBlackAndWhite(Mat gray){
        Imgproc.blur(gray,gray, new Size(3,3));
        Mat canny_output = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchay = new Mat();
        Imgproc.Canny(gray,canny_output,100,100*2);
        Imgproc.findContours(canny_output,contours,hierarchay,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE,new Point(0,0));
        Mat drawing = Mat.zeros(canny_output.size(), CvType.CV_8UC3);
        for (int i =0; i<contours.size(); i++){
            Scalar color = new Scalar(255,255,255);
            Imgproc.drawContours(drawing,contours,i,color,1,1,hierarchay,0,new Point());
        }
        return drawing;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    //public native String stringFromJNI();

    private CameraDevice cameraDevice;
    private File file;
    private CaptureRequest.Builder captureRequestBuilder;
    private TextureView textureView;
    private CameraCaptureSession cameraCaptureSessions;
    private android.util.Size imageDimension;
    private Handler mBackgroundHandler;
    private  String cameraId;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private Mat rsImage;

    CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraDevice.close();
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int i) {
            cameraDevice.close();
            cameraDevice=null;
        }
    };

    private void createCameraPreview() {
        try{
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert  texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(),imageDimension.getHeight());
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    if(cameraDevice == null)
                        return;
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(MainActivity.this, "Changed", Toast.LENGTH_SHORT).show();
                }
            },null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void takePicture() {
        if(cameraDevice == null)
            return;
        CameraManager manager = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        try{
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraDevice.getId());
            android.util.Size[] jpegSizes = null;
            if(characteristics != null)
                jpegSizes = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        .getOutputSizes(ImageFormat.JPEG);

            //Capture image with custom size
            int width = 640;
            int height = 480;
            if(jpegSizes != null && jpegSizes.length > 0)
            {
                width = jpegSizes[0].getWidth();
                height = jpegSizes[0].getHeight();
            }
            final ImageReader reader = ImageReader.newInstance(width,height,ImageFormat.JPEG,1);
            List<Surface> outputSurface = new ArrayList<>(2);
            outputSurface.add(reader.getSurface());
            outputSurface.add(new Surface(textureView.getSurfaceTexture()));

            final CaptureRequest.Builder captureBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(reader.getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);

            //Check orientation base on device
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION,ORIENTATIONS.get(rotation));

            file = new File(Environment.getExternalStorageDirectory()+"/"+ UUID.randomUUID().toString()+".jpg");
            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader imageReader) {
                    Image image = null;
                    try{
                        image = reader.acquireLatestImage();
                        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                        byte[] bytes = new byte[buffer.capacity()];
                        buffer.get(bytes);
                        save(bytes);
                        rsImage = new Mat(mScreenWidth, mScreenHeight, CvType.CV_8UC3);
                        rsImage.put(0, 0, bytes);
                        grabCut(rsImage);
                    }
                    catch (FileNotFoundException e)
                    {
                        e.printStackTrace();
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                    }
                    finally {
                        {
                            if(image != null)
                                image.close();
                        }
                    }
                }
                private void save(byte[] bytes) throws IOException {
                    OutputStream outputStream = null;
                    try{
                        outputStream = new FileOutputStream(file);
                        outputStream.write(bytes);
                    }finally {
                        if(outputStream != null)
                            outputStream.close();
                    }
                }
            };

            reader.setOnImageAvailableListener(readerListener,mBackgroundHandler);
            final CameraCaptureSession.CaptureCallback captureListener = new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    Toast.makeText(MainActivity.this, "Saved "+file, Toast.LENGTH_SHORT).show();
                    createCameraPreview();
                }
            };

            cameraDevice.createCaptureSession(outputSurface, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    try{
                        cameraCaptureSession.capture(captureBuilder.build(),captureListener,mBackgroundHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {

                }
            },mBackgroundHandler);


        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void updatePreview() {
        if(cameraDevice == null)
            Toast.makeText(this, "Error", Toast.LENGTH_SHORT).show();
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE,CaptureRequest.CONTROL_MODE_AUTO);
        try{
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(),null,mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static{
        ORIENTATIONS.append(Surface.ROTATION_0,90);
        ORIENTATIONS.append(Surface.ROTATION_90,0);
        ORIENTATIONS.append(Surface.ROTATION_180,270);
        ORIENTATIONS.append(Surface.ROTATION_270,180);
    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i1) {
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i1) {

        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

        }
    };

    private void openCamera() {
        CameraManager manager = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        try{
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            //Check realtime permission if run higher API 23
            if(ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            {
                ActivityCompat.requestPermissions(this,new String[]{
                        Manifest.permission.CAMERA,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                },REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId,stateCallback,null);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
}