package vn.com.tma.hdnghia.hairrecognite;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

import static android.provider.Settings.System.DATE_FORMAT;

public class CameraActivity2 extends Activity{
    private static final int CAMERA_PHOTO = 111;
    private Uri imageToUploadUri;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    ImageView imageView;
    TextView loading;
    Mat matrix2_grabcut;
    Mat matrix3_skindetection;
    Mat matrix5_grabcut_quantized;
    Mat matrix6_skin_quantized;
    Mat matrix7_output;
    Mat erosion_dilutionMatrix;
    Mat matrix8_max_contour;
    Mat matrix9_finalOutput;
    Mat mat;
//    Mat dst;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    mat=new Mat();
//                    dst=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if(Build.VERSION.SDK_INT>=24){
            try{
                Method m = StrictMode.class.getMethod("disableDeathOnFileUriExposure");
                m.invoke(null);
            }catch(Exception e){
                e.printStackTrace();
            }
        }
        setContentView(R.layout.cameraview);
        imageView = findViewById(R.id.imageView1);
        loading = findViewById(R.id.txtLoading);
        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    MY_CAMERA_PERMISSION_CODE);
        } else {
            Intent chooserIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            File f = new File(Environment.getExternalStorageDirectory(), "POST_IMAGE.jpg");
            chooserIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(f));
            imageToUploadUri = Uri.fromFile(f);
            startActivityForResult(chooserIntent, CAMERA_PHOTO);
        }

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    public static Mat rotate(Mat src, double angle)
    {
        Mat dst = new Mat();
        if(angle == 180 || angle == -180) {
            Core.flip(src, dst, -1);
        } else if(angle == 90 || angle == -270) {
            Core.flip(src.t(), dst, 1);
        } else if(angle == 270 || angle == -90) {
            Core.flip(src.t(), dst, 0);
        }

        return dst;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_PHOTO && resultCode == Activity.RESULT_OK) {
            if(imageToUploadUri != null){
                Uri selectedImage = imageToUploadUri;
                getContentResolver().notifyChange(selectedImage, null);
                Bitmap reducedSizeBitmap = getBitmap(imageToUploadUri.getPath());
//                Bitmap reducedSizeBitmap = getBitmap(Environment.getExternalStorageDirectory()+"/quan.jpg");
                if(reducedSizeBitmap != null){
                    imageView.setRotation(90);
                    Bitmap bmp32 = reducedSizeBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    imageView.setImageBitmap(bmp32);
                    Utils.bitmapToMat(bmp32, mat);
                    loading.setVisibility(View.VISIBLE);

                    Mat newMat = new Mat(bmp32.getWidth(), bmp32.getHeight(), CvType.CV_8UC3);
                    Imgproc.cvtColor(mat,newMat,Imgproc.COLOR_RGB2BGR);
                    newMat=rotate(newMat,90);
                    //Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "dstTest" +".jpg",newMat);
                    grabCut(newMat);
                    skinSegmentation();
                    setQuantizedImages();
                    findImageDifference(newMat);
                    performErosion_Dilution();
                    findContours(newMat);
                    predict_hair();
                    loading.setVisibility(View.INVISIBLE);
                }else{
                    Toast.makeText(this,"Error while capturing Image",Toast.LENGTH_LONG).show();
                }
            }else{
                Toast.makeText(this,"Error while capturing Image",Toast.LENGTH_LONG).show();
            }
        }
    }
//    public void Capture(View view) {
//
//    }
    private Bitmap getBitmap(String path) {

        Uri uri = Uri.fromFile(new File(path));
        InputStream in = null;
        try {
            final int IMAGE_MAX_SIZE = 100000; // 1.2MP
            try {
                in = getContentResolver().openInputStream(uri);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            // Decode image size
            BitmapFactory.Options o = new BitmapFactory.Options();
            o.inJustDecodeBounds = true;
            BitmapFactory.decodeStream(in, null, o);
            try {
                in.close();
            } catch (IOException e) {
                e.printStackTrace();
            }


            int scale = 1;
            while ((o.outWidth * o.outHeight) * (1 / Math.pow(scale, 2)) >
                    IMAGE_MAX_SIZE) {
                scale++;
            }
            Log.d("", "scale = " + scale + ", orig-width: " + o.outWidth + ", orig-height: " + o.outHeight);

            Bitmap b = null;
            in = getContentResolver().openInputStream(uri);
            if (scale > 1) {
                scale--;
                // scale to max possible inSampleSize that still yields an image
                // larger than target
                o = new BitmapFactory.Options();
                o.inSampleSize = scale;
                b = BitmapFactory.decodeStream(in, null, o);

                // resize to desired dimensions
                int height = b.getHeight();
                int width = b.getWidth();
                Log.d("", "1th scale operation dimenions - width: " + width + ", height: " + height);

                double y = Math.sqrt(IMAGE_MAX_SIZE
                        / (((double) width) / height));
                double x = (y / height) * width;

                Bitmap scaledBitmap = Bitmap.createScaledBitmap(b, (int) x,
                        (int) y, true);
                b.recycle();
                b = scaledBitmap;

                System.gc();
            } else {
                b = BitmapFactory.decodeStream(in);
            }
            in.close();

            Log.d("", "bitmap size - width: " + b.getWidth() + ", height: " +
                    b.getHeight());
            return b;
        } catch (IOException e) {
            Log.e("", e.getMessage(), e);
            return null;
        }
    }
    private void grabCut(Mat srImage){
        Mat sourceImage = srImage;

        Mat result = new Mat(sourceImage.size(),sourceImage.type());
        Mat bgModel = new Mat();    //background model
        Mat fgModel = new Mat();    //foreground model

        //draw a rectangle
        Rect rectangle = new Rect(1,1,sourceImage.cols()-1,sourceImage.rows()-1);

        Imgproc.grabCut(sourceImage, result,rectangle, bgModel,fgModel,10,Imgproc.GC_INIT_WITH_RECT);
        Core.compare(result,new Scalar(3,3,3),result,Core.CMP_EQ);
        matrix2_grabcut = new Mat(sourceImage.size(), CvType.CV_8UC3,new Scalar(255,255,255));
        sourceImage.copyTo(matrix2_grabcut, result);
        Bitmap temp=null;
        //Utils.matToBitmap(matrix2_grabcut, temp);
        //imageView.setImageBitmap(temp);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "grabcut" +".jpg",matrix2_grabcut);

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

        //Bitmap temp=null;
        //Utils.matToBitmap(matrix3_skindetection, temp);
        //imageView.setImageBitmap(temp);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "skinDetection" +".jpg",matrix3_skindetection);

    }
    private void setQuantizedImages() {
        matrix5_grabcut_quantized = this.quantizeImage(matrix2_grabcut);
        matrix6_skin_quantized = this.quantizeImage(matrix3_skindetection);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "matrix5_grabcut_quantized" +".jpg",matrix5_grabcut_quantized);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "matrix6_skin_quantized" +".jpg",matrix6_skin_quantized);
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

    private void findImageDifference(Mat dst) {
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
                    Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "matrix7_output" +".jpg",matrix7_output);

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
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "erosion_dilutionMatrix" +".jpg",erosion_dilutionMatrix);

            /*
            //dilation
            Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
            Imgproc.dilate(erosion_dilutionMatrix, erosion_dilutionMatrix, element2);
            */
    }

    private void findContours(Mat dst){

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
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "matrix8_max_contour" +".jpg",matrix8_max_contour);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "matrix9_finalOutput" +".jpg",matrix9_finalOutput);


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
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "histImage" +".jpg",histImage);

        return histImage;
    }
}
