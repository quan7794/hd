package vn.com.tma.hdnghia.hairrecognite;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

public class MyCameraActivity extends Activity {
    private static final int CAMERA_REQUEST = 1888;
    private ImageView imageView;
    private static final int MY_CAMERA_PERMISSION_CODE = 100;
    private final int CAMERA_RESULT = 1;

    private final String Tag = getClass().getName();
//    static final int REQUEST_IMAGE_CAPTURE = 1;
//    private Bitmap mImageBitmap;
//    private String mCurrentPhotoPath;
//    private ImageView mImageView;

    Mat matrix2_grabcut;
    Mat matrix3_skindetection;
    Mat matrix5_grabcut_quantized;
    Mat matrix6_skin_quantized;
    Mat matrix7_output;
    Mat erosion_dilutionMatrix;
    Mat matrix8_max_contour;
    Mat matrix9_finalOutput;
    Mat mat;
    ImageView imageView1;
    Button photoButton;
    TextView loading;
    //Uri uri  = Uri.parse("file:///sdcard/photo.jpg");
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.cameraview);
        imageView1 = (ImageView)findViewById(R.id.imageView1);
       // photoButton = (Button) this.findViewById(R.id.button1);
        loading = this.findViewById(R.id.txtLoading);
        photoButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                            MY_CAMERA_PERMISSION_CODE);
                } else {
                    Intent photo = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    Uri uri  = Uri.parse(Environment.getExternalStorageDirectory().getPath()+"photo.jpg");
                    photo.putExtra(android.provider.MediaStore.EXTRA_OUTPUT, uri);
                    startActivityForResult(photo,CAMERA_REQUEST);
                }
            }
        });
    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    mat=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d("OpenCV","Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this, mLoaderCallback);
        }else{
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        imageView1 = null;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new
                        Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }

        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.i(Tag, "Receive the camera result");
        if (requestCode == CAMERA_REQUEST ) {
                //                Bitmap photo = (Bitmap) data.getExtras().get("data");
                //            imageView.setImageBitmap(photo);
                            //mat = new Mat();
                //            Bitmap bmp32 = photo.copy(Bitmap.Config.ARGB_8888, true);
                //            Utils.bitmapToMat(bmp32, mat);
                //            Mat dst = new Mat();
                //            Imgproc.cvtColor(mat,dst,Imgproc.COLOR_BGRA2BGR);
                //            grabCut(dst);
                //            skinSegmentation();
                //            File out = new File(getFilesDir(), "newImage.jpg");
//                            File out = new File(Environment.getExternalStorageDirectory().getPath(), "photo.jpg");
            File out = new File(Environment.getExternalStorageDirectory().getPath(), "photo.jpg");
            if(!out.exists()) {
                Toast.makeText(getBaseContext(),"Error while capturing image", Toast.LENGTH_LONG).show();
                return;
            }
            Toast.makeText(getBaseContext(),"Save image success!", Toast.LENGTH_LONG).show();
            Bitmap mBitmap = BitmapFactory.decodeFile(out.getAbsolutePath());
            imageView1.setImageBitmap(mBitmap);
//            Bitmap bmp32 = mBitmap.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mBitmap, mat);
            Mat dst = new Mat();
            Imgproc.cvtColor(mat,dst,Imgproc.COLOR_BGRA2BGR);
            loading.setVisibility(View.VISIBLE);
            grabCut(dst);
            skinSegmentation();
//            photoButton.setText("1");
            loading.setVisibility(View.INVISIBLE);
//            imageView1.setImageBitmap(mBitmap);



        }
        Toast.makeText(getBaseContext(),"Error", Toast.LENGTH_LONG).show();
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
        Utils.matToBitmap(matrix2_grabcut, temp);
        imageView1.setImageBitmap(temp);
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

        Bitmap temp=null;
        Utils.matToBitmap(matrix3_skindetection, temp);
        imageView1.setImageBitmap(temp);
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory()+"/"+ "skinDetection" +".jpg",matrix3_skindetection);

    }



}