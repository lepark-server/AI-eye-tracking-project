#include "IrisLandmark.hpp"
#include <iostream>
#include <iomanip> 
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define SHOW_FPS    (1)

#if SHOW_FPS
    #include <chrono>
#endif

//Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
//Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
double K[9] = { 1959.251759804, 0.0, 958.5, 0.0, 1469.438819852, 538.875, 0.0, 0.0, 1.0 };
double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };


int main(int argc, char* argv[]) {

    //fill in cam intrinsics and distortion coefficients
    cv::Mat cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    std::vector<cv::Point3d> object_pts;
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner
    
    //2D ref points(image coordinates), referenced from detected facial feature
    std::vector<cv::Point2d> image_pts;

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

    //reproject 3D points world coordinate axis to verify result pose
    std::vector<cv::Point3d> reprojectsrc;
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

    //reprojected 2D points
    std::vector<cv::Point2d> reprojectdst;
    reprojectdst.resize(8);

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

    //text on screen
    std::ostringstream outtext;
    
    my::IrisLandmark irisLandmarker("./models");
    cv::VideoCapture cap(0);




    bool success = cap.isOpened();
    if (success == false)
    {
        std::cerr << "Cannot open the camera." << std::endl;
        return 1;
    }

    #if SHOW_FPS
        float sum = 0;
        int count = 0;
    #endif

    while (success)
    {
        cv::Mat rframe, frame;
        success = cap.read(rframe); // read a new frame from video
        cv::resize(rframe, rframe, cv::Size(1920, 1080));
        if (success == false)
            break;
        
        cv::flip(rframe, rframe, 1);

        #if SHOW_FPS
            auto start = std::chrono::high_resolution_clock::now();
        #endif

        irisLandmarker.loadImageToInput(rframe);
        irisLandmarker.runInference();
        int i =0;
        // for (auto landmark: irisLandmarker.getAllFaceLandmarks()) {
        //     //cv::circle(rframe, landmark, 2, cv::Scalar(0, 255, 0), -1);
        //     cv::putText(rframe, std::to_string(i++), landmark, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 196, 255), 1);
        // }       
        auto landmark = irisLandmarker.getAllFaceLandmarks();
        if(landmark.size() == 468) {
            cv::putText(rframe, std::to_string(i++), landmark.at(156), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(55), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(285), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(276), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(130), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(243), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(463), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(359), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(129), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(358), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(61), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(291), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);
            cv::putText(rframe, std::to_string(i++), landmark.at(17), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            cv::putText(rframe, std::to_string(i++), landmark.at(152), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

            //fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
            image_pts.push_back(landmark.at(156)); //#17 left brow left corner
            image_pts.push_back(landmark.at(55)); //#21 left brow right corner

            image_pts.push_back(landmark.at(285));
            image_pts.push_back(landmark.at(276)); //#22 right brow left corner

            image_pts.push_back(landmark.at(130));
            image_pts.push_back(landmark.at(243));

            image_pts.push_back(landmark.at(463));
            image_pts.push_back(landmark.at(359));

            image_pts.push_back(landmark.at(129));
            image_pts.push_back(landmark.at(358));

            image_pts.push_back(landmark.at(61));
            image_pts.push_back(landmark.at(291));
            image_pts.push_back(landmark.at(17));

            image_pts.push_back(landmark.at(152));



    //calc pose
        cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

        //reproject
        cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

        //draw axis
        cv::line(rframe, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[1], reprojectdst[2], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[2], reprojectdst[3], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[3], reprojectdst[0], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[4], reprojectdst[5], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[5], reprojectdst[6], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[6], reprojectdst[7], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[7], reprojectdst[4], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[0], reprojectdst[4], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[1], reprojectdst[5], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[2], reprojectdst[6], cv::Scalar(0, 0, 255),2);
        cv::line(rframe, reprojectdst[3], reprojectdst[7], cv::Scalar(0, 0, 255),2);

        //calc euler angle
        cv::Rodrigues(rotation_vec, rotation_mat);
        cv::hconcat(rotation_mat, translation_vec, pose_mat);
        cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

        //show angle result
        outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);
        cv::putText(rframe, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
        outtext.str("");
        outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);
        cv::putText(rframe, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
        outtext.str("");
        outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);
        cv::putText(rframe, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
        outtext.str("");

        image_pts.clear();

        for (auto landmark: irisLandmarker.getAllEyeLandmarks(true, true)) {
            cv::circle(rframe, landmark, 2, cv::Scalar(0, 0, 255), -1);
        }

        for (auto landmark: irisLandmarker.getAllEyeLandmarks(false, true)) {
            cv::circle(rframe, landmark, 2, cv::Scalar(0, 0, 255), -1);
        }

        #if SHOW_FPS
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            float inferenceTime = duration.count() / 1e3;
            sum += inferenceTime;
            count += 1;
            int fps = (int) 1e3/ inferenceTime;

            cv::putText(rframe, std::to_string(fps), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 196, 255), 2);
        #endif

        cv::imshow("Face detector", rframe);

        if (cv::waitKey(10) == 27)
            break;
    }

    #if SHOW_FPS
        std::cout << "Average inference time: " << sum / count << "ms " << std::endl;
    #endif



    }

    cap.release();
    cv::destroyAllWindows();
 
    return 0;
}