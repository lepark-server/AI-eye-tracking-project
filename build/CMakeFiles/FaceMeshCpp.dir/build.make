# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/FaceMeshCpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FaceMeshCpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceMeshCpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceMeshCpp.dir/flags.make

CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o: ../src/demo.cpp
CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/demo.cpp

CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/demo.cpp > CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/demo.cpp -o CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.s

CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o: ../src/ModelLoader.cpp
CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/ModelLoader.cpp

CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/ModelLoader.cpp > CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/ModelLoader.cpp -o CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.s

CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o: ../src/DetectionPostProcess.cpp
CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/DetectionPostProcess.cpp

CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/DetectionPostProcess.cpp > CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/DetectionPostProcess.cpp -o CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.s

CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o: ../src/IrisLandmark.cpp
CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/IrisLandmark.cpp

CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/IrisLandmark.cpp > CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/IrisLandmark.cpp -o CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.s

CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o: ../src/FaceLandmark.cpp
CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceLandmark.cpp

CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceLandmark.cpp > CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceLandmark.cpp -o CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.s

CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o: CMakeFiles/FaceMeshCpp.dir/flags.make
CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o: ../src/FaceDetection.cpp
CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o: CMakeFiles/FaceMeshCpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o -MF CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o.d -o CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o -c /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceDetection.cpp

CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceDetection.cpp > CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.i

CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/src/FaceDetection.cpp -o CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.s

# Object files for target FaceMeshCpp
FaceMeshCpp_OBJECTS = \
"CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o" \
"CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o" \
"CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o" \
"CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o" \
"CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o" \
"CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o"

# External object files for target FaceMeshCpp
FaceMeshCpp_EXTERNAL_OBJECTS =

FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/demo.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/ModelLoader.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/DetectionPostProcess.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/IrisLandmark.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/FaceLandmark.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/src/FaceDetection.cpp.o
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/build.make
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
FaceMeshCpp: ../tflite-dist/libs/linux_x64/libtensorflowlite.so
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
FaceMeshCpp: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
FaceMeshCpp: CMakeFiles/FaceMeshCpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable FaceMeshCpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceMeshCpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceMeshCpp.dir/build: FaceMeshCpp
.PHONY : CMakeFiles/FaceMeshCpp.dir/build

CMakeFiles/FaceMeshCpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceMeshCpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceMeshCpp.dir/clean

CMakeFiles/FaceMeshCpp.dir/depend:
	cd /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build /home/lok/Work/Morning_Project/AI/mediapipe_face_iris_cpp/build/CMakeFiles/FaceMeshCpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceMeshCpp.dir/depend
