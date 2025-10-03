sudo apt update
sudo apt install -y libopencv-dev pkg-config clang libclang-dev llvm-dev
# (opsional, kalau pakai MySQL)
sudo apt install -y libmysqlclient-dev

pkg-config --modversion opencv4

export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
# dan/atau (kalau tetap gagal) set manual:
export OPENCV_INCLUDE_PATHS=/usr/include/opencv4
export OPENCV_LIB_PATHS=/usr/lib/aarch64-linux-gnu
export OPENCV_LINK_LIBS="opencv_core,opencv_imgproc,opencv_dnn,opencv_highgui,opencv_videoio"
