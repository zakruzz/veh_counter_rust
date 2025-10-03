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

error: failed to run custom build command for `opencv v0.86.1`

Caused by:
  process didn't exit successfully: `/home/ahmadradhy/veh_counter_rust/target/debug/build/opencv-50304f7f6f47cacd/build-script-build` (exit status: 101)
  --- stdout
  cargo:rerun-if-env-changed=OPENCV4_NO_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=SYSROOT
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG
  cargo:rerun-if-env-changed=PKG_CONFIG
  cargo:rerun-if-env-changed=OPENCV4_STATIC
  cargo:rerun-if-env-changed=OPENCV4_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_STATIC
  cargo:rerun-if-env-changed=PKG_CONFIG_ALL_DYNAMIC
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_PATH
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64-unknown-linux-gnu
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR_aarch64_unknown_linux_gnu
  cargo:rerun-if-env-changed=HOST_PKG_CONFIG_SYSROOT_DIR
  cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
  cargo:rustc-cfg=ocvrs_opencv_branch_4

  --- stderr
  === Crate version: Some("0.86.1")
  === Environment configuration:
  ===   OPENCV_PACKAGE_NAME = None
  ===   OPENCV_PKGCONFIG_NAME = None
  ===   OPENCV_CMAKE_NAME = None
  ===   OPENCV_CMAKE_BIN = None
  ===   OPENCV_VCPKG_NAME = None
  ===   OPENCV_LINK_LIBS = None
  ===   OPENCV_LINK_PATHS = None
  ===   OPENCV_INCLUDE_PATHS = None
  ===   OPENCV_DISABLE_PROBES = None
  ===   OPENCV_MSVC_CRT = None
  ===   CMAKE_PREFIX_PATH = None
  ===   OpenCV_DIR = None
  ===   PKG_CONFIG_PATH = None
  ===   VCPKG_ROOT = None
  ===   VCPKGRS_DYNAMIC = None
  ===   VCPKGRS_TRIPLET = None
  ===   OCVRS_DOCS_GENERATE_DIR = None
  ===   DOCS_RS = None
  ===   PATH = Some("/usr/local/cuda-12.6/bin:/home/ahmadradhy/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin")
  === Enabled features:
  ===   ALPHAMAT
  ===   ARUCO
  ===   ARUCO_DETECTOR
  ===   BARCODE
  ===   BGSEGM
  ===   BIOINSPIRED
  ===   CALIB3D
  ===   CCALIB
  ===   CUDAARITHM
  ===   CUDABGSEGM
  ===   CUDACODEC
  ===   CUDAFEATURES2D
  ===   CUDAFILTERS
  ===   CUDAIMGPROC
  ===   CUDAOBJDETECT
  ===   CUDAOPTFLOW
  ===   CUDASTEREO
  ===   CUDAWARPING
  ===   CVV
  ===   DEFAULT
  ===   DNN
  ===   DNN_SUPERRES
  ===   DPM
  ===   FACE
  ===   FEATURES2D
  ===   FLANN
  ===   FREETYPE
  ===   FUZZY
  ===   GAPI
  ===   HDF
  ===   HFS
  ===   HIGHGUI
  ===   IMGCODECS
  ===   IMGPROC
  ===   IMG_HASH
  ===   INTENSITY_TRANSFORM
  ===   LINE_DESCRIPTOR
  ===   MCC
  ===   ML
  ===   OBJDETECT
  ===   OPTFLOW
  ===   OVIS
  ===   PHASE_UNWRAPPING
  ===   PHOTO
  ===   PLOT
  ===   QUALITY
  ===   RAPID
  ===   RGBD
  ===   SALIENCY
  ===   SFM
  ===   SHAPE
  ===   STEREO
  ===   STITCHING
  ===   STRUCTURED_LIGHT
  ===   SUPERRES
  ===   SURFACE_MATCHING
  ===   TEXT
  ===   TRACKING
  ===   VIDEO
  ===   VIDEOIO
  ===   VIDEOSTAB
  ===   VIZ
  ===   WECHAT_QRCODE
  ===   XFEATURES2D
  ===   XIMGPROC
  ===   XOBJDETECT
  ===   XPHOTO
  === Detected probe priority based on environment vars: pkg_config: false, cmake: false, vcpkg: false
  === Probing the OpenCV library in the following order: environment, pkg_config, cmake, vcpkg_cmake, vcpkg
  === Can't probe using: environment, continuing with other methods because: Some environment variables are missing
  === Probing OpenCV library using pkg_config
  === Successfully probed using: pkg_config
  === OpenCV library configuration: Library {
      include_paths: [
          "/usr/local/include/opencv4",
      ],
      version: Version {
          major: 4,
          minor: 8,
          patch: 0,
      },
      cargo_metadata: [
          "cargo:rustc-link-search=/usr/local/lib",
          "cargo:rustc-link-lib=opencv_gapi",
          "cargo:rustc-link-lib=opencv_highgui",
          "cargo:rustc-link-lib=opencv_ml",
          "cargo:rustc-link-lib=opencv_objdetect",
          "cargo:rustc-link-lib=opencv_photo",
          "cargo:rustc-link-lib=opencv_stitching",
          "cargo:rustc-link-lib=opencv_video",
          "cargo:rustc-link-lib=opencv_calib3d",
          "cargo:rustc-link-lib=opencv_features2d",
          "cargo:rustc-link-lib=opencv_dnn",
          "cargo:rustc-link-lib=opencv_flann",
          "cargo:rustc-link-lib=opencv_videoio",
          "cargo:rustc-link-lib=opencv_imgcodecs",
          "cargo:rustc-link-lib=opencv_imgproc",
          "cargo:rustc-link-lib=opencv_core",
      ],
  }

  thread 'main' panicked at /home/ahmadradhy/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/opencv-0.86.1/build.rs:363:10:
  Discovered OpenCV include paths is empty or contains non-existent paths
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...
