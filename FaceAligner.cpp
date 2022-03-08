#include "FaceAligner.h"
#include <numeric>
#include <vector>
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/opencv/cv_image.h"
#include "dlib/opencv/to_open_cv.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace dlib;
using namespace std;

namespace FaceAlignment {
FaceAligner::FaceAligner(shape_predictor& _sp, dpoint _desiredLeftEye, int _desiredFaceWidth, int _desiredFaceHeight) {
  sp = _sp;
  desiredLeftEye = _desiredLeftEye;
  desiredFaceWidth = _desiredFaceWidth;
  desiredFaceHeight = _desiredFaceHeight;
  feats.push_back("le");
  feats.push_back("re");
  feats.push_back("mouth");
  feats.push_back("nose");
  feats.push_back("leb");
  feats.push_back("reb");
  feats.push_back("fp");
  feats.push_back("mouth_outer");
  feats.push_back("mouth_inner");
  bounds_68 = buildBounds_68();
}

FaceAligner::FaceAligner(shape_predictor& _sp, dpoint _desiredLeftEye, int _desiredFaceWidth) {
  sp = _sp;
  desiredLeftEye = _desiredLeftEye;
  desiredFaceWidth = _desiredFaceWidth;
  desiredFaceHeight = _desiredFaceWidth;
  feats.push_back("le");
  feats.push_back("re");
  feats.push_back("mouth");
  feats.push_back("nose");
  feats.push_back("leb");
  feats.push_back("reb");
  feats.push_back("fp");
  feats.push_back("mouth_outer");
  feats.push_back("mouth_inner");
  bounds_68 = buildBounds_68();
}

FaceAligner::FaceAligner(shape_predictor& _sp, int _desiredFaceWidth) {
  sp = _sp;
  desiredFaceWidth = _desiredFaceWidth;
  desiredFaceHeight = _desiredFaceWidth;
  feats.push_back("le");
  feats.push_back("re");
  feats.push_back("mouth");
  feats.push_back("nose");
  feats.push_back("leb");
  feats.push_back("reb");
  feats.push_back("fp");
  feats.push_back("mouth_outer");
  feats.push_back("mouth_inner");
  bounds_68 = buildBounds_68();
}

FaceAligner::~FaceAligner() {}

std::map<string, std::vector<int>> FaceAligner::buildBounds_68() {
  std::map<string, std::vector<int>> bounds;
  bounds["leb"] = std::vector<int>({17, 21});
  bounds["le"] = std::vector<int>({36, 41});
  bounds["reb"] = std::vector<int>({22, 26});
  bounds["re"] = std::vector<int>({42, 47});
  bounds["nose"] = std::vector<int>({28, 35});
  bounds["mouth"] = std::vector<int>({48, 67});
  bounds["fp"] = std::vector<int>({0, 16});
  bounds["mouth_outer"] = std::vector<int>({48, 59});
  bounds["mouth_inner"] = std::vector<int>({60, 67});
  return bounds;
}

std::vector<int> FaceAligner::getLandmarkIndexes(std::map<string, std::vector<int>>& bounds, std::string feat) {
  std::vector<int> indexes;
  std::vector<int> fb = bounds[feat];
  for (int i = 0; i < fb.size() / 2; i++) {
    // std::cout << "lb:" << fb[2 * i] << " hb:" << fb[2 * i + 1] << std::endl;
    std::vector<int> indexes_ = CRange(fb[2 * i], fb[2 * i + 1], 1).toArray();
    indexes.insert(indexes.end(), indexes_.begin(), indexes_.end());
  }
  return indexes;
}

/*dpoint FaceAligner::calcFeatCentre(PointVector& featPts)
{
        dpoint centre = dpoint(0.0, 0.0);

        for (int i = 0; i < featPts.size(); i++)
        {
                centre += featPts[i];
        }

        centre /= featPts.size();
        return centre;
}*/

array2d<rgb_pixel> FaceAligner::align(array2d<rgb_pixel>& image, rectangle& rect, AlignmentType type) {
  if (type == AlignmentType::VM)
    return align_vm(image, rect);
  else
    return align_celeba(image, rect);
}

array2d<rgb_pixel> FaceAligner::align_celeba(array2d<rgb_pixel>& image, rectangle& rect) {
  full_object_detection shape = sp(image, rect);
  std::map<string, PointVector> feat2points;
  //image_window win;
  //std::vector<full_object_detection> shapes;
  //shapes.push_back(shape);
  //win.clear_overlay();
  //win.set_image(image);
  //win.add_overlay(render_face_detections(shapes));
  //std::cout << "Press any key to continue..." << std::endl;
  //std::cin.get();

  for (int k = 0; k < feats.size(); ++k) {
    string feat = feats[k];
    std::vector<int> lm = getLandmarkIndexes(bounds_68, feat);
    PointVector feat_lm(lm.size());

    for (int ii = 0; ii < lm.size(); ii++) {
      feat_lm[ii] = shape.part(lm[ii]);
    }

    feat2points[feat] = feat_lm;
  }

  PointVector& leftEyePts = feat2points["le"];
  PointVector& rightEyePts = feat2points["re"];

  point leftEyeCentre = std::accumulate(leftEyePts.begin(), leftEyePts.end(), point(0, 0)) / leftEyePts.size();
  point rightEyeCentre = std::accumulate(rightEyePts.begin(), rightEyePts.end(), point(0, 0)) / rightEyePts.size();
  point eye_avg = (leftEyeCentre + rightEyeCentre) / 2;
  point eye_to_eye = rightEyeCentre - leftEyeCentre;
  point mouth_left = feat2points["mouth_outer"][0];
  point mouth_right = feat2points["mouth_outer"][6];
  point mouth_avg = (mouth_left + mouth_right) / 2;
  point eye_to_mouth = mouth_avg - eye_avg;

  dpoint x = eye_to_eye - point(-1 * eye_to_mouth.y(), eye_to_mouth.x());
  x /= x.length();
  x *= std::max<double>(eye_to_eye.length() * 2, eye_to_mouth.length() * 1.8);
  dpoint y = dpoint(-1 * x.y(), x.x());
  dpoint c = eye_avg + eye_to_mouth * 0.1;
  PointVector quad_pts({c - x - y, c - x + y, c + x + y, c + x - y});  // LB, LT, RT, RB
  std::vector<double> quad_val;
  for (PointVector::const_iterator it = quad_pts.begin(); it != quad_pts.end(); ++it) {
    quad_val.push_back((*it).x());
    quad_val.push_back((*it).y());
  }
  dlib::matrix<double, 4, 2> quad = dlib::reshape(dlib::mat(quad_val), 4, 2);

  double qsize = x.length() * 2;

  // Shrink
  double shrink = std::floor(qsize / desiredFaceWidth * 0.5);
  cv::Mat img = dlib::toMat(image);
  if (shrink > 1) {
    cv::Size siz((int)std::round(image.nc() / shrink), (int)std::round(image.nr() / shrink));
    cv::resize(img, img, siz, cv::INTER_LINEAR);
    quad /= shrink;
    qsize /= shrink;
  }

  // Crop.
  int border = std::max<int>((int)std::round(qsize * 0.1), 3);
  double min_quad_0, max_quad_0, min_quad_1, max_quad_1;
  min_quad_0 = dlib::min(dlib::subm(quad, 0, 0, 4, 1));
  max_quad_0 = dlib::max(dlib::subm(quad, 0, 0, 4, 1));
  min_quad_1 = dlib::min(dlib::subm(quad, 0, 1, 4, 1));
  max_quad_1 = dlib::max(dlib::subm(quad, 0, 1, 4, 1));
  int crop[4] = {(int)std::floor(min_quad_0), (int)std::floor(min_quad_1), (int)std::ceil(max_quad_0),
                 (int)std::ceil(max_quad_1)};
  crop[0] = std::max<int>(crop[0] - border, 0);                  // Left
  crop[1] = std::max<int>(crop[1] - border, 0);                  // Bottom
  crop[2] = std::min<int>(crop[2] + border, img.size().width);   // Right
  crop[3] = std::min<int>(crop[3] + border, img.size().height);  // Top
  cv::Mat croppedImg;
  if (crop[2] - crop[0] < img.size().width || crop[3] - crop[1] < img.size().height) {
    img(cv::Rect(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1])).copyTo(croppedImg);
    for (int i = 0; i < 4; i++) {
      quad(i, 0) -= crop[0];
      quad(i, 1) -= crop[1];
    }
  } else
    img.copyTo(croppedImg); // No cropping

  // Pad
  std::vector<int> pad({(int)std::floor(min_quad_0), (int)std::floor(min_quad_1), (int)std::ceil(max_quad_0),
                (int)std::ceil(max_quad_1)});
  pad[0] = std::max<int>(-pad[0] + border, 0);
  pad[1] = std::max<int>(-pad[1] + border, 0);
  pad[2] = std::max<int>(pad[2] - croppedImg.size[0] + border, 0);
  pad[3] = std::max<int>(pad[3] - croppedImg.size[1] + border, 0);
  cv::Mat paddedImg;
  int max_padding = *std::max_element(std::begin(pad), std::end(pad));
  if (max_padding > border - 4) {
    int a = (int)std::round(qsize * 0.3);
    std::for_each(pad.begin(), pad.end(), [a](int& elem) -> void {  // Equivalent to np.maximum
      if (elem < a) elem = a;
    });
    cv::copyMakeBorder(croppedImg, paddedImg, pad[1], pad[3], pad[0], pad[2], cv::BORDER_REFLECT);
    int w = paddedImg.size().width, h = paddedImg.size().height;
    // y, x, _ = np.ogrid[:h, :w, :1]
    cv::Mat ogrid_y(h, 1, CV_32F), ogrid_x(1, w, CV_32F);
    std::generate(ogrid_y.begin<float>(), ogrid_y.end<float>(), [n=0]() mutable {
      return n++;
    });
    std::generate(ogrid_x.begin<float>(), ogrid_x.end<float>(), [n=0]() mutable {
      return n++;
    });
    cv::Mat x1 = ogrid_x / pad[0], x2 = (w - 1 - ogrid_x) / pad[2];
    cv::MatIterator_<float> it = ogrid_x.begin<float>();
    cv::MatConstIterator_<float> it1 = x1.begin<float>();
    cv::MatConstIterator_<float> it2 = x2.begin<float>();
    for (; it != ogrid_x.end<float>(); ++it, ++it1, ++it2) {
      if (*it1 < *it2)
        *it = 1 - (*it1);
      else
        *it = 1 - (*it2);
    }
    cv::Mat y1 = ogrid_y / pad[1], y2 = (h - 1 - ogrid_y) / pad[3];
    it = ogrid_y.begin<float>();
    it1 = y1.begin<float>();
    it2 = y2.begin<float>();
    for (; it != ogrid_y.end<float>(); ++it, ++it1, ++it2) {
      if (*it1 < *it2)
        *it = 1 - (*it1);
      else
        *it = 1 - (*it2);
    }
    cv::Mat grid_x, grid_y, mask(h, w, CV_32F);
    cv::repeat(ogrid_x, h, 1, grid_x);
    cv::repeat(ogrid_y, 1, w, grid_y);
    it = mask.begin<float>();
    it1 = grid_x.begin<float>();
    it2 = grid_y.begin<float>();
    for (; it != mask.end<float>(); ++it, ++it1, ++it2) {
      if (*it1 > *it2)
        *it = *it1;
      else
        *it = *it2;
    }

    ///// img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    /////////
    double sigma = qsize * 0.02;
    cv::Mat blurredImg, residual, clippedMask;
    cv::GaussianBlur(paddedImg, blurredImg, cv::Size(0, 0), sigma);
    cv::Mat blurredImg_32f, paddedImg_32f;
    blurredImg.convertTo(blurredImg_32f, CV_32F);
    paddedImg.convertTo(paddedImg_32f, CV_32F);
    residual = blurredImg_32f - paddedImg_32f;
    cv::Mat channels[3];
    cv::split(residual, channels); // Allocate new memory
    clippedMask = mask * 3.0 + 1.0;
    clippedMask.setTo(0, clippedMask < 0);
    clippedMask.setTo(1.0, clippedMask > 1.0);
    for (int i = 0; i < 3; i++) channels[i] = channels[i].mul(clippedMask);
    cv::merge(channels, 3, residual);
    paddedImg_32f += residual;

    /////// img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0) //////
    cv::Mat dst_chnls[3];
    cv::split(paddedImg_32f, dst_chnls);
    cv::split(paddedImg_32f, channels);  // cv::split always allocates new memory
    mask.setTo(0, mask < 0);
    mask.setTo(1.0, mask > 1.0);
    for (int i = 0; i < 3; i++) {
      // Find the median of pixels in each channel
      int siz = channels[i].total();  // channels[i].size().width * channels[i].size().height;
      auto m = channels[i].begin<float>() + siz / 2;
      std::nth_element(channels[i].begin<float>(), m, channels[i].end<float>());
      dst_chnls[i] = mask.mul(*m - dst_chnls[i]);
    }

    cv::Mat tmpimg;
    cv::merge(dst_chnls, 3, tmpimg);
    paddedImg_32f += tmpimg;
    paddedImg_32f.setTo(0, paddedImg_32f < 0);
    paddedImg_32f.setTo(255, paddedImg_32f > 255);
    paddedImg_32f.convertTo(paddedImg, CV_8U);
    for (int i = 0; i < quad.nr(); i++) {
      quad(i, 0) += pad[0];
      quad(i, 1) += pad[1];
    }
  } else { // No padding
    croppedImg.copyTo(paddedImg);
  }
  cv::imwrite("padded.png", paddedImg);

  // Transform
  ///// img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
  ///PIL.Image.BILINEAR)
  cv::Mat warpped_img;  //= cv::Mat::zeros(desiredFaceHeight, desiredFaceWidth, CV_8UC3);

  cv::Point2f src_pts[4], dst_pts[4];
  for (int i = 0; i < 4; i++) {
    src_pts[i].x = quad(i, 0);
    src_pts[i].y = quad(i, 1);
  }
  dst_pts[0].x = 0;
  dst_pts[0].y = 0;
  dst_pts[1].x = 0;
  dst_pts[1].y = desiredFaceHeight - 1;
  dst_pts[2].x = desiredFaceWidth - 1;
  dst_pts[2].y = desiredFaceHeight - 1;
  dst_pts[3].x = desiredFaceWidth - 1;
  dst_pts[3].y = 0;

  cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
  cv::warpPerspective(paddedImg, warpped_img, M, cv::Size(desiredFaceWidth, desiredFaceHeight), cv::INTER_LINEAR);

  // return the aligned face
  array2d<rgb_pixel> output_img(desiredFaceHeight, desiredFaceWidth);
  dlib::assign_image(output_img, cv_image<rgb_pixel>(warpped_img));

  return output_img;
}

array2d<rgb_pixel> FaceAligner::align_vm(array2d<rgb_pixel>& image, rectangle& rect) {
  full_object_detection shape = sp(image, rect);
  std::map<string, PointVector> feat2points;
  image_window win;
  std::vector<full_object_detection> shapes;
  shapes.push_back(shape);
  win.clear_overlay();
  win.set_image(image);
  win.add_overlay(render_face_detections(shapes));
  std::cout << "Press any key to continue..." << std::endl;
  std::cin.get();

  for (int k = 0; k < feats.size(); ++k) {
    string feat = feats[k];
    std::vector<int> lm = getLandmarkIndexes(bounds_68, feat);
    PointVector feat_lm(lm.size());

    for (int ii = 0; ii < lm.size(); ii++) {
      feat_lm[ii] = shape.part(lm[ii]);
    }

    feat2points[feat] = feat_lm;
  }

  PointVector& leftEyePts = feat2points["le"];
  PointVector& rightEyePts = feat2points["re"];

  point leftEyeCentre = std::accumulate(leftEyePts.begin(), leftEyePts.end(), point(0, 0)) / leftEyePts.size();
  point rightEyeCentre = std::accumulate(rightEyePts.begin(), rightEyePts.end(), point(0, 0)) / rightEyePts.size();

  // compute the angle between the eye centroids
  float dY = rightEyeCentre.y() - leftEyeCentre.y();
  float dX = rightEyeCentre.x() - leftEyeCentre.x();
  float angle = atan2f(dY, dX) / pi * 180;

  // compute the desired right eye x - coordinate based on the
  // desired x - coordinate of the left eye
  float desiredRightEyeX = 1.0 - desiredLeftEye.x();

  // determine the scale of the new resulting image by taking
  // the ratio of the distance between eyes in the *current*
  // image to the ratio of distance between eyes in the
  // *desired* image
  float dist = sqrtf(dX * dX + dY * dY);
  float desiredDist = (desiredRightEyeX - desiredLeftEye.x());
  desiredDist *= desiredFaceWidth;
  float scale = desiredDist / dist;

  // compute center(x, y) - coordinates(i.e., the median point)
  // between the two eyes in the input image
  float cx = (leftEyeCentre.x() + rightEyeCentre.x()) / 2;
  float cy = (leftEyeCentre.y() + rightEyeCentre.y()) / 2;
  cv::Point2f eyesCentre(cx, cy);

  // grab the rotation matrix for rotating and scaling the face
  cv::Mat M = cv::getRotationMatrix2D(eyesCentre, angle, scale);

  // update the translation component of the matrix
  float tX = desiredFaceWidth * 0.5f;
  float tY = desiredFaceHeight * desiredLeftEye.y();
  M.at<double>(0, 2) += (tX - eyesCentre.x);
  M.at<double>(1, 2) += (tY - eyesCentre.y);

  // apply the affine transformation
  cv::Mat warp_src = toMat(image);
  cv::Mat warp_dst = cv::Mat::zeros(desiredFaceHeight, desiredFaceWidth, CV_8UC3);
  cv::warpAffine(warp_src, warp_dst, M, warp_dst.size(), cv::INTER_CUBIC);

  // return the aligned face
  array2d<rgb_pixel> output_img(desiredFaceHeight, desiredFaceWidth);
  assign_image(output_img, cv_image<rgb_pixel>(warp_dst));
  return output_img;
}
}  // namespace FaceAlignment