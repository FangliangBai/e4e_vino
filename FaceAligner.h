#pragma once
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/matrix.h>

using namespace dlib;

namespace FaceAlignment {

typedef std::vector<point> PointVector;

enum AlignmentType { VM, CELEBA };

class CRange {
 public:
  CRange(int lb, int ub, int step) {
    m_lowerBound = lb;
    m_upperBound = ub;
    m_step = step;
  }

  ~CRange(){};

  std::vector<int> toArray() {
    std::vector<int> intArray;
    if (m_lowerBound == m_upperBound) {
      intArray.push_back(m_lowerBound);
    } else if (m_lowerBound < m_upperBound && m_step > 0) {
      for (int i = m_lowerBound; i <= m_upperBound; i += m_step) {
        intArray.push_back(i);
      }
    } else if (m_lowerBound > m_upperBound && m_step < 0) {
      for (int i = m_lowerBound; i >= m_upperBound; i += m_step) {
        intArray.push_back(i);
      }
    }
    return intArray;
  }

  CRange& operator=(const CRange& other) {
    if (this != &other) {
      m_lowerBound = other.m_lowerBound;
      m_upperBound = other.m_upperBound;
      m_step = other.m_step;
    }
    return *this;
  }

 private:
  int m_lowerBound;
  int m_upperBound;
  int m_step;
};

class FaceAligner {
 public:
  FaceAligner(shape_predictor& _sp, dpoint _desiredLeftEye, int _desiredFaceWidth, int _desiredFaceHeight);
  FaceAligner(shape_predictor& _sp, dpoint _desiredLeftEye, int _desiredFaceWidth);
  FaceAligner(shape_predictor& _sp, int _desiredFaceWidth);  // For CelebA style face alignment
  ~FaceAligner();

  array2d<rgb_pixel> align(array2d<rgb_pixel>& image, rectangle& rect, AlignmentType type = AlignmentType::VM);

 private:
  std::vector<int> getLandmarkIndexes(std::map<std::string, std::vector<int>>& bounds, std::string feat);
  std::map<std::string, std::vector<int>> buildBounds_68();
  array2d<rgb_pixel> align_celeba(array2d<rgb_pixel>& image, rectangle& rect);
  array2d<rgb_pixel> align_vm(array2d<rgb_pixel>& image, rectangle& rect);

 private:
  shape_predictor sp;
  dpoint desiredLeftEye;
  int desiredFaceWidth;
  int desiredFaceHeight;
  std::vector<std::string> feats;
  std::map<std::string, std::vector<int>> bounds_68;
};

}  // namespace FaceAlignment