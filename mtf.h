#pragma  once

#include <vector>
#include <opencv2/core.hpp>

struct LineData {
    LineData(int y2);

    int x1;
    int y1;
    int x2;
    int y2;
    cv::Mat mat;
    LineData(cv::Mat _mat, int _x1, int _y1, int _x2, int _y2) : x1(_x1), y1(_y1), x2(_x2), y2(_y2), mat(_mat) {}
};

/**
 * @brief   Calculate MTF at point.
 * @param   std::vector<uint8_t>&linePixels     Array of pixels
 *
 * This function calculates MTF in region by counting number of gray pixels
 * that are too far from brightest and darkest pixels in the region.
 */
double calculateMTFbyRange(std::vector<uint8_t> &linePixels);

/**
 * @brief   Calculate MTF at point.
 * @param   std::vector<uint8_t> &linePixels    Array of pixels
 * @param   double th                           Rate of pixels to be used as min/max
 *
 * This function calculates MTF in region using contrast formula,
 * using sum of @c th % of brightest/darkest pixels as min/max value in the formula
 */
double calculateMTFbyPercent(std::vector<uint8_t> &linePixels);

/**
 * @brief   Calculate MTF at point.
 * @param   std::vector<uint8_t>&linePixels     Array of pixels
 *
 * This function calculates MTF in region using contrast formula.
 */
double calculateMTFbyContrast(std::vector<uint8_t> &linePixels);

/**
 * @brief   Generate image containing info about MTF at every pixel.
 * @param   cv::Mat tg          Image to process
 * @param   int cell            Cell size in pixels
 * @param   bool showDebugInfo  If <b>true</b>, display intermediate results
 *
 */
cv::Mat processTargetCheckers(cv::Mat tg, int cell, bool showDebugInfo);

cv::Mat processTargetStar(cv::Mat tg, int step, bool showDebugInfo);


/**
 * @brief   Apply discrete Fourier transformation
 * @param   cv::Mat img          Image to process
 *
 */
cv::Mat applyDFT(cv::Mat img);

void processLine(cv::Mat tg, int i, int j, int cell, int x, int y, bool showDebugInfo, double noise);

double getNoise(cv::Mat mat);

cv::Mat getROI(cv::Mat img, int x, int y, int cell);
