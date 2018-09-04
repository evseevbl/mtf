#pragma  once

#include <vector>
#include <opencv2/core.hpp>


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
 * @param   std::string name    Filename to open
 * @param   int cell            Cell size in pixels
 * @param   bool showDebugInfo  If @c true, display intermediate results
 *
 */
cv::Mat processTarget(std::string name, int cell, bool showDebugInfo);
