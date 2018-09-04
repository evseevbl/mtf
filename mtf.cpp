#include "mtf.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double calculateMTFbyContrast(std::vector<uint8_t> &linePixels) {
    double mx = *std::max_element(linePixels.begin(), linePixels.end());
    double mn = *std::min_element(linePixels.begin(), linePixels.end());
    return (mx - mn) / (mx + mn);
}

double calculateMTFbyRange(std::vector<uint8_t> &linePixels) {
    double th = 0.1;
    int borderThreshold = 1;
    std::sort(linePixels.begin(), linePixels.end());
    double minval = *linePixels.begin() * (1 + th);
    double maxval = *(linePixels.end() - 1) * (1 - th);
    int cnt = 0;
    for (size_t i = 0; i != linePixels.size(); ++i) {
        if (maxval > linePixels[i] && linePixels[i] > minval) {
            ++cnt;
        }
    }
    return std::min(1.0, 1.0 * (cnt + borderThreshold) / linePixels.size());
}

double calculateMTFbyPercent(std::vector<uint8_t> &linePixels) {
    std::sort(linePixels.begin(), linePixels.end());
    double pc = 0.25;
    int cap = linePixels.size() * pc;
    double mn = std::accumulate(linePixels.begin(), linePixels.begin() + cap, 0) / linePixels.size();
    double mx = std::accumulate(linePixels.end() - cap, linePixels.end(), 0) / linePixels.size();
    return (mx - mn) / (mx + mn);
}


cv::Mat getROI(cv::Mat img, int x, int y, int cell) {
    return img(cv::Range(x, x + cell), cv::Range(y, y + cell));
}

std::pair<cv::Point, cv::Point> getBrightestDarkest(cv::Mat roi) {
    cv::Point mn, mx;
    double mn_val, mx_val;
    cv::minMaxLoc(roi, &mn_val, &mx_val, &mn, &mx);
    return std::make_pair(mx, mn);
};

cv::Mat processTarget(cv::Mat tg, int cell, bool showDebugInfo = true) {
    std::string ext = ".jpg";
    int rows = tg.rows / cell;
    int cols = tg.cols / cell;
    double val;
    cv::Mat roi, roi_gray;
    cv::Mat mtf1(rows, cols, CV_8UC1);
    cv::Mat mtf2(rows, cols, CV_8UC1);
    auto blurSize = cv::Size(3, 3);
    auto blurPoint = cv::Point(-1, -1);

    // slightly blur the image
    cv::blur(tg, tg, blurSize, blurPoint);
    for (int i = 0; i != rows; ++i) {
        for (int j = 0; j != cols; ++j) {
            roi = getROI(tg, i * cell, j * cell, cell);
            cv::cvtColor(roi, roi_gray, cv::COLOR_RGB2GRAY);
            auto ret = getBrightestDarkest(roi_gray);
            cv::Point mx = ret.first;
            cv::Point mn = ret.second;
            std::vector<cv::Point> linePixels;
            std::vector<uint8_t> vec;
            if (showDebugInfo) {
                cv::line(roi, mx, mn, cv::Scalar(255 * ((i + j) % 2), 0, 255));
            }
            for (cv::LineIterator li(roi, mx, mn); li.pos() != mn; ++li) {
                linePixels.push_back(li.pos());
                vec.push_back(roi_gray.at<uint8_t>(li.pos()));
            }
            val = calculateMTFbyContrast(vec);
            mtf1.at<uint8_t>(i, j) = static_cast<uint8_t >(val * 255);
            val = calculateMTFbyPercent(vec);
            mtf2.at<uint8_t>(i, j) = static_cast<uint8_t >(val * 255);
        }
    }
    cv::Mat mtf_cr, mtf_pr;
    cv::blur(mtf1, mtf1, blurSize, blurPoint);
    cv::blur(mtf2, mtf2, blurSize, blurPoint);
    cv::resize(mtf1, mtf_cr, tg.size(), cv::INTER_AREA);
    cv::resize(mtf2, mtf_pr, tg.size(), cv::INTER_AREA);
    cv::Mat blend;
    double alpla = 0.7, beta = 0.3;
    cv::addWeighted(mtf_cr, alpla, mtf_pr, beta, 0, blend);

    if (showDebugInfo) {
        cv::imshow("tg", tg);
        cv::imshow("roi", roi);
        cv::imshow("mtf1", mtf1);
        cv::imshow("mtf_cr", mtf_cr);
        cv::imshow("mtf_pr", mtf_pr);
        cv::imshow("blend", blend);
        cv::waitKey(0);
    }
    return blend;
}
