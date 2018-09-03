#include <iostream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat getROI(cv::Mat img, int x, int y, int cell) {
    return img(cv::Range(x, x + cell), cv::Range(y, y + cell));
}

std::pair<cv::Point, cv::Point> getBrightestDarkest(cv::Mat roi) {
    cv::Point mn, mx;
    double mn_val, mx_val;
    cv::minMaxLoc(roi, &mn_val, &mx_val, &mn, &mx);
    std::cout << "min=" << mn_val << " @ (" << mn.x << ";" << mn.y << ")";
    std::cout << " max=" << mx_val << " @ (" << mx.x << ";" << mx.y << ")" << std::endl;
    return std::make_pair(mx, mn);
};

double calculateMTFbyContrast(std::vector<uint8_t> &vec) {
    double mx = *std::max_element(vec.begin(), vec.end());
    double mn = *std::min_element(vec.begin(), vec.end());
    return (mx - mn) / (mx + mn);
}

double calculateMTFbyRange(std::vector<uint8_t> &vec) {
    double th = 0.5;
    std::sort(vec.begin(), vec.end());
    double minval = *vec.begin() * (1 + th);
    double maxval = *(vec.end() - 1) * (1 - th);
    int cnt = 0;
    for (size_t i = 0; i != vec.size(); ++i) {
        if (maxval > vec[i] && vec[i] > minval) {
            ++cnt;
        }
    }
    return (1.0 * cnt / vec.size());
}

int main() {
    auto tg = cv::imread("img/tg01.jpg", cv::IMREAD_COLOR);
    auto tg_blur = cv::imread("img/tg01_blur.jpg", cv::IMREAD_COLOR);
    int cell = 20;
    int rows = tg.rows / cell;
    int cols = tg.cols / cell;
    double val;

    cv::Mat roi, roi_gray;
    cv::Mat mtf_c(rows, cols, CV_8UC1);
    cv::Mat mtf_r(rows, cols, CV_8UC1);
    //*
    for (int i = 0; i != rows; ++i) {
        for (int j = 0; j != cols; ++j) {
            roi = getROI(tg_blur, i * cell, j * cell, cell);
            cv::cvtColor(roi, roi_gray, cv::COLOR_RGB2GRAY);
            auto ret = getBrightestDarkest(roi_gray);
            cv::Point mx = ret.first;
            cv::Point mn = ret.second;
            std::vector<cv::Point> linePixels;
            std::vector<uint8_t> vec;
            cv::line(roi, mx, mn, cv::Scalar(255, 255, 0));
            //cv::LineIterator li(roi, mx, mn);
            for (cv::LineIterator li(roi, mx, mn); li.pos() != mn; ++li) {
                linePixels.push_back(li.pos());
                vec.push_back(roi_gray.at<uint8_t>(li.pos()));
            }
            val = calculateMTFbyContrast(vec);
            mtf_c.at<uint8_t>(i, j) = static_cast<uint8_t >(val * 255);
            val = calculateMTFbyRange(vec);
            mtf_r.at<uint8_t>(i, j) = static_cast<uint8_t >(val * 255);
        }
    }
    //*/
    cv::imshow("tg", tg);
    cv::imshow("tg_blur", tg_blur);
    cv::imshow("roi", roi);
    cv::Mat mtf_cr, mtf_rr;
    cv::resize(mtf_c, mtf_cr, tg.size());
    cv::resize(mtf_r, mtf_rr, tg.size());
    cv::imshow("mtf_c", mtf_cr);
    cv::imshow("mtf_r", mtf_rr);
    cv::waitKey(0);
    return 0;
}
