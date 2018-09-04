#include <iostream>
#include <iomanip>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <mtf.h>

int main() {
    // set this to true to see how images are processed
    bool showDebugInfo = false;
    std::string ext = ".jpg";
    std::string target = "tg01";
    cv::Mat tg = cv::imread("img/" + target + "_blur" + ext, cv::IMREAD_COLOR);
    auto mtf = processTarget(tg, 20, showDebugInfo);
    cv::imshow("mtf", mtf);
    cv::waitKey(0);
    return 0;
}
