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

    // Note: make sure that working directory is set to mtf/
    cv::Mat tg = cv::imread("img/" + target + "_blur" + ext, cv::IMREAD_COLOR);
    auto mtf = processTarget(tg, 20, showDebugInfo);
    auto dft = applyDFT(mtf);

    cv::imshow("mtf", mtf);
    cv::imshow("dft", dft);
    cv::waitKey(0);
    return 0;
}
