#include <iostream>
#include <iomanip>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <mtf.h>


int halfCell = 100;

void callback(int event, int x, int y, int flags, void* data) {
    LineData * ptr = reinterpret_cast<LineData *>(data);
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::cout << "LMB at (" << x << ", " << y << ")" << std::endl;
    } else if (event == cv::EVENT_MBUTTONDOWN) {
        std::cout << "MMB at (" << x << ", " << y << ")" << std::endl;
    }
}

int main() {
    // read image
    std::string ext = ".jpg";
    std::string target = "img2";
    cv::Mat tg = cv::imread("img/" + target + ext, cv::IMREAD_COLOR);

    // create window and set callback
    cv::namedWindow("tg", 1);
    cv::setMouseCallback("tg", callback, nullptr);

    // calc noise
    cv::Mat noiseROI = tg(cv::Range(0, 30), cv::Range(90, 120));
    cv::imshow("noise", noiseROI);


    double noise = getNoise(noiseROI);
    std::cout << "noise=" << noise << std::endl;
    int x = 0;
    int y = 0;
    processLine(tg, 4, 4, 70, x - 20, y + 50, true, noise);
    processLine(tg, 0, 4, 70, x, y, true, noise);
    cv::imshow("tg", tg);
    cv::waitKey();

    return 0;
}
