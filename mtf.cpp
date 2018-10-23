#include "mtf.h"

#include <algorithm>
#include <cstdint>
#include <complex>
#include <iostream>
#include <math.h>
#include <numeric>
#include <valarray>
#include <sstream>

#include <cmath>
#include <vector>
#include <assert.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;


inline double gauss(double sigma, double x) {
    double expVal = -1 * (pow(x, 2) / pow(2 * sigma, 2));
    double divider = sqrt(2 * M_PI * pow(sigma, 2));
    return (1 / divider) * exp(expVal);
}

double integrate(CArray &arr) {
    double sum = 0;
    for (int j = 0; j != arr.size() - 1; j++) {
        sum += 0.5 * arr[j].real() + arr[j + 1].real();
    }
    return sum;
}

inline std::vector<Complex> gaussKernel(int samples, double sigma) {
    std::vector<Complex> v;
    bool doubleCenter = false;
    if (samples % 2 == 0) {
        doubleCenter = true;
        samples--;
    }
    int steps = (samples - 1) / 2;
    double stepSize = (3 * sigma) / steps;
    for (int i = steps; i >= 1; i--) {
        v.push_back(gauss(sigma, i * stepSize * -1));
    }
    v.push_back(gauss(sigma, 0));
    if (doubleCenter) {
        v.push_back(gauss(sigma, 0));
    }
    for (int i = 1; i <= steps; i++) {
        v.push_back(gauss(sigma, i * stepSize));
    }
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << ' ' << *it;
    }
    std::cout << std::endl;
    //assert(v.size() == samples);
    return v;
}

CArray gaussSmoothen(CArray values, double sigma, int samples) {
    CArray out(values.size());
    auto kernel = gaussKernel(samples, sigma);
    int sampleSide = samples / 2;
    unsigned long ubound = values.size();
    for (unsigned long i = 0; i < ubound; i++) {
        Complex sample = 0;
        int sampleCtr = 0;
        for (long j = i - sampleSide; j <= i + sampleSide; j++) {
            if (j > 0 && j < ubound) {
                int sampleWeightIndex = sampleSide + (j - i);
                sample += kernel[sampleWeightIndex] * values[j];
                sampleCtr++;
            }
        }
        Complex smoothed = sample / (Complex(sampleCtr));
        out[i] = smoothed;
    }
    return out;
}

double getNoise(cv::Mat mat) {
    cv::Mat mat_gray;
    cv::cvtColor(mat, mat_gray, cv::COLOR_BGR2GRAY);
    cv::Scalar tempVal = cv::mean(mat_gray);
    return 1 - tempVal.val[0] / 255;
}

void fft(CArray &x) {
    const size_t N = x.size();
    if (N <= 1) return;
    // divide
    CArray even = x[std::slice(0, N / 2, 2)];
    CArray odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

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

void toFile(std::string fname, std::string title, CArray &arr) {
    std::string sep = ";";
    std::ofstream fout;
    fout.open(fname + ".txt");
    fout << title << sep << "Re" << sep << "Im" << std::endl;
    for (size_t i = 0; i != arr.size(); ++i) {
        fout << i << sep << arr[i].real() << sep << arr[i].imag() << std::endl;
    }
    fout.close();
}

void cutHeadTail(CArray &arr) {
    arr[arr.size() - 1] = arr[arr.size() - 2];
    arr[0] = arr[1];
}

void smooth(CArray &hist, int winSize) {
    int winMidSize = winSize / 2;

    for (int i = winMidSize; i < hist.size() - winMidSize; ++i) {
        Complex mean = 0;
        for (int j = i - winMidSize; j <= (i + winMidSize); ++j) {
            mean += hist[j];
        }
        hist[i] = mean / Complex(winSize);
    }
}


void ifft(CArray &x) {
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

CArray signalDerivative(CArray &arr) {
    CArray ret(arr.size());
    for (int j = 0; j != arr.size() - 1; ++j) {
        ret[j] = arr[j] - arr[j + 1];
    }
    ret[arr.size() - 2] = ret[arr.size() - 3];
    ret[arr.size() - 1] = ret[arr.size() - 3];
    return std::abs(ret);
}


void processLine(cv::Mat tg, int i, int j, int cell, int x, int y, bool showDebugInfo, double noise) {
    std::stringstream ss;
    ss << i << "x" << j;
    std::string s = ss.str();

    cv::Mat roi, roi_gray;
    //roi = getROI(tg, i * cell, j * cell, cell);
    roi = tg(cv::Range(i * cell + x, i * cell + x + cell), cv::Range(j * cell + y, j * cell + cell + y));

    cv::cvtColor(roi, roi_gray, cv::COLOR_RGB2GRAY);
    auto ret = getBrightestDarkest(roi_gray);
    cv::Point mx = ret.first;
    cv::Point mn = ret.second;

    std::vector<cv::Point> linePixels;
    std::vector<double> vec;
    if (showDebugInfo) {
        cv::line(roi, mx, mn, cv::Scalar(255 * ((i + j) % 2), 0, 255));
        cv::imshow("tg", tg);
        cv::imshow("roi", roi);
    }

    for (cv::LineIterator li(roi, mx, mn); li.pos() != mn; ++li) {
        linePixels.push_back(li.pos());
        vec.push_back(roi_gray.at<uint8_t>(li.pos()));
    }

    CArray arr(vec.size() * 2);
    CArray step(vec.size() * 2);
    auto th = vec[0] - vec[vec.size() - 1];
    for (size_t c = 0; c != vec.size(); ++c) {
        arr[2 * c] = std::complex<double>(vec[c] / 255);
        step[c] = (arr[2 * c].real() < th) ? 1.0 : 0.0;
    }
    for (size_t c = 1; c != arr.size() - 1; c += 2) {
        arr[c] = std::complex<double>(0.5 * (arr[c - 1] + arr[c + 1]));
    }

    toFile("arr", "arr", arr); // real

    CArray dif = signalDerivative(arr);

    smooth(dif, 5);
    smooth(dif, 6);
    smooth(dif, 7);
    smooth(dif, 8);
    double sq = integrate(dif);
    dif /= sq;
    double sq2 = integrate(dif);
    std::cout << "square=" << sq << " norm=" << sq2 << std::endl;
    toFile("dif", "dif", dif);
    fft(dif);
    CArray mtf(dif.size());
    for (int j = 0; j != dif.size(); ++j) {
        mtf[j] = std::abs(dif[j]);
    }
    int samples = 30;
    toFile("mtf", "mtf", mtf);
    smooth(mtf, 3);
    CArray ans(samples);
    for (int j = 0; j != samples; ++j) {
        ans[j] = mtf[j];
    }
    toFile("mtf" + s, s, ans);
    return;
/*
    toFile("step", "step", step); // perfect

    fft(arr);
    toFile("arrfft", "arrfft", arr);

    fft(step);
    toFile("stepfft", "stepfft", step);
    double k = noise;
    std::stringstream ss;
    ss << k;
    std::string code = ss.str();

    CArray step_conj(step.size()), step_sq(step.size());
    for (int j = 0; j != step.size(); ++j) {
        step_conj[j] = std::conj(step[j]);
        step_sq[j] = std::pow(step[j], 2) + std::complex<double>(k);
    }
    toFile("step_sq", "step_sq", step_sq);
    CArray psf(arr.size()), flt(arr.size());
    for (int c = 0; c != psf.size(); ++c) {
        flt[c] = step_conj[c] / step_sq[c];
        psf[c] = flt[c] * arr[c];
    }
    toFile("psf", "psf", psf);
    toFile("flt", "flt", flt);

    // mul by -1
    for (int j = 0; j != psf.size(); ++j) {
        psf[j] *= std::pow(-1, j);
    }

    //CArray mtf(psf.size());
    for (int j = 0; j != psf.size(); ++j) {
        mtf[j] = std::abs(psf[j]);
    }
    toFile("mtf" + code, "k=" + code, mtf);
    */
}

cv::Mat processTargetCheckers(cv::Mat tg, int cell, bool showDebugInfo = true) {
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
            double noise = 0;
            //processLine(tg, i, j, cell, showDebugInfo, noise);
        }
    }
    return tg;
}


cv::Mat applyDFT(cv::Mat img) {
    cv::Mat padded;
    //expand input image to optimal size
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    // on the border add zero values
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    // Add to the expanded another plane with zeros
    cv::merge(planes, 2, complexI);

    // this way the result may fit in the source matrix
    cv::dft(complexI, complexI);

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(img))^2 + Im(DFT(img))^2))
    // planes[0] = Re(DFT(img), planes[1] = Im(DFT(img))
    split(complexI, planes);
    // planes[0] = magnitude
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    // switch to logarithmic scale
    magI += cv::Scalar::all(1);
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, CV_MINMAX);

    return magI;
}


cv::Mat processTargetStar(cv::Mat tg, int step, bool showDebugInfo = true) {
    std::string ext = ".jpg";
    double val;
    auto center = cv::Point(250, 250);
    double alpha = M_2_PI / 360 * 25;
    auto shift = cv::Point(0, 3);
    for (int r = 30; r <= center.y - 10; r += step) {
        int x = r * sin(alpha);
        int y = r * cos(alpha);
        auto a = cv::Point(center.x - x, center.y + y);
        auto b = cv::Point(center.x + x, center.y + y);
        if (showDebugInfo) {
            /*
            cv::line(tg, a, b, cv::Scalar(0, 0, 255));
            cv::line(tg, a + shift, b + shift, cv::Scalar(0, 100, 255));
            cv::line(tg, a - shift, b + shift, cv::Scalar(0, 0, 255));
             */
            cv::rectangle(tg, a - shift, b + shift, cv::Scalar(0, 127, 255));

            auto roi = tg(
                    cv::Range(a.x, b.x),
                    cv::Range(a.y - shift.y, b.y + shift.y)
            );
            cv::Mat roi_gray;
            cv::cvtColor(roi, roi_gray, cv::COLOR_RGB2GRAY);
            auto ret = getBrightestDarkest(roi_gray);

            cv::Point mx = ret.first;
            cv::Point mn = ret.second;
            std::vector<cv::Point> linePixels;
            std::vector<uint8_t> vec;
            for (cv::LineIterator li(roi, mx, mn); li.pos() != mn; ++li) {
                linePixels.push_back(li.pos());
                vec.push_back(roi_gray.at<uint8_t>(li.pos()));
            }
            val = calculateMTFbyContrast(vec);
            std::cout << 1.0 / r << ";" << val << std::endl;
        }
    }

    if (showDebugInfo) {
        cv::imshow("tg", tg);
    }
    return tg;
}

