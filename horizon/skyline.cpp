#include <iterator>
#include <numeric>
#include <chrono>
#include <algorithm>
#include "opencv2/opencv.hpp"

typedef struct {
    float r = 0;
    float t = 0;
    int counter = 0;
} LineInfo_t;

typedef std::vector<cv::Point> Contour_t;

typedef struct {
    Contour_t points;
    double area;
} ContourArea_t;

std::vector<ContourArea_t> get_contours(const cv::Mat& img, double thresh)
{
    cv::Mat thr_frame;
    //cv::blur(img, img, cv::Size(3, 3));
    cv::threshold(img, thr_frame, thresh, 255, cv::THRESH_BINARY);

    //cv::dilate(img, img, cv::Mat());
    //cv::erode(img, img, cv::Mat());

    std::vector<Contour_t> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
        thr_frame,
        contours,
        hierarchy,
        cv::RETR_EXTERNAL,
        //cv::CHAIN_APPROX_TC89_KCOS
        //cv::CHAIN_APPROX_TC89_L1
        cv::CHAIN_APPROX_SIMPLE
    );

    if (contours.empty())
        return std::vector<ContourArea_t>{};

    // caclculate areas
    std::vector<ContourArea_t> areas;
    std::transform(contours.begin(), contours.end(), std::back_inserter(areas),
        [](const auto& contour) { return ContourArea_t{ contour, cv::contourArea(contour) }; });

    std::sort(areas.begin(), areas.end(), [](auto a, auto b) { return a.area > b.area; });

    return areas;
}

Contour_t flatten_contour(Contour_t contour)
{
    const int size = contour.size();
    if (size < 4)
        return contour;

    // find left and right points
    int left = 0, right = 0;
    for (auto i = 0; i < size; ++i) {
        auto x = contour[i].x;
        auto y = contour[i].y;
        if (x == contour[left].x && y > contour[left].y)
            left = i;
        if (x < contour[left].x)
            left = i;
        if (x == contour[right].x && y > contour[right].y)
            right = i;
        if (x > contour[right].x)
            right = i;
    }

    // remove point above left and right
    Contour_t flat_contour;
    flat_contour.push_back(cv::Point{ contour[left].x, 0 });
    do {
        flat_contour.push_back(contour[left]);
        ++left;
        if (left < 0)
            left = size - 1;
        if (left == size)
            left = 0;
    } while (left <= right);
    flat_contour.push_back(cv::Point{ contour[left].x, 0 });

    return flat_contour;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "USAGE: skyline <path to video>" << std::endl;
        return 0;
    }

    cv::VideoCapture cap(argv[1]);

    if (!cap.isOpened()) {
        std::cout << "Cannot open video!\n";
        return -1;
    }

    std::cerr << "Press Escape to exit" << std::endl;
    std::cerr << "Press x to toggle frame delay (delay skipped by default)" << std::endl;

    const int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    auto fps = cap.get(CV_CAP_PROP_FPS);
    int frame_delay = static_cast<int>(1000 / fps);

    auto last_frame_moment = std::chrono::steady_clock::now();
    bool skip_delay = true;

    cv::Mat frame;
    while (cap.read(frame)) {

        //cv::pyrMeanShiftFiltering(frame, frame_gray, 10, 10, 1);
        cv::Mat frame_gray = frame;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // take upper 50% of frame to improve performance
        cv::Rect roi(0, 0, width, height / 2);
        frame_gray = cv::Mat1b{ frame_gray(roi) };
       
        // calculate mean color of image's top 10%-30%
        cv::Rect mean_rect(0, height / 10, width, height / 3);
        double mean = cv::mean(cv::Mat1b{ frame_gray(mean_rect) })[0];
        if (mean > 170.0f)
            mean *= 0.9f;
        if (mean > 200.0f)
            mean *= 0.8f;

        auto contours = get_contours(frame_gray, mean);
        cv::Mat zeros = cv::Mat::zeros(frame.size(), CV_8UC3);
        for (const auto& contour : contours) {
            auto rect = cv::boundingRect(contour.points);
            if (rect.x < width / 5) {
                auto points = flatten_contour(contour.points);

                cv::Mat skyline = cv::Mat::zeros(frame.size(), CV_8UC3);
                cv::drawContours(skyline, std::vector<Contour_t>{points}, 0, cv::Scalar(0,0,255), CV_FILLED);
                cv::drawContours(skyline, std::vector<Contour_t>{points}, 0, cv::Scalar(0, 0, 0), CV_FILLED, 8, cv::noArray(), INT_MAX, cv::Point{0, -2});

                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                cv::drawContours(mask, std::vector<Contour_t>{points}, 0, cv::Scalar(255), CV_FILLED);
                cv::drawContours(mask, std::vector<Contour_t>{points}, 0, cv::Scalar(0), CV_FILLED, 8, cv::noArray(), INT_MAX, cv::Point{ 0, -2 });

                cv::bitwise_and(frame, zeros, frame, mask);
                cv::bitwise_or(frame, skyline, frame, mask);

                break;
            }
        }

        cv::imshow("Horizon Detection", frame);

        auto current_time = std::chrono::steady_clock::now();
        int delay = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_moment).count());
        delay = std::max(frame_delay - delay, 1);
        last_frame_moment = current_time;

        char key_pressed = cv::waitKey(skip_delay ? 1 : delay) & 0xFF;
        if (key_pressed == 0x1B)
            break;
        if (key_pressed == 'x' || key_pressed == 'X') {
            skip_delay = !skip_delay;
            std::cerr << "Skip delay: " << std::boolalpha << skip_delay << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
