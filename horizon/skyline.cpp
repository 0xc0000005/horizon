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
    std::cerr << "Press x to toggle frame delay" << std::endl;

    const int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    auto fps = cap.get(CV_CAP_PROP_FPS);
    int frame_delay = static_cast<int>(1000 / fps);

    cv::Mat frame;
    std::list<LineInfo_t> buffer;
    LineInfo_t skyline;
    const int max_size = 20;

    auto last_frame_moment = std::chrono::steady_clock::now();
    bool skip_delay = false;

    while (cap.read(frame)) {

        //cv::pyrMeanShiftFiltering(frame, frame_gray, 10, 10, 1);
        cv::Mat frame_gray = frame;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        cv::dilate(frame_gray, frame_gray, cv::Mat());
        cv::blur(frame_gray, frame_gray, cv::Size(9, 9));
        cv::erode(frame_gray, frame_gray, cv::Mat());
        
        // calculate mean color of image's top 20% (and strip 40 pixels above containing timeline)
        cv::Rect mean_rect(0, 40, width, height / 5);
        cv::Scalar mean = cv::mean(cv::Mat1b{ frame_gray(mean_rect) });

        cv::threshold(frame_gray, frame_gray, mean[0] * 0.9f, 255, cv::THRESH_BINARY);
        cv::Mat edges;
        cv::Canny(frame_gray, edges, 100, 200, 3, true);

        std::vector<cv::Vec2f> lines;
        cv::HoughLines(edges, lines, 1, CV_PI / 180.0f, 150);

        // filter lines to be close to PI/2 and to be in above 10%-50% range of image
        lines.erase(std::remove_if(lines.begin(),
                                   lines.end(),
                                   [=](cv::Vec2f v) {
                float r = v[0];
                float t = v[1];
                return abs(CV_PI / 2 - t) > CV_PI / 32 || r < height / 5 || r > height / 2;
            }), lines.end());

        if (!lines.empty()) {

            // cluster lines
            std::vector<int> labels;
            float msx_distance = 30;
            cv::partition(lines, labels, [=](const cv::Vec2f& a, const cv::Vec2f b) { return abs(a[0] - b[0]) < msx_distance; });
            // map clustered lines by labels
            std::map<int, LineInfo_t> clusters;
            int index = 0;
            for (auto& label : labels) {
                auto it = clusters.find(label);
                auto elm = it == clusters.end() ? LineInfo_t() : it->second;
                elm.r += lines[index][0];
                elm.t += lines[index][1];
                ++elm.counter;
                ++index;
                clusters[label] = elm;
            }
            // map to vector and calculate mean values
            std::vector<LineInfo_t> lines_vec;
            std::transform(clusters.begin(),
                clusters.end(),
                std::back_inserter(lines_vec),
                [](auto v) {
                auto elm = v.second;
                elm.r /= elm.counter;
                elm.t /= elm.counter;
                return elm; });

            // init skyline with firts 20 most populated clusters (can be a bad idea but I don't have another)
            if (buffer.size() < max_size) {
                auto max_cluster = std::max_element(lines_vec.begin(),
                                                    lines_vec.end(),
                                                    [](auto a, auto b) { return a.counter > b.counter; });
                buffer.push_back(*max_cluster);
            }
            // find nearest cluster
            else {
                float min_distance = static_cast<float>(height);
                LineInfo_t* nearest = nullptr;
                for (auto& elm : lines_vec) {
                    float distance = abs(elm.r - skyline.r);
                    if (distance <= min_distance) {
                        min_distance = distance;
                        nearest = &elm;
                    }
                }
                buffer.push_back(*nearest);
                buffer.pop_front();
            }

            skyline = std::accumulate(buffer.begin(),
                                      buffer.end(),
                                      LineInfo_t(),
                                      [](auto a, auto b) { a.r += b.r; a.t += b.t; return a; });
            skyline.r /= buffer.size();
            skyline.t /= buffer.size();
        }

        cv::Point p1(0, static_cast<int>(skyline.r / sin(skyline.t)));
        cv::Point p2(width, static_cast<int>((skyline.r - width * cos(skyline.t)) / sin(skyline.t)));
        cv::line(frame, p1, p2, cv::Scalar{ 0,0,255,255 }, 1);

        //cv::Mat sum_frame;
        //cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
        //cv::addWeighted(frame, 0.75, edges, 0.25, 0.0, sum_frame);

        cv::imshow("Horizon Detection", frame);
        //cv::imshow("Horizon Detection", frame_gray);
        //cv::imshow("Horizon Detection", sum_frame);

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
