// Observed boundary lines and depth sampling. All geometric gates match Python.
#include "face_ops.hpp"
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Pixels = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using Depth = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Quad = std::array<cv::Point2d, 4>;
struct Line { cv::Point2d normal; double offset; };
struct Report {
    cv::Point2d first, last;
    std::size_t support;
    double span, rms;
};

Quad read_quad(const Array& array) {
    if (array.ndim() != 2 || array.shape(0) != 4 || array.shape(1) != 2)
        throw std::invalid_argument("expected quad[4,2]");
    Quad quad;
    const double* data = array.data();
    for (int i = 0; i < 4; ++i) {
        quad[i] = {data[2*i], data[2*i+1]};
        if (!std::isfinite(quad[i].x) || !std::isfinite(quad[i].y))
            throw std::invalid_argument("quad must be finite");
    }
    return quad;
}

py::array_t<double> quad_array(const Quad& quad) {
    py::array_t<double> output({4, 2});
    double* data = output.mutable_data();
    for (int i = 0; i < 4; ++i) {
        data[2*i] = quad[i].x;
        data[2*i+1] = quad[i].y;
    }
    return output;
}

cv::Rect bounds(const Quad& quad, int width, int height, double low_pad, double high_pad) {
    double min_x = quad[0].x, max_x = min_x, min_y = quad[0].y, max_y = min_y;
    for (const auto& p : quad) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
    }
    int x0 = std::clamp(static_cast<int>(std::floor(min_x-low_pad)), 0, width);
    int y0 = std::clamp(static_cast<int>(std::floor(min_y-low_pad)), 0, height);
    int x1 = std::clamp(static_cast<int>(std::ceil(max_x+high_pad)), 0, width);
    int y1 = std::clamp(static_cast<int>(std::ceil(max_y+high_pad)), 0, height);
    return {x0, y0, std::max(0, x1-x0), std::max(0, y1-y0)};
}

std::vector<cv::Point2d> edge_points(const Pixels& image, const cv::Rect& roi) {
    std::vector<cv::Point2d> points;
    const auto width = image.shape(1);
    const auto* pixels = image.data();
    for (int y = roi.y; y < roi.y+roi.height; ++y)
        for (int x = roi.x; x < roi.x+roi.width; ++x)
            if (pixels[y*width+x]) points.emplace_back(x, y);
    return points;
}

bool ordered(Quad& quad) {
    cv::Point2d center(0., 0.);
    for (int i = 0; i < 4; ++i) {
        const auto& p = quad[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
        for (int j = 0; j < i; ++j) if (p == quad[j]) return false;
        center += p;
    }
    center *= .25;
    std::sort(quad.begin(), quad.end(), [&](const auto& a, const auto& b) {
        return std::atan2(a.y-center.y, a.x-center.x) < std::atan2(b.y-center.y, b.x-center.x);
    });
    const auto first = std::min_element(quad.begin(), quad.end(), [](const auto& a, const auto& b) {
        return a.x+a.y < b.x+b.y;
    });
    std::rotate(quad.begin(), first, quad.end());
    std::vector<cv::Point2f> contour;
    for (const auto& p : quad) contour.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    return cv::isContourConvex(contour) && cv::contourArea(contour) >= 1.;
}

bool intersections(const std::array<Line,4>& lines, const Quad& original,
                   double determinant_min, double shift_max, Quad& result) {
    for (int i = 0; i < 4; ++i) {
        const auto& previous = lines[(i+3)%4];
        const auto& current = lines[i];
        const auto& a = previous.normal;
        const auto& b = current.normal;
        const double det = a.x*b.y-a.y*b.x;
        if (std::abs(det) < determinant_min) return false;
        result[i] = {(-previous.offset*b.y+a.y*current.offset)/det,
                     (-a.x*current.offset+previous.offset*b.x)/det};
        if (cv::norm(result[i]-original[i]) > shift_max) return false;
    }
    return ordered(result);
}

cv::Vec4f robust_line(const std::vector<cv::Point2d>& points) {
    // Python explicitly converts the input to float32 before fitLine.
    std::vector<cv::Point2f> input;
    input.reserve(points.size());
    for (const auto& p : points) input.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    cv::Vec4f line;
    cv::fitLine(input, line, cv::DIST_HUBER, 0, .01, .01);
    return line;
}

py::array_t<double> refine_quad(const Array& array, const Pixels& image) {
    const Quad input = read_quad(array);
    if (image.ndim() != 2) throw std::invalid_argument("expected edge_image[H,W]");
    Quad output = input;
    {
        py::gil_scoped_release release;
        const auto points = edge_points(image, bounds(input, image.shape(1), image.shape(0), 5., 6.));
        if (points.size() >= 16) {
            std::array<Line,4> lines;
            bool regular = true;
            for (int side = 0; side < 4; ++side) {
                const auto& a = input[side];
                const auto edge = input[(side+1)%4]-a;
                const double length = cv::norm(edge);
                if (length == 0.) { regular = false; break; }
                const auto tangent = edge*(1./length);
                const cv::Point2d normal(-tangent.y, tangent.x);
                std::vector<cv::Point2d> selected;
                for (const auto& p : points) {
                    const auto relative = p-a;
                    const double along = relative.dot(tangent);
                    if (std::abs(relative.dot(normal)) < 2.5 && along > length*.12 && along < length*.88)
                        selected.push_back(p);
                }
                if (selected.size() < 8) lines[side] = {normal, -normal.dot(a)};
                else {
                    const auto line = robust_line(selected);
                    const cv::Point2d n(-static_cast<double>(line[1]), static_cast<double>(line[0]));
                    lines[side] = {n, -n.dot(cv::Point2d(line[2], line[3]))};
                }
            }
            Quad candidate;
            if (regular && intersections(lines, input, .05, 5., candidate)) output = candidate;
        }
    }
    return quad_array(output);
}

py::object fit_boundary_lines(const Array& array, const Pixels& image, double band,
                             double minimum_edge, int minimum_points, double minimum_span) {
    const Quad input = read_quad(array);
    if (image.ndim() != 2) throw std::invalid_argument("expected rgb_edges[H,W]");
    if (!std::isfinite(band) || band <= 0 || minimum_points < 1)
        throw std::invalid_argument("invalid observed boundary parameters");
    Quad output;
    std::array<Report,4> reports;
    bool success = true;
    {
        py::gil_scoped_release release;
        const auto points = edge_points(image, bounds(input, image.shape(1), image.shape(0), band+2., band+3.));
        std::array<Line,4> lines;
        for (int side = 0; side < 4 && success; ++side) {
            const auto& a = input[side];
            const auto edge = input[(side+1)%4]-a;
            const double length = cv::norm(edge);
            if (length < minimum_edge || length == 0.) { success = false; break; }
            const auto tangent = edge*(1./length);
            const cv::Point2d normal(-tangent.y, tangent.x);
            std::vector<cv::Point2d> selected;
            for (const auto& p : points) {
                const auto relative = p-a;
                const double along = relative.dot(tangent);
                if (std::abs(relative.dot(normal)) <= band && along > .08*length && along < .92*length)
                    selected.push_back(p);
            }
            if (selected.size() < static_cast<std::size_t>(minimum_points)) { success = false; break; }
            auto fitted = robust_line(selected);
            cv::Point2d direction(fitted[0], fitted[1]), origin(fitted[2], fitted[3]);
            if (std::abs(direction.dot(tangent)) < std::cos(12.*CV_PI/180.)) { success = false; break; }
            cv::Point2d n(-direction.y, direction.x);
            std::vector<cv::Point2d> trimmed;
            for (const auto& p : selected)
                if (std::abs((p-origin).dot(n)) <= std::min(band, 2.)) trimmed.push_back(p);
            if (trimmed.size() < static_cast<std::size_t>(minimum_points)) { success = false; break; }
            fitted = robust_line(trimmed);
            direction = {fitted[0], fitted[1]}; origin = {fitted[2], fitted[3]};
            n = {-direction.y, direction.x};
            double low = std::numeric_limits<double>::infinity(), high = -low, squares = 0.;
            for (const auto& p : trimmed) {
                const auto relative = p-origin;
                const double position = relative.dot(direction), error = relative.dot(n);
                low = std::min(low, position); high = std::max(high, position);
                squares += error*error;
            }
            const double span = (high-low)/length;
            if (span < minimum_span) { success = false; break; }
            lines[side] = {n, -n.dot(origin)};
            reports[side] = {origin+direction*low, origin+direction*high, trimmed.size(), span,
                             std::sqrt(squares/static_cast<double>(trimmed.size()))};
        }
        if (success) success = intersections(lines, input, .1, 2.*band, output);
    }
    if (!success) return py::none();
    py::list report_list;
    for (const auto& report : reports) {
        py::dict item;
        py::list first, last, segment;
        first.append(report.first.x); first.append(report.first.y);
        last.append(report.last.x); last.append(report.last.y);
        segment.append(first); segment.append(last);
        item["segment"] = segment;
        item["support_pixels"] = report.support;
        item["span_fraction"] = report.span;
        item["rms_px"] = report.rms;
        report_list.append(item);
    }
    return py::make_tuple(quad_array(output), report_list);
}

py::tuple sample_face_pixels(const Depth& depth, const Array& array, double minimum_depth,
                             double maximum_depth, int maximum_samples) {
    const Quad quad = read_quad(array);
    if (depth.ndim() != 2 || maximum_samples < 1)
        throw std::invalid_argument("expected depth[H,W] and positive maximum_samples");
    std::vector<cv::Point> valid;
    std::vector<float> sampled;
    std::size_t interior_count = 0, valid_count = 0;
    const auto width = depth.shape(1);
    const float* pixels = depth.data();
    // NumPy compares a float32 depth array to Python scalar bounds in float32.
    const float low_depth = static_cast<float>(minimum_depth), high_depth = static_cast<float>(maximum_depth);
    {
        py::gil_scoped_release release;
        const auto roi = bounds(quad, width, depth.shape(0), 0., 0.);
        // sample_face uses ceil(max)+1 rather than ceil(max+1), including integer max.
        const auto expanded = cv::Rect(roi.x, roi.y, std::min<int>(width-roi.x, roi.width+1),
                                      std::min<int>(depth.shape(0)-roi.y, roi.height+1));
        if (expanded.width < 4 || expanded.height < 4)
            throw std::invalid_argument("face outside image or too small");
        cv::Mat mask = cv::Mat::zeros(expanded.height, expanded.width, CV_8UC1);
        std::array<cv::Point,4> polygon;
        for (int i = 0; i < 4; ++i)
            polygon[i] = {static_cast<int>(std::nearbyint(quad[i].x-expanded.x)),
                          static_cast<int>(std::nearbyint(quad[i].y-expanded.y))};
        cv::fillConvexPoly(mask, polygon.data(), 4, cv::Scalar(255));
        cv::erode(mask, mask, cv::Mat::ones(5,5,CV_8U));
        for (int y = 0; y < expanded.height; ++y) {
            const auto* row = mask.ptr<std::uint8_t>(y);
            for (int x = 0; x < expanded.width; ++x) {
                if (!row[x]) continue;
                ++interior_count;
                const float z = pixels[(y+expanded.y)*width+x+expanded.x];
                if (std::isfinite(z) && z >= low_depth && z <= high_depth)
                    valid.emplace_back(x+expanded.x, y+expanded.y);
            }
        }
        valid_count = valid.size();
        if (valid.size() > static_cast<std::size_t>(maximum_samples)) {
            std::vector<cv::Point> reduced;
            reduced.reserve(maximum_samples);
            const double step = maximum_samples > 1 ? static_cast<double>(valid.size()-1)/(maximum_samples-1) : 0.;
            for (int i = 0; i < maximum_samples; ++i) {
                const auto index = i == maximum_samples-1 && maximum_samples > 1 ? valid.size()-1 :
                                   static_cast<std::size_t>(i*step);
                reduced.push_back(valid[index]);
            }
            valid = std::move(reduced);
        }
        sampled.reserve(valid.size());
        for (const auto& pixel : valid) sampled.push_back(pixels[pixel.y*width+pixel.x]);
    }
    py::array_t<std::int64_t> uv({static_cast<py::ssize_t>(valid.size()), py::ssize_t(2)});
    py::array_t<float> z(static_cast<py::ssize_t>(sampled.size()));
    auto* coordinates = uv.mutable_data();
    for (std::size_t i = 0; i < valid.size(); ++i) {
        coordinates[2*i] = valid[i].x; coordinates[2*i+1] = valid[i].y;
    }
    std::copy(sampled.begin(), sampled.end(), z.mutable_data());
    const double fraction = static_cast<double>(valid_count)/std::max<std::size_t>(1, interior_count);
    return py::make_tuple(uv, z, fraction, valid_count, interior_count);
}
} // namespace

void bind_face_ops(py::module_& module) {
    module.def("refine_quad", &refine_quad, py::arg("quad"), py::arg("edge_image"));
    module.def("fit_boundary_lines", &fit_boundary_lines, py::arg("quad"), py::arg("rgb_edges"),
               py::arg("band"), py::arg("min_edge_px"), py::arg("min_points"), py::arg("min_span"));
    module.def("sample_face_pixels", &sample_face_pixels, py::arg("depth"), py::arg("quad"),
               py::arg("min_depth"), py::arg("max_depth"), py::arg("max_samples"));
}
