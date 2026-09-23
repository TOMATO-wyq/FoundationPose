#include <sl/Camera.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;

struct Options {
    fs::path output_root = "captures";
    std::string resolution = "HD1080";
    int fps = 30;
    int camera_id = 0;
    int count = 0;
    double interval_seconds = 1.0;
    bool no_preview = false;
};

static void print_help(const char* program) {
    std::cout
        << "ZED 双目拍照工具\n\n"
        << "用法: " << program << " [选项]\n\n"
        << "默认打开实时预览：空格/s/回车拍照，q/ESC 退出。\n\n"
        << "选项:\n"
        << "  --output DIR       保存根目录（默认 captures）\n"
        << "  --resolution MODE  AUTO/HD2K/HD1200/HD1080/HD720/VGA（默认 HD1080）\n"
        << "  --fps N            帧率（默认 30）\n"
        << "  --camera-id N      相机编号（默认 0）\n"
        << "  --no-preview       不显示窗口，自动拍照\n"
        << "  --count N          自动拍 N 组；使用 --no-preview 时必须大于 0\n"
        << "  --interval SEC     自动拍照间隔秒数（默认 1.0）\n"
        << "  -h, --help         显示帮助\n\n"
        << "每组保存 RGB、右目、双目图、浮点/16 位深度、深度预览和彩色点云。\n";
}

static std::string require_value(int& i, int argc, char** argv, const std::string& option) {
    if (++i >= argc) {
        throw std::runtime_error(option + " 缺少参数");
    }
    return argv[i];
}

static Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--output") {
            options.output_root = require_value(i, argc, argv, arg);
        } else if (arg == "--resolution") {
            options.resolution = require_value(i, argc, argv, arg);
        } else if (arg == "--fps") {
            options.fps = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--camera-id") {
            options.camera_id = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--count") {
            options.count = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--interval") {
            options.interval_seconds = std::stod(require_value(i, argc, argv, arg));
        } else if (arg == "--no-preview") {
            options.no_preview = true;
        } else {
            throw std::runtime_error("未知选项: " + arg);
        }
    }
    if (options.fps <= 0 || options.count < 0 || options.interval_seconds < 0.0) {
        throw std::runtime_error("fps 必须大于 0；count 和 interval 不能小于 0");
    }
    if (options.no_preview && options.count == 0) {
        throw std::runtime_error("--no-preview 模式需要同时指定 --count N");
    }
    return options;
}

static sl::RESOLUTION parse_resolution(const std::string& value) {
    if (value == "AUTO") return sl::RESOLUTION::AUTO;
    if (value == "HD2K") return sl::RESOLUTION::HD2K;
    if (value == "HD1200") return sl::RESOLUTION::HD1200;
    if (value == "HD1080") return sl::RESOLUTION::HD1080;
    if (value == "HD720") return sl::RESOLUTION::HD720;
    if (value == "VGA") return sl::RESOLUTION::VGA;
    throw std::runtime_error("不支持的分辨率: " + value);
}

static std::string local_time_string(const char* format) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&value, &local);
    std::ostringstream stream;
    stream << std::put_time(&local, format);
    return stream.str();
}

static cv::Mat as_cv_mat(sl::Mat& image) {
    return cv::Mat(
        static_cast<int>(image.getHeight()),
        static_cast<int>(image.getWidth()),
        CV_8UC4,
        image.getPtr<sl::uchar1>(sl::MEM::CPU),
        image.getStepBytes(sl::MEM::CPU));
}

static bool save_npy_f32(const fs::path& path, const sl::Mat& depth) {
    if (depth.getDataType() != sl::MAT_TYPE::F32_C1) return false;

    std::ofstream output(path, std::ios::binary);
    if (!output) return false;

    const std::string shape = "(" + std::to_string(depth.getHeight()) + ", " +
                              std::to_string(depth.getWidth()) + ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
    // NPY v1.0 要求 magic/version/header-len/header 的总长度为 16 的整数倍。
    const std::size_t preamble_size = 10;
    const std::size_t padding = (16 - ((preamble_size + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');

    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    output.write(magic, sizeof(magic));
    const char version[] = {1, 0};
    output.write(version, sizeof(version));
    const std::uint16_t header_size = static_cast<std::uint16_t>(header.size());
    const char length[] = {static_cast<char>(header_size & 0xff),
                           static_cast<char>((header_size >> 8) & 0xff)};
    output.write(length, sizeof(length));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));

    const auto* base = depth.getPtr<sl::float1>(sl::MEM::CPU);
    const std::size_t row_bytes = static_cast<std::size_t>(depth.getWidth()) * sizeof(float);
    const std::size_t step_bytes = depth.getStepBytes(sl::MEM::CPU);
    for (std::size_t row = 0; row < depth.getHeight(); ++row) {
        output.write(reinterpret_cast<const char*>(base) + row * step_bytes,
                     static_cast<std::streamsize>(row_bytes));
    }
    return output.good();
}

static void write_camera_files(const fs::path& session_dir,
                               const sl::CameraInformation& info) {
    const auto& camera = info.camera_configuration.calibration_parameters.left_cam;
    std::ofstream matrix(session_dir / "cam_K.txt");
    matrix << std::setprecision(10)
           << camera.fx << " 0 " << camera.cx << '\n'
           << "0 " << camera.fy << ' ' << camera.cy << '\n'
           << "0 0 1\n";

    std::ofstream json(session_dir / "camera_info.json");
    json << std::setprecision(10)
         << "{\n"
         << "  \"camera_model\": \"" << info.camera_model << "\",\n"
         << "  \"serial_number\": " << info.serial_number << ",\n"
         << "  \"image_width\": " << camera.image_size.width << ",\n"
         << "  \"image_height\": " << camera.image_size.height << ",\n"
         << "  \"fx\": " << camera.fx << ",\n"
         << "  \"fy\": " << camera.fy << ",\n"
         << "  \"cx\": " << camera.cx << ",\n"
         << "  \"cy\": " << camera.cy << ",\n"
         << "  \"distortion\": [";
    for (int i = 0; i < 12; ++i) {
        if (i) json << ", ";
        json << camera.disto[i];
    }
    json << "],\n"
         << "  \"image_rectified\": true,\n"
         << "  \"depth_aligned_to\": \"rectified left/RGB image\",\n"
         << "  \"depth_unit_npy\": \"millimeter\",\n"
         << "  \"depth_unit_png\": \"millimeter (uint16; 0 means invalid)\",\n"
         << "  \"coordinate_system\": \"IMAGE (X right, Y down, Z forward)\"\n"
         << "}\n";
}

static bool save_capture(sl::Camera& camera,
                         const fs::path& session_dir,
                         std::ofstream& metadata,
                         int index,
                         unsigned int serial_number) {
    sl::Mat left;
    sl::Mat right;
    sl::Mat stereo;
    sl::Mat depth;
    sl::Mat depth_u16;
    sl::Mat depth_visual;
    sl::Mat point_cloud;

    auto state = camera.retrieveImage(left, sl::VIEW::LEFT, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取左目图像失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveImage(right, sl::VIEW::RIGHT, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取右目图像失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveImage(stereo, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取双目并排图像失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取浮点深度失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveMeasure(depth_u16, sl::MEASURE::DEPTH_U16_MM, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取 16 位深度失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveImage(depth_visual, sl::VIEW::DEPTH, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取深度可视化失败: " << state << '\n';
        return false;
    }
    state = camera.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU);
    if (state > sl::ERROR_CODE::SUCCESS) {
        std::cerr << "读取点云失败: " << state << '\n';
        return false;
    }

    std::ostringstream base;
    base << local_time_string("%Y%m%d_%H%M%S") << '_' << std::setw(4) << std::setfill('0') << index;
    const std::string name = base.str();
    // rgb/ 与 depth/ 使用完全相同的主文件名，兼容 FoundationPose 的数据读取器。
    const fs::path left_path = session_dir / "rgb" / (name + ".png");
    const fs::path right_path = session_dir / "right" / (name + ".png");
    const fs::path stereo_path = session_dir / "stereo" / (name + ".png");
    const fs::path depth_npy_path = session_dir / "depth_npy" / (name + ".npy");
    const fs::path depth_png_path = session_dir / "depth" / (name + ".png");
    const fs::path depth_visual_path = session_dir / "depth_visual" / (name + ".png");
    const fs::path point_cloud_path = session_dir / "point_cloud" / (name + ".ply");

    const auto left_result = left.write(left_path.string().c_str());
    const auto right_result = right.write(right_path.string().c_str());
    const auto stereo_result = stereo.write(stereo_path.string().c_str());
    const bool npy_result = save_npy_f32(depth_npy_path, depth);
    const auto depth_png_result = depth_u16.write(depth_png_path.string().c_str());
    const auto depth_visual_result = depth_visual.write(depth_visual_path.string().c_str());
    const auto point_cloud_result = point_cloud.write(point_cloud_path.string().c_str());
    if (left_result != sl::ERROR_CODE::SUCCESS || right_result != sl::ERROR_CODE::SUCCESS ||
        stereo_result != sl::ERROR_CODE::SUCCESS || !npy_result ||
        depth_png_result != sl::ERROR_CODE::SUCCESS ||
        depth_visual_result != sl::ERROR_CODE::SUCCESS ||
        point_cloud_result != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "写入文件失败: rgb=" << left_result << ", right=" << right_result
                  << ", stereo=" << stereo_result << ", npy=" << npy_result
                  << ", depth_png=" << depth_png_result
                  << ", depth_visual=" << depth_visual_result
                  << ", point_cloud=" << point_cloud_result << '\n';
        return false;
    }

    const auto timestamp = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).data_ns;
    metadata << index << ',' << local_time_string("%Y-%m-%d %H:%M:%S") << ',' << timestamp << ','
             << serial_number << ',' << left.getWidth() << ',' << left.getHeight() << ','
             << left_path.filename().string() << ',' << right_path.filename().string() << ','
             << stereo_path.filename().string() << ',' << depth_npy_path.filename().string() << ','
             << depth_png_path.filename().string() << ',' << depth_visual_path.filename().string() << ','
             << point_cloud_path.filename().string() << '\n';
    metadata.flush();

    std::cout << "已保存第 " << index << " 组照片: " << fs::absolute(session_dir) << '\n';
    return true;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const fs::path session_dir = fs::absolute(options.output_root) / local_time_string("%Y%m%d_%H%M%S");
        fs::create_directories(session_dir / "rgb");
        fs::create_directories(session_dir / "right");
        fs::create_directories(session_dir / "stereo");
        fs::create_directories(session_dir / "depth");
        fs::create_directories(session_dir / "depth_npy");
        fs::create_directories(session_dir / "depth_visual");
        fs::create_directories(session_dir / "point_cloud");

        sl::InitParameters init;
        init.camera_resolution = parse_resolution(options.resolution);
        init.camera_fps = options.fps;
        init.depth_mode = sl::DEPTH_MODE::NEURAL;
        init.coordinate_units = sl::UNIT::MILLIMETER;
        init.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
        init.input.setFromCameraID(options.camera_id);

        sl::Camera camera;
        std::cout << "正在打开 ZED 相机 " << options.camera_id << " ...\n";
        const auto open_state = camera.open(init);
        if (open_state > sl::ERROR_CODE::SUCCESS) {
            std::cerr << "打开相机失败: " << open_state
                      << "\n可先运行 ZED_Explorer --all 检查设备，或确认没有其他程序占用相机。\n";
            fs::remove_all(session_dir);
            return 2;
        }

        const auto info = camera.getCameraInformation();
        const unsigned int serial_number = info.serial_number;
        std::cout << "已连接: " << info.camera_model << "，S/N " << serial_number << "，"
                  << info.camera_configuration.resolution.width << 'x'
                  << info.camera_configuration.resolution.height << " @ "
                  << info.camera_configuration.fps << " FPS\n";
        std::cout << "本次保存目录: " << session_dir << '\n';
        write_camera_files(session_dir, info);

        std::ofstream metadata(session_dir / "capture_info.csv");
        metadata << "index,local_time,camera_timestamp_ns,serial,width,height,rgb_file,right_file,stereo_file,"
                    "depth_npy_file,depth_png_file,depth_visual_file,point_cloud_file\n";

        // 丢弃启动初期的若干帧，让自动曝光先稳定。
        for (int i = 0; i < 15; ++i) {
            camera.grab();
        }

        int saved = 0;
        sl::Mat preview_image;
        auto next_capture = std::chrono::steady_clock::now();
        if (!options.no_preview) {
            cv::namedWindow("ZED Photo Capture", cv::WINDOW_NORMAL);
            std::cout << "按 空格 / s / 回车 拍照，按 q / ESC 退出。\n";
        }

        while (options.count == 0 || saved < options.count) {
            const auto grab_state = camera.grab();
            if (grab_state > sl::ERROR_CODE::SUCCESS) {
                std::cerr << "抓取帧失败: " << grab_state << '\n';
                break;
            }

            bool should_capture = false;
            if (options.no_preview) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_capture) {
                    should_capture = true;
                    next_capture = now + std::chrono::milliseconds(
                        static_cast<long long>(options.interval_seconds * 1000.0));
                }
            } else {
                camera.retrieveImage(preview_image, sl::VIEW::SIDE_BY_SIDE, sl::MEM::CPU);
                cv::Mat display;
                cv::resize(as_cv_mat(preview_image), display, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                cv::putText(display, "SPACE / S / ENTER: capture    Q / ESC: quit", {20, 35},
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0, 255}, 2, cv::LINE_AA);
                cv::imshow("ZED Photo Capture", display);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 27) break;
                should_capture = key == ' ' || key == 's' || key == 13 || key == 10;
            }

            if (should_capture && save_capture(camera, session_dir, metadata, saved + 1, serial_number)) {
                ++saved;
            }
        }

        if (!options.no_preview) cv::destroyAllWindows();
        camera.close();
        std::cout << "拍照结束，共保存 " << saved << " 组。目录: " << session_dir << '\n';
        return saved > 0 || options.count == 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
