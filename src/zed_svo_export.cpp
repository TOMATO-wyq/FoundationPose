#include <sl/Camera.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Options {
    fs::path svo;
    fs::path output;
    int start_frame = 0;
    int end_frame = -1;
    int stride = 1;
};

static void usage(const char* program) {
    std::cout
        << "将 ZED SVO/SVO2 转为 FoundationPose 可读取的 RGB-D 序列。\n\n"
        << "用法: " << program << " --svo FILE --output DIR [选项]\n\n"
        << "  --start-frame N  从第 N 帧开始（默认 0）\n"
        << "  --end-frame N    在第 N 帧结束，包含该帧（默认到结尾）\n"
        << "  --stride N       每 N 帧导出一帧（默认 1）\n"
        << "  -h, --help       显示帮助\n";
}

static std::string value_after(int& i, int argc, char** argv, const std::string& option) {
    if (++i >= argc) throw std::runtime_error(option + " 缺少参数");
    return argv[i];
}

static Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--svo") {
            options.svo = value_after(i, argc, argv, arg);
        } else if (arg == "--output") {
            options.output = value_after(i, argc, argv, arg);
        } else if (arg == "--start-frame") {
            options.start_frame = std::stoi(value_after(i, argc, argv, arg));
        } else if (arg == "--end-frame") {
            options.end_frame = std::stoi(value_after(i, argc, argv, arg));
        } else if (arg == "--stride") {
            options.stride = std::stoi(value_after(i, argc, argv, arg));
        } else {
            throw std::runtime_error("未知参数: " + arg);
        }
    }
    if (options.svo.empty() || options.output.empty()) {
        throw std::runtime_error("必须提供 --svo 和 --output");
    }
    if (!fs::is_regular_file(options.svo)) {
        throw std::runtime_error("找不到 SVO 文件: " + options.svo.string());
    }
    if (fs::file_size(options.svo) == 0) {
        throw std::runtime_error("SVO 文件大小为 0 字节；请重新录制，并在停止录制后正常关闭相机以完成文件写入: " +
                                 options.svo.string());
    }
    if (options.start_frame < 0 || options.stride < 1 ||
        (options.end_frame >= 0 && options.end_frame < options.start_frame)) {
        throw std::runtime_error("帧范围或 stride 无效");
    }
    return options;
}

static void write_intrinsics(const fs::path& path, const sl::CameraInformation& info) {
    const auto& camera = info.camera_configuration.calibration_parameters.left_cam;
    std::ofstream output(path);
    if (!output) throw std::runtime_error("无法写入内参: " + path.string());
    output << std::setprecision(10)
           << camera.fx << " 0 " << camera.cx << '\n'
           << "0 " << camera.fy << ' ' << camera.cy << '\n'
           << "0 0 1\n";
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const fs::path output = fs::absolute(options.output);
        fs::create_directories(output / "rgb");
        fs::create_directories(output / "depth");

        sl::InitParameters init;
        init.input.setFromSVOFile(fs::absolute(options.svo).string().c_str());
        init.svo_real_time_mode = false;
        init.depth_mode = sl::DEPTH_MODE::NEURAL;
        init.coordinate_units = sl::UNIT::MILLIMETER;
        init.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;

        sl::Camera camera;
        std::cout << "正在打开 " << fs::absolute(options.svo) << " ...\n";
        const auto open_state = camera.open(init);
        if (open_state != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "打开 SVO 失败: " << open_state << '\n';
            return 2;
        }

        const int total_frames = camera.getSVONumberOfFrames();
        const int last_frame = options.end_frame < 0
                                   ? total_frames - 1
                                   : std::min(options.end_frame, total_frames - 1);
        if (options.start_frame >= total_frames) {
            throw std::runtime_error("start-frame 超出录像总帧数 " + std::to_string(total_frames));
        }
        write_intrinsics(output / "cam_K.txt", camera.getCameraInformation());

        std::ofstream metadata(output / "frames.csv");
        metadata << "output_index,svo_frame,timestamp_ns,rgb_file,depth_file\n";

        camera.setSVOPosition(options.start_frame);
        sl::RuntimeParameters runtime;
        sl::Mat rgb;
        sl::Mat depth;
        int source_frame = options.start_frame;
        int output_index = 0;

        while (source_frame <= last_frame) {
            const auto grab_state = camera.grab(runtime);
            if (grab_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) break;
            if (grab_state != sl::ERROR_CODE::SUCCESS) {
                std::cerr << "跳过无法解码的 SVO 帧 " << source_frame << ": " << grab_state << '\n';
                ++source_frame;
                continue;
            }

            if ((source_frame - options.start_frame) % options.stride == 0) {
                const auto rgb_state = camera.retrieveImage(rgb, sl::VIEW::LEFT, sl::MEM::CPU);
                const auto depth_state = camera.retrieveMeasure(
                    depth, sl::MEASURE::DEPTH_U16_MM, sl::MEM::CPU);
                if (rgb_state != sl::ERROR_CODE::SUCCESS || depth_state != sl::ERROR_CODE::SUCCESS) {
                    throw std::runtime_error("读取 RGB 或深度失败，SVO 帧 " +
                                             std::to_string(source_frame));
                }

                std::ostringstream name;
                name << std::setw(6) << std::setfill('0') << output_index;
                const std::string rgb_name = name.str() + ".png";
                const std::string depth_name = name.str() + ".png";
                const fs::path rgb_path = output / "rgb" / rgb_name;
                const fs::path depth_path = output / "depth" / depth_name;
                if (rgb.write(rgb_path.string().c_str()) != sl::ERROR_CODE::SUCCESS ||
                    depth.write(depth_path.string().c_str()) != sl::ERROR_CODE::SUCCESS) {
                    throw std::runtime_error("写入 RGB-D 帧失败: " + name.str());
                }
                metadata << output_index << ',' << source_frame << ','
                         << camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).data_ns << ','
                         << rgb_name << ',' << depth_name << '\n';
                if (output_index % 30 == 0) {
                    std::cout << "已导出 " << output_index + 1 << " 帧\r" << std::flush;
                }
                ++output_index;
            }
            ++source_frame;
        }

        camera.close();
        std::cout << "\n完成：" << output_index << " 帧，输出目录 " << output << '\n';
        return output_index > 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
