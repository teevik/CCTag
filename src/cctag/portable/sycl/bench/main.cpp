/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Times one stage of the portable pipeline in isolation
#include <cctag/portable/sycl/Stages.hpp>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using cctag::portable::Level;
using cctag::portable::Mark;
using Json = nlohmann::ordered_json;

enum class Stage {
    pyramid,
    gradient,
    suppress
};

const std::map<std::string, Stage> stage_names = {
    {"pyramid", Stage::pyramid},
    {"gradient", Stage::gradient},
    {"suppress", Stage::suppress},
};

constexpr float canny_low = 0.01f * 256;
constexpr float canny_high = 0.04f * 256;
constexpr std::uint32_t level_count = 4;
constexpr std::size_t warmup = 3;

} // namespace

int main(int argc, char** argv) {
    std::string stage_name = "pyramid";
    std::string image_path;
    std::string level_arg = "0";
    std::size_t batch = 100;
    std::size_t samples = 20;
    std::string out_path;

    CLI::App app(
        "Times one stage of the portable SYCL pipeline. ACPP_VISIBILITY_MASK=cuda|omp picks the "
        "device.",
        "cctag-portable-bench"
    );
    app.add_option("--stage", stage_name)->check(CLI::IsMember(stage_names))->capture_default_str();
    app.add_option("--image", image_path, "Image to load as gray")->required();
    app.add_option(
           "--level",
           level_arg,
           "Level to run the stage on, or all. For pyramid, the level downsampled into the next"
    )
        ->check(CLI::IsMember({"all"}) | CLI::NonNegativeNumber)
        ->capture_default_str();
    app.add_option("--batch", batch, "Calls per timed batch")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--samples", samples, "Measured batches")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--out", out_path, "JSON Lines output");
    CLI11_PARSE(app, argc, argv);

    const Stage stage = stage_names.at(stage_name);

    // The pyramid's last level is not downsampled
    const std::uint32_t sources = stage == Stage::pyramid ? level_count - 1 : level_count;
    std::vector<std::uint32_t> selected;
    if (level_arg == "all") {
        for (std::uint32_t i = 0; i < sources; ++i) {
            selected.push_back(i);
        }
    } else {
        selected.push_back(std::stoul(level_arg));
    }
    if (selected.back() >= sources) {
        std::cerr << "--level must be all or below " << sources << "\n";
        return 1;
    }

    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "cannot read " << image_path << "\n";
        return 1;
    }

    sycl::queue queue{sycl::property::queue::in_order{}};
    const auto device = queue.get_device().get_info<sycl::info::device::name>();

    std::vector<Level> levels;
    std::uint32_t width = image.cols;
    std::uint32_t height = image.rows;
    for (std::uint32_t i = 0; i < level_count; ++i, width /= 2, height /= 2) {
        const auto size = std::size_t(width) * height;
        levels.push_back(
            {width,
             height,
             sycl::malloc_device<std::uint8_t>(size, queue),
             sycl::malloc_device<std::int16_t>(size, queue),
             sycl::malloc_device<std::int16_t>(size, queue),
             sycl::malloc_device<Mark>(size, queue)}
        );
    }
    auto* intermediate = sycl::malloc_device<float>(std::size_t(image.cols) * image.rows, queue);

    // The stages before the timed one, as run() does
    queue.memcpy(levels[0].source, image.data, std::size_t(image.cols) * image.rows);
    for (std::size_t i = 1; i < levels.size(); ++i) {
        cctag::portable::downsample(queue, levels[i - 1], levels[i]);
    }

    if (stage == Stage::suppress) {
        for (const auto& level : levels) {
            cctag::portable::gradient(queue, level, intermediate);
        }
    }

    const auto call = [&] {
        for (const auto i : selected) {
            switch (stage) {
                case Stage::pyramid:
                    cctag::portable::downsample(queue, levels[i], levels[i + 1]);
                    break;
                case Stage::gradient:
                    cctag::portable::gradient(queue, levels[i], intermediate);
                    break;
                case Stage::suppress:
                    cctag::portable::suppress(queue, levels[i], canny_low, canny_high);
                    break;
            }
        }
    };

    std::vector<double> ns;
    for (std::size_t sample = 0; sample < warmup + samples; ++sample) {
        queue.wait();
        const auto start = std::chrono::steady_clock::now();

        for (std::size_t k = 0; k < batch; ++k) {
            call();
        }
        queue.wait();

        const std::chrono::duration<double, std::nano> elapsed =
            std::chrono::steady_clock::now() - start;

        if (sample >= warmup) {
            ns.push_back(elapsed.count() / batch);
        }
    }

    auto sorted = ns;
    std::sort(sorted.begin(), sorted.end());
    const double median_ns = sorted[(sorted.size() + 1) / 2 - 1];
    const double min_ns = sorted.front();

    std::printf(
        "%s level %s: median %.2f us, min %.2f us per call, %zu x %zu calls on %s\n",
        stage_name.c_str(),
        level_arg.c_str(),
        median_ns * 1e-3,
        min_ns * 1e-3,
        samples,
        batch,
        device.c_str()
    );

    if (!out_path.empty()) {
        std::ofstream out(out_path);

        out << Json{
                   {"type", "run"},
                   {"stage", stage_name},
                   {"image", image_path},
                   {"width", image.cols},
                   {"height", image.rows},
                   {"levels", level_count},
                   {"level", level_arg},
                   {"device", device},
                   {"batch", batch},
                   {"samples", samples},
                   {"warmup", warmup}
               }.dump()
            << "\n";

        for (std::size_t i = 0; i < ns.size(); ++i) {
            out << Json{{"type", "sample"}, {"sample", i}, {"ns", ns[i]}}.dump() << "\n";
        }

        out
            << Json{{"type", "summary"}, {"median_ns", median_ns}, {"min_ns", min_ns}}.dump()
            << "\n";
    }
}
