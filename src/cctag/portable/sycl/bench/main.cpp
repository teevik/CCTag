/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Times one stage of the portable pipeline in isolation: runs the pipeline up to the stage once on
// a real image, then times batches of the stage alone on the in-order queue, copying back whatever
// the stage rewrites in place before every call. Prints one line per run and, with --out, writes
// one JSON line for the run, one per sample and one for the summary.
//
// A batch is timed by the wall clock from an idle queue to an idle queue, so a call's time includes
// its launch. The time of a batch of restores alone is subtracted to give the stage's time.
#include <cctag/portable/sycl/Stages.hpp>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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
using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;

/// The stages that can be timed, in pipeline order
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

/// The detector parameters the stages take, by default those of cctag/Params.hpp
struct Params {
    float canny_thr_low = 0.01f;
    float canny_thr_high = 0.04f;
};

/// Device memory, as a pointer and a size in bytes
struct Span {
    void* data;
    std::size_t bytes;
};

/// The device images of every level and the scratch buffers, as allocate() of Pipeline.cpp
struct Buffers {
    sycl::queue& queue;
    std::vector<Level> levels;
    /// Temporary image the size of level 0
    float* intermediate;

    Buffers(sycl::queue& queue, std::uint32_t width, std::uint32_t height, std::uint32_t count) :
        queue(queue) {
        intermediate = sycl::malloc_device<float>(std::size_t(width) * height, queue);
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::size_t size = std::size_t(width) * height;
            levels.push_back(
                {width,
                 height,
                 sycl::malloc_device<std::uint8_t>(size, queue),
                 sycl::malloc_device<std::int16_t>(size, queue),
                 sycl::malloc_device<std::int16_t>(size, queue),
                 sycl::malloc_device<Mark>(size, queue)}
            );
            width /= 2;
            height /= 2;
        }
    }

    ~Buffers() {
        queue.wait();
        for (const Level& level : levels) {
            sycl::free(level.source, queue);
            sycl::free(level.dx, queue);
            sycl::free(level.dy, queue);
            sycl::free(level.map, queue);
        }
        sycl::free(intermediate, queue);
    }
};

/// Enqueues a stage on level i, as run() does. For the pyramid, the downsample of level i into
/// level i + 1
void enqueue_stage(Stage stage, Buffers& buffers, const Params& params, std::size_t i) {
    sycl::queue& queue = buffers.queue;
    const Level& level = buffers.levels[i];
    switch (stage) {
        case Stage::pyramid:
            cctag::portable::downsample(queue, level, buffers.levels[i + 1]);
            break;
        case Stage::gradient:
            cctag::portable::gradient(queue, level, buffers.intermediate);
            break;
        case Stage::suppress:
            // cvRecodedCanny scales the thresholds by 256 to compare with the magnitude
            cctag::portable::suppress(
                queue,
                level,
                params.canny_thr_low * 256,
                params.canny_thr_high * 256
            );
            break;
    }
}

/// What a stage rewrites in place on a level, so a repeated call must start from a copy, or
/// nothing for stages whose outputs do not feed their inputs, which is every current stage
std::vector<Span> overwritten(Stage stage, const Level& level) {
    return {};
}

/// What a stage writes on a level, compared by --verify
std::vector<Span> outputs(Stage stage, const Level& level) {
    const std::size_t pixels = std::size_t(level.width) * level.height;
    switch (stage) {
        case Stage::gradient:
            return {
                {level.dx, pixels * sizeof(std::int16_t)},
                {level.dy, pixels * sizeof(std::int16_t)}
            };
        default:
            return {{level.map, pixels * sizeof(Mark)}};
    }
}

/// Copies device spans to the host
std::vector<std::vector<std::uint8_t>>
download(sycl::queue& queue, const std::vector<Span>& spans) {
    std::vector<std::vector<std::uint8_t>> copies;
    for (const Span& span : spans) {
        copies.emplace_back(span.bytes);
        queue.memcpy(copies.back().data(), span.data, span.bytes);
    }
    queue.wait();
    return copies;
}

/// The bytes that differ between two downloads of the same spans
std::size_t mismatches(
    const std::vector<std::vector<std::uint8_t>>& a,
    const std::vector<std::vector<std::uint8_t>>& b
) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            count += a[i][j] != b[i][j];
        }
    }
    return count;
}

/// The pixels of a downsampled level that are not the rounded 2x2 average of the level above
std::size_t pyramid_mismatches(sycl::queue& queue, const Level& src, const Level& dst) {
    std::vector<std::uint8_t> above(std::size_t(src.width) * src.height);
    std::vector<std::uint8_t> below(std::size_t(dst.width) * dst.height);
    queue.memcpy(above.data(), src.source, above.size());
    queue.memcpy(below.data(), dst.source, below.size());
    queue.wait();
    std::size_t count = 0;
    for (std::size_t y = 0; y < dst.height; ++y) {
        for (std::size_t x = 0; x < dst.width; ++x) {
            const std::uint8_t* corner = &above[2 * y * src.width + 2 * x];
            const std::uint32_t sum =
                corner[0] + corner[1] + corner[src.width] + corner[src.width + 1];
            count += below[y * dst.width + x] != (sum + 2) / 4;
        }
    }
    return count;
}

/// A float parameter rounded to 1e-6, so the JSON does not print every digit of the widened double
double rounded(float value) {
    return std::round(value * 1e6) / 1e6;
}

std::int64_t nanoseconds(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

/// The time of k calls of enqueue, from an idle queue until the queue is idle again
template <typename Enqueue>
Clock::duration time_batch(sycl::queue& queue, std::size_t k, Enqueue enqueue) {
    queue.wait();
    const Clock::time_point start = Clock::now();
    for (std::size_t i = 0; i < k; ++i) {
        enqueue();
    }
    queue.wait();
    return Clock::now() - start;
}

/// The nearest-rank percentile: the smallest value with at least `percent`% of values <= it
double percentile(std::vector<double> values, std::size_t percent) {
    std::sort(values.begin(), values.end());
    const std::size_t rank = (percent * values.size() + 99) / 100;
    return values[std::max<std::size_t>(rank, 1) - 1];
}

} // namespace

int main(int argc, char** argv) {
    std::string stage_name = "pyramid";
    std::string image_path;
    std::uint32_t level_count = 4;
    std::string level_arg = "0";
    std::size_t batch = 100;
    std::size_t samples = 20;
    std::size_t warmup = 3;
    std::string out_path;
    std::string label;
    bool verify = false;
    Params params;

    CLI::App app(
        "Times one stage of the portable SYCL pipeline in isolation, after running the stages "
        "before it once on the image. "
        "ACPP_VISIBILITY_MASK=cuda|omp "
        "picks the device.",
        "cctag-portable-bench"
    );
    app.add_option("--stage", stage_name, "The stage to time")
        ->check(CLI::IsMember(stage_names))
        ->capture_default_str();
    app.add_option("--image", image_path, "Image to load as gray")->required();
    app.add_option("--levels", level_count, "Pyramid levels, as numberOfMultiresLayers")
        ->check(CLI::Range(1u, 16u))
        ->capture_default_str();
    app.add_option(
           "--level",
           level_arg,
           "Level to run the stage on, or all for every level as a detection does. For pyramid, "
           "the level downsampled into the next"
    )
        ->capture_default_str();
    app.add_option("--batch", batch, "Calls per timed batch")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--samples", samples, "Measured batches")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app.add_option("--warmup", warmup, "Warm-up batches")->capture_default_str();
    app.add_option("--out", out_path, "JSON Lines output, written after the run");
    app.add_option("--label", label, "Copied into the run record");
    app.add_flag(
        "--verify",
        verify,
        "Check the outputs after timing: the pyramid against a host 2x2 average, other stages "
        "against one call straight after setup"
    );
    app.add_option("--canny-thr-low", params.canny_thr_low, "_cannyThrLow")->capture_default_str();
    app.add_option("--canny-thr-high", params.canny_thr_high, "_cannyThrHigh")
        ->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    const Stage stage = stage_names.at(stage_name);
    const bool pyramid = stage == Stage::pyramid;
    const bool suppress = stage == Stage::suppress;
    // The stages whose bytes per call and bandwidth are reported
    const bool counted = pyramid || suppress;
    // The pyramid's calls write the next level, so its last level is no source
    const std::uint32_t last = pyramid ? level_count - 1 : level_count;
    if (last == 0) {
        std::cerr << "pyramid needs --levels of at least 2\n";
        return 1;
    }
    std::vector<std::size_t> selected;
    if (level_arg == "all") {
        for (std::size_t i = 0; i < last; ++i) {
            selected.push_back(i);
        }
    } else {
        std::size_t i = 0;
        try {
            i = std::stoul(level_arg);
        } catch (const std::exception&) {
            i = last;
        }
        if (i >= last) {
            std::cerr << "--level must be all or below " << last << "\n";
            return 1;
        }
        selected.push_back(i);
    }

    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty() || !image.isContinuous()) {
        std::cerr << "cannot read " << image_path << "\n";
        return 1;
    }

    sycl::queue queue{sycl::property::queue::in_order{}};
    const std::string device = queue.get_device().get_info<sycl::info::device::name>();
    std::cerr << "SYCL device: " << device << std::endl;

    Buffers buffers(queue, image.cols, image.rows, level_count);
    const std::vector<Level>& levels = buffers.levels;

    // Setup: the stages before this one on every level, as run() does
    queue.memcpy(levels[0].source, image.data, std::size_t(image.cols) * image.rows);
    for (std::size_t i = 1; i < levels.size(); ++i) {
        cctag::portable::downsample(queue, levels[i - 1], levels[i]);
    }
    for (std::size_t i = 0; i < levels.size(); ++i) {
        for (Stage before = Stage::gradient; before < stage; before = Stage(int(before) + 1)) {
            enqueue_stage(before, buffers, params, i);
        }
    }
    queue.wait();

    // A device copy of everything the stage rewrites in place, to restore before every call
    std::vector<Span> targets;
    std::vector<Span> saved;
    for (const std::size_t i : selected) {
        for (const Span& target : overwritten(stage, levels[i])) {
            targets.push_back(target);
            saved.push_back({sycl::malloc_device<std::uint8_t>(target.bytes, queue), target.bytes});
            queue.memcpy(saved.back().data, target.data, target.bytes);
        }
    }
    const bool restores = !targets.empty();
    const auto restore = [&]() {
        for (std::size_t j = 0; j < targets.size(); ++j) {
            queue.memcpy(targets[j].data, saved[j].data, targets[j].bytes);
        }
    };
    const auto call = [&]() {
        restore();
        for (const std::size_t i : selected) {
            enqueue_stage(stage, buffers, params, i);
        }
    };

    // Other stages are checked against one call straight after setup
    std::vector<std::vector<std::vector<std::uint8_t>>> reference;
    if (verify && !pyramid) {
        call();
        for (const std::size_t i : selected) {
            reference.push_back(download(queue, outputs(stage, levels[i])));
        }
    }
    // The pyramid's destinations start blank, so --verify sees what the timed calls wrote
    std::size_t bytes = 0;
    if (pyramid) {
        for (const std::size_t i : selected) {
            const Level& dst = levels[i + 1];
            const std::size_t pixels = std::size_t(dst.width) * dst.height;
            queue.memset(dst.source, 0, pixels);
            // Every output pixel reads its 2x2 block and writes itself
            bytes += 5 * pixels;
        }
    }
    if (suppress) {
        for (const std::size_t i : selected) {
            const std::size_t pixels = std::size_t(levels[i].width) * levels[i].height;
            // Every output pixel reads its two int16 gradients and writes itself, the neighbours'
            // gradients counted as cache hits
            bytes += 5 * pixels;
        }
    }
    queue.wait();

    std::vector<Json> records;
    std::vector<double> stage_ns;
    std::vector<double> call_ns;
    std::vector<double> restore_ns;
    for (std::size_t sample = 0; sample < warmup + samples; ++sample) {
        const bool measured = sample >= warmup;
        // Restores first, so the stage's outputs are in place after the last sample
        const double alone =
            restores ? nanoseconds(time_batch(queue, batch, restore)) / double(batch) : 0;
        const double with = nanoseconds(time_batch(queue, batch, call)) / double(batch);
        records.push_back(
            {{"type", "sample"},
             {"phase", measured ? "measured" : "warmup"},
             {"sample", measured ? sample - warmup : sample},
             {"stage_ns", with - alone},
             {"call_ns", with},
             {"restore_ns", alone}}
        );
        if (measured) {
            stage_ns.push_back(with - alone);
            call_ns.push_back(with);
            restore_ns.push_back(alone);
        }
    }

    std::size_t wrong = 0;
    if (verify) {
        for (std::size_t j = 0; j < selected.size(); ++j) {
            const std::size_t i = selected[j];
            wrong += pyramid ? pyramid_mismatches(queue, levels[i], levels[i + 1])
                             : mismatches(reference[j], download(queue, outputs(stage, levels[i])));
        }
    }
    for (const Span& span : saved) {
        sycl::free(span.data, queue);
    }

    const double median_ns = percentile(stage_ns, 50);
    const double min_ns = *std::min_element(stage_ns.begin(), stage_ns.end());
    const double call_median_ns = percentile(call_ns, 50);
    const double restore_median_ns = percentile(restore_ns, 50);
    const double gb_per_s = bytes / median_ns;

    std::printf(
        "%s level %s: median %.2f us, min %.2f us per call",
        stage_name.c_str(),
        level_arg.c_str(),
        median_ns * 1e-3,
        min_ns * 1e-3
    );
    if (restores) {
        std::printf(
            " (with restore %.2f us, restore %.2f us)",
            call_median_ns * 1e-3,
            restore_median_ns * 1e-3
        );
    }
    if (counted) {
        std::printf(", %.2f MB, %.1f GB/s", bytes * 1e-6, gb_per_s);
    }
    std::printf(", %zu x %zu calls on %s\n", samples, batch, device.c_str());
    if (verify) {
        std::printf(
            "verify: %zu mismatching %s\n",
            wrong,
            pyramid ? "pixels against the host average" : "bytes against the first call"
        );
    }

    if (!out_path.empty()) {
        Json run{
            {"type", "run"},
            {"label", label},
            {"stage", stage_name},
            {"image", image_path},
            {"width", image.cols},
            {"height", image.rows},
            {"levels", level_count},
            {"level", level_arg},
            {"device", device},
            {"batch", batch},
            {"samples", samples},
            {"warmup", warmup},
            {"canny_thr_low", rounded(params.canny_thr_low)},
            {"canny_thr_high", rounded(params.canny_thr_high)}
        };
        Json summary{
            {"type", "summary"},
            {"median_ns", median_ns},
            {"min_ns", min_ns},
            {"call_median_ns", call_median_ns},
            {"restore_median_ns", restore_median_ns}
        };
        if (counted) {
            run["bytes_per_call"] = bytes;
            summary["gb_per_s"] = gb_per_s;
        }
        if (verify) {
            summary["mismatches"] = wrong;
        }
        std::ofstream out(out_path);
        out << run.dump() << "\n";
        for (const Json& record : records) {
            out << record.dump() << "\n";
        }
        out << summary.dump() << "\n";
        out.close();
        if (!out) {
            std::cerr << "cannot write " << out_path << "\n";
            return 1;
        }
    }
    return verify && wrong > 0 ? 2 : 0;
}
