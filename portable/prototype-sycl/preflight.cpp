// THROWAWAY: compiler/runtime gate before building the detector experiment.
#include <sycl/sycl.hpp>
#include "kernels/gradient.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: sycl-preflight cpu|cuda|hip");
        const std::string wanted = argv[1];
        std::vector<sycl::device> matches;
        for (const auto& platform : sycl::platform::get_platforms()) {
            for (const auto& device : platform.get_devices()) {
                const auto name = device.get_info<sycl::info::device::name>();
                const auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::cout << "device=" << name << " platform=" << platform_name
                          << " device_usm=" << device.has(sycl::aspect::usm_device_allocations)
                          << " host_usm=" << device.has(sycl::aspect::usm_host_allocations) << '\n';
                const bool selected = (wanted == "cpu" && device.is_cpu())
                    || (wanted == "cuda" && name.find("RTX 4090") != std::string::npos)
                    || (wanted == "hip" && name.find("gfx1032") != std::string::npos);
                if (selected) matches.push_back(device);
            }
        }
        if (matches.size() != 1) throw std::runtime_error("required platform missing or ambiguous");
        auto device = matches.front();
        if (!device.has(sycl::aspect::usm_device_allocations))
            throw std::runtime_error("required device USM unsupported");
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (const auto& error : errors) std::rethrow_exception(error);
        });
        std::cout << "selected=" << device.get_info<sycl::info::device::name>() << '\n';
        constexpr unsigned width = 37, height = 23, stride = 43;
        std::vector<std::uint8_t> input(stride * height, 0xA5);
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[y * stride + x] = static_cast<std::uint8_t>((x * 37 + y * 71) % 256);
        auto* source = sycl::malloc_device<std::uint8_t>(input.size(), queue);
        auto* output = sycl::malloc_device<std::int16_t>(2 * width * height, queue);
        if (!source || !output) throw std::bad_alloc();
        const auto upload = queue.memcpy(source, input.data(), input.size());
        const auto dx = cctag::portable::kernels::kDxTaps;
        const auto dy = cctag::portable::kernels::kDyTaps;
        auto kernel = queue.submit([&](sycl::handler& handler) {
            handler.depends_on(upload);
            handler.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
                const auto y = static_cast<unsigned>(id[0]);
                const auto x = static_cast<unsigned>(id[1]);
                output[y * width + x] = cctag::portable::kernels::gradient_at(
                    source, stride, width, height, x, y, dx);
                output[width * height + y * width + x] = cctag::portable::kernels::gradient_at(
                    source, stride, width, height, x, y, dy);
            });
        });
        std::vector<std::int16_t> actual(2 * width * height);
        queue.submit([&](sycl::handler& handler) {
            handler.depends_on(kernel);
            handler.memcpy(actual.data(), output, actual.size() * sizeof(actual[0]));
        }).wait_and_throw();
        queue.wait_and_throw();
        sycl::free(source, queue);
        sycl::free(output, queue);
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
                for (unsigned component = 0; component < 2; ++component) {
                    const auto expected = cctag::portable::kernels::gradient_at(input.data(),
                        stride, width, height, x, y, component == 0 ? dx : dy);
                    if (actual[component * width * height + y * width + x] != expected)
                        throw std::runtime_error("shared gradient differs from host evaluation");
                }
            }
        }
        // Independent expected values cover nearest-even conversion, including ties.
        const std::array<float, 8> values{2.5f, 3.5f, -2.5f, -3.5f, 0.f, 32768.f, -32769.f, 2.6f};
        const std::array<std::int16_t, 8> expected{2, 4, -2, -4, 0, 32767, -32768, 3};
        auto* rounded = sycl::malloc_device<std::int16_t>(values.size(), queue);
        if (!rounded) throw std::bad_alloc();
        queue.parallel_for(sycl::range<1>(values.size()), [=](sycl::id<1> id) {
            rounded[id[0]] = cctag::portable::kernels::round_to_int16(values[id[0]]);
        }).wait_and_throw();
        std::array<std::int16_t, 8> rounding{};
        queue.memcpy(rounding.data(), rounded, sizeof(rounding)).wait_and_throw();
        sycl::free(rounded, queue);
        if (rounding != expected) throw std::runtime_error("nearest-even conversion mismatch");
        std::cout << "PASS: shared gradient 37x23, padded stride 43; independent rounding values\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
