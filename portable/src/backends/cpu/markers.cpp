/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"

#include <Eigen/LU>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace cctag::portable::cpu {
namespace {

using Image = kernels::Plane<const std::uint8_t>;
using Cut = IdentificationScratch::Cut;

// The public status values are defined by the legacy CCTag API
constexpr int id_reliable = 1;
constexpr int no_collected_cuts = -1;
constexpr int no_selected_cuts = -2;
constexpr int opti_has_diverged = -3;
constexpr int id_not_reliable = -4;
constexpr int degenerate = -5;

/// Samples a point whose four neighbours are inside the plane
float bilinear(Image image, float x, float y) {
    const int px = static_cast<int>(x), py = static_cast<int>(y);
    const auto* top = image.row(py) + px;
    const auto* bottom = image.row(py + 1) + px;
    const float fx = x - px, fy = y - py;
    const float w1 = (1.f - fx) * (1.f - fy), w2 = fx * (1.f - fy);
    const float w3 = (1.f - fx) * fy, w4 = fx * fy;
    // Identification in the legacy pipeline uses half-intensity image cuts
    return (top[0] * w1 + top[1] * w2 + bottom[0] * w3 + bottom[1] * w4) / 2;
}

bool sample_line(
    Image image,
    Eigen::Vector2f start,
    Eigen::Vector2f stop,
    std::span<float> signal
) {
    const float step_x = (stop.x() - start.x()) / (signal.size() - 1.f);
    const float step_y = (stop.y() - start.y()) / (signal.size() - 1.f);
    float x = start.x(), y = start.y();
    for (float& value : signal) {
        if (!(x >= 1.f && x < image.width - 1.f && y >= 1.f && y < image.height - 1.f)) {
            return false;
        }
        value = bilinear(image, x, y);
        x += step_x;
        y += step_y;
    }
    return true;
}

Eigen::Vector2f project(const Eigen::Matrix3f& matrix, float x, float y) {
    const float denominator = matrix(2, 0) * x + matrix(2, 1) * y + matrix(2, 2);
    return {
        (matrix(0, 0) * x + matrix(0, 1) * y + matrix(0, 2)) / denominator,
        (matrix(1, 0) * x + matrix(1, 1) * y + matrix(1, 2)) / denominator
    };
}

/// Projects the outer point radially onto the fitted ellipse
Eigen::Vector2f point_on_ellipse(const Ellipse& ellipse, const DirectedPoint& point) {
    const float x = point.x - ellipse.cx, y = point.y - ellipse.cy;
    // Keep the legacy's double trigonometry and float intermediate coordinates
    float u = x * std::cos(double(ellipse.angle)) + y * std::sin(double(ellipse.angle));
    float v = -x * std::sin(double(ellipse.angle)) + y * std::cos(double(ellipse.angle));
    const float scale =
        std::sqrt(u * u / (ellipse.a * ellipse.a) + v * v / (ellipse.b * ellipse.b));
    u /= scale;
    v /= scale;
    return {
        u * std::cos(double(ellipse.angle)) - v * std::sin(double(ellipse.angle)) + ellipse.cx,
        u * std::sin(double(ellipse.angle)) + v * std::cos(double(ellipse.angle)) + ellipse.cy
    };
}

bool refine_outer_point(Cut& cut, Image image, float scale, std::span<float> signal) {
    const float length = 3.f * std::sqrt(2.f) * scale;
    const Eigen::Vector2f direction = Eigen::Vector2f(cut.stop.dx, cut.stop.dy).normalized();
    if (!direction.allFinite() || direction.isZero()) {
        return false;
    }
    const Eigen::Vector2f stop(cut.stop.x, cut.stop.y);
    const Eigen::Vector2f start = stop - (length / 2.f) * direction;
    if (!sample_line(image, start, stop + (length / 2.f) * direction, signal)) {
        return false;
    }
    constexpr std::array<std::array<float, 9>, 3> kernels{
        {{-0.0000f, -0.0003f, -0.1065f, -0.7863f, 0.f, 0.7863f, 0.1065f, 0.0003f, 0.0000f},
         {-0.0044f, -0.0540f, -0.2376f, -0.3450f, 0.f, 0.3450f, 0.2376f, 0.0540f, 0.0044f},
         {-0.0366f, -0.1113f, -0.1801f, -0.1594f, 0.f, 0.1594f, 0.1801f, 0.1113f, 0.0366f}}
    };
    float maximum = -std::numeric_limits<float>::infinity();
    int location = 0;
    for (const auto& kernel : kernels) {
        for (int i = 0; i < static_cast<int>(signal.size()); ++i) {
            float value = 0;
            for (int j = 0; j < 9; ++j) {
                const int index = std::clamp(i - 4 + j, 0, static_cast<int>(signal.size()) - 1);
                value += signal[index] * kernel[j];
            }
            // First sample and first kernel win equal peaks, as in the legacy map
            if (value > maximum) {
                maximum = value;
                location = i;
            }
        }
    }
    const float step = length / (signal.size() - 1.f);
    cut.stop.x = start.x() + step * location * direction.x();
    cut.stop.y = start.y() + step * location * direction.y();
    return true;
}

int select_cuts(
    const CandidateMarker& candidate,
    Image image,
    float begin,
    const Parameters& params,
    IdentificationScratch& scratch
) {
    const auto& ellipse = candidate.rescaled_outer_ellipse;
    scratch.angles.clear();
    scratch.selected.clear();
    scratch.eligible.clear();
    for (std::size_t i = 0; i < candidate.outer_points.size(); ++i) {
        const auto& point = candidate.outer_points[i];
        scratch.angles.emplace_back(std::atan2(point.y - ellipse.cy, point.x - ellipse.cx), i);
    }
    std::sort(scratch.angles.begin(), scratch.angles.end());
    const auto count = std::min(params._nSamplesOuterEllipse, scratch.angles.size());
    if (count < 5) {
        return no_collected_cuts;
    }
    if (scratch.cuts.size() < count) {
        scratch.cuts.resize(count);
    }
    const float step = std::max(1.f, float(scratch.angles.size()) / float(count - 1));
    std::size_t collected = 0, sampled = 0;
    for (std::size_t k = 0; std::size_t(k * step) < scratch.angles.size(); ++k) {
        ++sampled;
        auto& cut = scratch.cuts[collected];
        cut.stop = candidate.outer_points[scratch.angles[std::size_t(k * step)].second];
        cut.out_of_bounds = false;
        cut.signal.resize(params._sampleCutLength);
        const Eigen::Vector2f start(
            ellipse.cx + (cut.stop.x - ellipse.cx) * begin,
            ellipse.cy + (cut.stop.y - ellipse.cy) * begin
        );
        if (sample_line(image, start, {cut.stop.x, cut.stop.y}, cut.signal)) {
            using namespace boost::accumulators;
            accumulator_set<float, features<tag::variance>> statistics;
            for (const float value : cut.signal) {
                statistics(value);
            }
            cut.variance = variance(statistics);
            ++collected;
        }
    }
    if (sampled < 5 || collected == 0) {
        return no_collected_cuts;
    }
    float maximum_variance = 0;
    for (std::size_t i = 0; i < collected; ++i) {
        maximum_variance = std::max(maximum_variance, scratch.cuts[i].variance);
    }
    scratch.refinement.resize(params._numSamplesOuterEdgePointsRefinement);
    for (std::size_t i = 0; i < collected; ++i) {
        auto& cut = scratch.cuts[i];
        const auto point = point_on_ellipse(ellipse, cut.stop);
        cut.stop.x = point.x();
        cut.stop.y = point.y();
        if (refine_outer_point(cut, image, candidate.scale, scratch.refinement)
            && cut.variance / maximum_variance > 0.5f) {
            scratch.eligible.push_back(i);
        }
    }
    const auto selected_count = std::min(params._numCutsInIdentStep, collected);
    if (selected_count == 0) {
        return no_selected_cuts;
    }
    const float selection_step = std::max(1.f, float(scratch.eligible.size()) / selected_count);
    for (std::size_t k = 0; std::size_t(k * selection_step) < scratch.eligible.size()
         && scratch.selected.size() < selected_count;
         ++k) {
        scratch.selected.push_back(scratch.eligible[std::size_t(k * selection_step)]);
    }
    return scratch.selected.empty() ? no_selected_cuts : id_reliable;
}

/// Canonical conic and the transforms from image coordinates and back
struct CanonicalEllipse {
    float q1, q2, q3;
    Eigen::Matrix3f primal;
    Eigen::Matrix3f dual;
};

CanonicalEllipse canonical_form(const Ellipse& ellipse) {
    const auto& q = ellipse.conic;
    const float angle = 0.5f * std::atan2(2 * q(0, 1), q(0, 0) - q(1, 1));
    const float c = std::cos(angle), s = std::sin(angle);
    const float au = 2 * q(0, 2) * c + 2 * q(1, 2) * s;
    const float av = -2 * q(0, 2) * s + 2 * q(1, 2) * c;
    const float auu = q(0, 0) * (c * c) + q(1, 1) * (s * s) + 2 * q(0, 1) * (s * c);
    const float avv = q(0, 0) * (s * s) + q(1, 1) * (c * c) - 2 * q(0, 1) * (s * c);
    const float u = -au / (2 * auu), v = -av / (2 * avv);
    const float x = u * c - v * s, y = u * s + v * c;
    CanonicalEllipse result;
    result.q1 = c * (c * q(0, 0) + q(0, 1) * s) + s * (c * q(0, 1) + q(1, 1) * s);
    result.q2 = c * (c * q(1, 1) - q(0, 1) * s) - s * (c * q(0, 1) - q(0, 0) * s);
    result.q3 = q(2, 2) + x * (q(0, 2) + q(0, 0) * x + q(0, 1) * y)
        + y * (q(1, 2) + q(0, 1) * x + q(1, 1) * y) + q(0, 2) * x + q(1, 2) * y;
    result.primal << c, s, -c * x - s * y, -s, c, s * x - c * y, 0, 0, c * c + s * s;
    result.dual << c, -s, x, s, c, y, 0, 0, 1;
    return result;
}

bool homography_from_center(
    const CanonicalEllipse& ellipse,
    Eigen::Vector2f center,
    Eigen::Matrix3f& homography
) {
    const auto point = project(ellipse.primal, center.x(), center.y());
    const float x = point.x(), y = point.y();
    const float q1 = ellipse.q1, q2 = ellipse.q2, q3 = ellipse.q3;
    homography << q3, q2 * x * y, -q3 * x, 0.f, -q1 * x * x - q3, -q3 * y, -q1 * x, q2 * y, -q3;
    const std::array<float, 3> diagonal{
        std::sqrt(q2 * q3 / q1 * (q1 * x * x + q2 * y * y + q3)),
        q3,
        std::sqrt(-q2 * (q1 * x * x + q3))
    };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            homography(i, j) *= diagonal[j];
        }
    }
    homography = ellipse.dual * homography;
    return homography.allFinite();
}

void rectify_cuts(
    Image image,
    const Eigen::Matrix3f& homography,
    float begin,
    IdentificationScratch& scratch
) {
    const Eigen::Matrix3f inverse = homography.inverse();
    for (const auto index : scratch.selected) {
        auto& cut = scratch.cuts[index];
        const auto stop = project(inverse, cut.stop.x, cut.stop.y);
        const Eigen::Vector2f start = stop * begin;
        const float step_x = (stop.x() - start.x()) / (cut.signal.size() - 1.f);
        const float step_y = (stop.y() - start.y()) / (cut.signal.size() - 1.f);
        float x = start.x(), y = start.y();
        for (float& value : cut.signal) {
            const auto point = project(homography, x, y);
            if (point.x() >= 0.f && point.x() < image.width - 1.f && point.y() >= 0.f
                && point.y() < image.height - 1.f) {
                value = bilinear(image, point.x(), point.y());
            } else {
                // Once a search point leaves the image, the legacy keeps that cut excluded
                cut.out_of_bounds = true;
            }
            x += step_x;
            y += step_y;
        }
    }
}

float cut_cost(IdentificationScratch& scratch) {
    float residual = 0;
    std::size_t pairs = 0;
    for (std::size_t i = 0; i < scratch.selected.size(); ++i) {
        const auto& left = scratch.cuts[scratch.selected[i]];
        for (std::size_t j = i + 1; j < scratch.selected.size(); ++j) {
            const auto& right = scratch.cuts[scratch.selected[j]];
            if (left.out_of_bounds || right.out_of_bounds) {
                continue;
            }
            for (std::size_t k = 0; k < left.signal.size(); ++k) {
                residual += std::pow(left.signal[k] - right.signal[k], 2);
            }
            ++pairs;
        }
    }
    return pairs == 0 ? std::numeric_limits<float>::max() : residual / pairs;
}

bool refine_center(
    Marker& marker,
    Image image,
    float begin,
    const Parameters& params,
    IdentificationScratch& scratch,
    float& residual
) {
    const auto& ellipse = marker.outer_ellipse;
    const auto canonical = canonical_form(ellipse);
    const float mean_axis = (ellipse.a + ellipse.b) / 2.f;
    const float sqrt2 = std::sqrt(2.f);
    Eigen::Matrix3f conditioner;
    // The scale belongs to this ellipse; the legacy's static mean axis depends on the first call
    conditioner << sqrt2 / mean_axis, 0.f, -sqrt2 * ellipse.cx / mean_axis, 0.f, sqrt2 / mean_axis,
        -sqrt2 * ellipse.cy / mean_axis, 0.f, 0.f, 1.f;
    const Eigen::Matrix3f inverse = conditioner.inverse();
    Ellipse conditioned;
    if (!conditioned.set_conic(inverse.transpose() * ellipse.conic * inverse)) {
        return false;
    }
    float neighbour_size = params._imagedCenterNeighbourSize;
    const auto grid = params._imagedCenterNGridSample;
    if (neighbour_size * std::max(ellipse.a, ellipse.b) <= 0.02) {
        if (!homography_from_center(canonical, marker.center, marker.homography)) {
            return false;
        }
        rectify_cuts(image, marker.homography, begin, scratch);
        residual = cut_cost(scratch);
    }
    while (neighbour_size * std::max(ellipse.a, ellipse.b) > 0.02) {
        const float width = neighbour_size * std::max(conditioned.a, conditioned.b);
        const float half_width = width / 2.f;
        const float step = width / (grid - 1);
        const Eigen::Vector3f center =
            conditioner * Eigen::Vector3f(marker.center.x(), marker.center.y(), 1.f);
        float minimum = std::numeric_limits<float>::max();
        Eigen::Vector2f best_center = marker.center;
        Eigen::Matrix3f best_homography = marker.homography;
        for (std::size_t i = 0; i < grid; ++i) {
            for (std::size_t j = 0; j < grid; ++j) {
                const Eigen::Vector3f point = inverse
                    * Eigen::Vector3f(center.x() / center.z() - half_width + i * step,
                                      center.y() / center.z() - half_width + j * step,
                                      1.f);
                const Eigen::Vector2f image_point = point.head<2>() / point.z();
                Eigen::Matrix3f homography;
                if (!homography_from_center(canonical, image_point, homography)) {
                    continue;
                }
                rectify_cuts(image, homography, begin, scratch);
                const float cost = cut_cost(scratch);
                if (cost < minimum) {
                    minimum = cost;
                    best_center = image_point;
                    best_homography = homography;
                }
            }
        }
        residual = minimum;
        if (minimum == std::numeric_limits<float>::max()) {
            return false;
        }
        marker.center = best_center;
        marker.homography = best_homography;
        neighbour_size /= float((grid - 1) / 2);
    }
    rectify_cuts(image, marker.homography, begin, scratch);
    scratch.barcode.resize(params._sampleCutLength);
    for (std::size_t i = 0; i < scratch.barcode.size(); ++i) {
        scratch.sample.clear();
        for (const auto index : scratch.selected) {
            if (!scratch.cuts[index].out_of_bounds) {
                scratch.sample.push_back(scratch.cuts[index].signal[i]);
            }
        }
        if (scratch.sample.size() < 2) {
            return false;
        }
        std::sort(scratch.sample.begin(), scratch.sample.end());
        const auto middle = scratch.sample.size() / 2;
        scratch.barcode[i] = scratch.sample.size() % 2 != 0
            ? scratch.sample[middle]
            : (scratch.sample[middle] + scratch.sample[middle - 1]) / 2.0;
    }
    const auto [minimum, maximum] =
        std::minmax_element(scratch.barcode.begin(), scratch.barcode.end());
    residual = std::sqrt(double(residual)) / (*maximum - *minimum);
    return std::isfinite(residual) && residual <= 2.7f;
}

/// Reads cuts sequentially so equal id vote counts retain the lowest bank id
int orazio_distance_robust(
    Marker& marker,
    const MarkerBank& bank,
    float begin,
    const Parameters& params,
    IdentificationScratch& scratch
) {
    using namespace boost::accumulators;
    scratch.scores.assign(bank.offsets.size() - 1, {});
    for (const auto index : scratch.selected) {
        const auto& cut = scratch.cuts[index];
        if (cut.out_of_bounds) {
            continue;
        }
        accumulator_set<float, features<tag::variance>> statistics;
        for (std::size_t i = 30; i < cut.signal.size(); ++i) {
            statistics(cut.signal[i]);
        }
        const float midpoint = mean(statistics), signal_variance = variance(statistics);
        accumulator_set<float, features<tag::mean>> below, above;
        bool accumulate = false;
        for (const float value : cut.signal) {
            if (value < midpoint) {
                accumulate = true;
            }
            if (accumulate) {
                if (value < midpoint) {
                    below(value);
                } else {
                    above(value);
                }
            }
        }
        const float black = mean(below), white = mean(above);
        if (!(signal_variance > 0) || !std::isfinite(black) || !std::isfinite(white)) {
            continue;
        }
        const float step = (1.f - begin) / (cut.signal.size() - 1.f);
        float best_probability = -1;
        std::size_t best_id = 0;
        for (std::size_t id = 0; id < scratch.scores.size(); ++id) {
            float x = begin, distance = 0;
            for (const float value : cut.signal) {
                std::size_t crossings = 0;
                for (std::size_t r = bank.offsets[id]; r < bank.offsets[id + 1]; ++r) {
                    crossings += 1.f / bank.ratios[r] <= x;
                }
                const float difference = crossings % 2 != 0 ? std::max(value - black, 0.f)
                                                            : std::min(value - white, 0.f);
                distance += (difference * difference) / (2.f * signal_variance);
                x += step;
            }
            const float probability = std::exp(-distance);
            // Equal probabilities overwrite the id in the legacy's per-cut map
            if (probability >= best_probability) {
                best_probability = probability;
                best_id = id;
            }
        }
        if (best_probability >= 0) {
            auto& score = scratch.scores[best_id];
            ++score.count;
            score.sum += best_probability;
        }
    }
    const auto best = std::max_element(
        scratch.scores.begin(),
        scratch.scores.end(),
        [](const auto& a, const auto& b) { return a.count < b.count; }
    );
    if (best == scratch.scores.end() || best->count == 0) {
        return id_not_reliable;
    }
    marker.id = static_cast<std::int32_t>(best - scratch.scores.begin());
    const Eigen::Matrix3f inverse = marker.homography.inverse();
    for (std::size_t i = bank.offsets[marker.id]; i < bank.offsets[marker.id + 1]; ++i) {
        const float radius = 1.f / bank.ratios[i];
        Eigen::Matrix3f circle = Eigen::Matrix3f::Identity();
        circle(2, 2) = -radius * radius;
        Ellipse projected;
        if (!projected.set_conic(inverse.transpose() * circle * inverse)) {
            return degenerate;
        }
    }
    return best->sum / best->count > params._minIdentProba ? id_reliable : id_not_reliable;
}

void identify(
    const CandidateMarker& candidate,
    Marker& marker,
    IdentificationScratch& scratch,
    Image image,
    const MarkerBank& bank,
    const Parameters& params
) {
    marker = {};
    marker.center = candidate.center;
    marker.outer_ellipse = candidate.rescaled_outer_ellipse;
    marker.quality = candidate.quality;
    if (!params._doIdentification) {
        marker.status = 0;
        return;
    }
    const float begin = params._nCrowns == 3 ? 1 - (2 * params._nCrowns - 1) * 0.15f : 0.26f;
    marker.status = select_cuts(candidate, image, begin, params, scratch);
    if (marker.status != id_reliable) {
        return;
    }
    float residual = std::numeric_limits<float>::max();
    const bool converged = refine_center(marker, image, begin, params, scratch, residual);
    marker.quality = 1.f / residual;
    marker.status = converged ? orazio_distance_robust(marker, bank, begin, params, scratch)
                              : opti_has_diverged;
}

/// Replaces every overlapping reliable detection with the better one, preserving list order
void update(std::vector<Marker>& markers, const Marker& marker) {
    bool found = false;
    for (auto& current : markers) {
        if (current.status <= 0 || marker.status <= 0) {
            continue;
        }
        const auto& a = current.outer_ellipse;
        const auto& b = marker.outer_ellipse;
        const float dx = a.cx - b.cx, dy = a.cy - b.cy;
        const float distance = dx * dx + dy * dy;
        if (distance < (a.b * 0.5f) * (a.b * 0.5f) || distance < (b.b * 0.5f) * (b.b * 0.5f)) {
            if (marker.quality > current.quality) {
                current = marker;
            }
            found = true;
        }
    }
    if (!found) {
        markers.push_back(marker);
    }
}

} // namespace

void Backend::markers(Context<Backend>& context, const Parameters& params) {
    if (params._doIdentification
        && (params._sampleCutLength <= 30 || params._numSamplesOuterEdgePointsRefinement < 2
            || params._imagedCenterNGridSample < 5 || params._imagedCenterNGridSample % 2 == 0
            || !(params._imagedCenterNeighbourSize > 0)
            || !std::isfinite(params._imagedCenterNeighbourSize))) {
        throw std::invalid_argument("markers: invalid cut sampling or center search parameters");
    }
    context.bank.ensure(params._nCrowns);
    const auto count = context.candidate_markers.size();
    context.identified_markers.resize(count);
    if (context.identification.size() < count) {
        context.identification.resize(count);
    }
    const Image image = host_pyramid(context.levels[0]).src;
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < static_cast<int>(count); ++i) {
        identify(
            context.candidate_markers[i],
            context.identified_markers[i],
            context.identification[i],
            image,
            context.bank,
            params
        );
    }

    // Candidate markers already follow level order, then candidate order within each level
    context.preliminary_markers.clear();
    for (const auto& marker : context.identified_markers) {
        update(context.preliminary_markers, marker);
    }
    context.markers.clear();
    for (const auto& marker : context.preliminary_markers) {
        update(context.markers, marker);
    }
    // Stable insertion sort keeps equal ids in pinned order and needs no temporary allocation
    for (std::size_t i = 1; i < context.markers.size(); ++i) {
        const Marker marker = context.markers[i];
        std::size_t j = i;
        while (j > 0 && marker.id < context.markers[j - 1].id) {
            context.markers[j] = context.markers[j - 1];
            --j;
        }
        context.markers[j] = marker;
    }
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::cpu::tests {

using namespace boost::ut;

suite<"markers_stage"> markers_suite = [] {
    "markers search valid nearby centers when the initial center is outside the ellipse"_test = [] {
        Context<Backend> context;
        const Parameters params(3);
        context.ensure(256, 256, params);
        // Three concentric black rings, with boundaries at radii 20, 24, 28, 32, 36 and 40
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                const float radius = std::hypot(float(x - 128), float(y - 128));
                const bool black = (radius >= 20 && radius < 24) || (radius >= 28 && radius < 32)
                    || (radius >= 36 && radius < 40);
                context.levels[0].src.at<std::uint8_t>(y, x) = black ? 40 : 200;
            }
        }
        CandidateMarker candidate;
        candidate.center = {169, 128};
        expect(candidate.rescaled_outer_ellipse.set_parameters(128, 128, 40, 40, 0));
        for (int i = 0; i < 160; ++i) {
            const float angle = i * 6.2831853f / 160;
            const float x = std::cos(angle), y = std::sin(angle);
            candidate.outer_points.push_back({128 + 40 * x, 128 + 40 * y, x, y});
        }
        context.candidate_markers = {candidate};
        Backend::markers(context, params);
        expect(eq(context.markers.size(), 1u)) << fatal;
        // The initial homography is invalid, but the first grid reaches inside the ellipse
        expect(context.markers.front().center.x() < 168.f);
        expect(context.markers.front().homography.allFinite());
    };

    "markers retain unidentified candidates when identification is disabled"_test = [] {
        Context<Backend> context;
        Parameters params(3);
        params._doIdentification = false;
        context.ensure(32, 32, params);
        CandidateMarker candidate;
        candidate.center = {12, 13};
        expect(candidate.rescaled_outer_ellipse.set_parameters(14, 15, 8, 6, 0.3f));
        candidate.quality = 7;
        context.candidate_markers = {candidate, candidate};
        Backend::markers(context, params);
        // Unidentified overlaps survive both dedup passes, with the candidate's center intact
        expect(eq(context.markers.size(), 2u)) << fatal;
        for (const auto& marker : context.markers) {
            expect(eq(marker.id, -1));
            expect(eq(marker.status, 0));
            expect(marker.center == candidate.center);
            expect(eq(marker.outer_ellipse.cx, 14.f));
            expect(eq(marker.quality, 7.f));
        }
        context.candidate_markers.clear();
        Backend::markers(context, params);
        expect(context.markers.empty());
        const auto view = host_markers(context);
        expect(view.xy.empty() && view.ids.empty() && view.statuses.empty());
    };

    "markers reject unreadable outer points without retaining earlier cuts"_test = [] {
        Context<Backend> context;
        const Parameters params(3);
        context.ensure(32, 32, params);
        context.levels[0].src.setTo(0);
        CandidateMarker candidate;
        candidate.center = {16, 16};
        expect(candidate.rescaled_outer_ellipse.set_parameters(16, 16, 6, 6, 0));
        for (int i = 0; i < 24; ++i) {
            const float angle = i * 6.2831853f / 24;
            const float x = std::cos(angle), y = std::sin(angle);
            candidate.outer_points.push_back({16 + 6 * x, 16 + 6 * y, x, y});
        }
        context.candidate_markers = {candidate};
        Backend::markers(context, params);
        expect(eq(context.markers.front().status, -2)); // Constant cuts fail the variance gate
        for (auto& point : context.candidate_markers.front().outer_points) {
            point.x += 100;
        }
        Backend::markers(context, params);
        expect(eq(context.markers.front().status, -1)); // All cuts leave the image
        context.candidate_markers.front().outer_points.resize(4);
        Backend::markers(context, params);
        expect(eq(context.markers.front().status, -1)); // Too few outer points
    };

    "markers reject search parameters that cannot sample or shrink"_test = [] {
        Context<Backend> context;
        const Parameters defaults(3);
        context.ensure(32, 32, defaults);
        for (const std::size_t grid : {0u, 1u, 3u, 4u}) {
            Parameters params = defaults;
            params._imagedCenterNGridSample = grid;
            expect(throws<std::invalid_argument>([&] { Backend::markers(context, params); }));
        }
        Parameters params = defaults;
        params._sampleCutLength = 30;
        expect(throws<std::invalid_argument>([&] { Backend::markers(context, params); }));
        params = defaults;
        params._numSamplesOuterEdgePointsRefinement = 1;
        expect(throws<std::invalid_argument>([&] { Backend::markers(context, params); }));
        Backend::markers(context, defaults);
        expect(context.markers.empty());
    };
};

} // namespace cctag::portable::cpu::tests
#endif // CCTAG_TEST
