/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>

namespace cctag::portable::cpu {

namespace {

Eigen::Vector2f position(EdgePointsHost points, std::int32_t index) {
    return {static_cast<float>(points.xy[2 * index]), static_cast<float>(points.xy[2 * index + 1])};
}

Eigen::Vector2f gradient(EdgePointsHost points, std::int32_t index) {
    return {points.gradients[2 * index], points.gradients[2 * index + 1]};
}

DirectedPoint directed_point(EdgePointsHost points, std::int32_t index) {
    const auto point = position(points, index);
    const auto direction = gradient(points, index);
    return {point.x(), point.y(), direction.x(), direction.y()};
}

void clear_marks(CandidateSlot& slot) {
    for (const auto index : slot.touched) {
        slot.processed[index] = 0;
    }
    slot.touched.clear();
}

void mark(CandidateSlot& slot, std::int32_t index) {
    if (!slot.processed[index]) {
        slot.processed[index] = 1;
        slot.touched.push_back(index);
    }
}

void copy_fit_points(
    EdgePointsHost points,
    std::span<const std::int32_t> indices,
    CandidateSlot& slot
) {
    slot.fit_points.clear();
    for (const auto index : indices) {
        slot.fit_points.push_back(position(points, index));
    }
}

float median(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

float median_distance(
    EdgePointsHost points,
    std::span<const std::int32_t> indices,
    const Ellipse& ellipse,
    CandidateSlot& slot
) {
    slot.distances.clear();
    for (const auto index : indices) {
        const auto point = position(points, index);
        slot.distances.push_back(distance_to_ellipse(ellipse, point.x(), point.y()));
    }
    return median(slot.distances);
}

/// Robust conic estimation on a regularly spaced subset, preserving the legacy's draw order
void remove_outliers(
    EdgePointsHost points,
    std::span<const std::int32_t> children,
    float threshold,
    bool weighted,
    CandidateSlot& slot,
    std::vector<std::int32_t>& filtered
) {
    filtered.clear();
    const auto count = std::min(children.size(), std::size_t{60});
    if (count < 5) {
        return;
    }
    const float step = static_cast<float>(children.size()) / static_cast<float>(count);
    std::array<Eigen::Vector2f, 60> sample;
    std::array<float, 60> weights;
    std::size_t sampled = 0;
    for (std::size_t i = 0; i < children.size() && sampled < count; ++i) {
        if (i == static_cast<std::size_t>(sampled * step)) {
            sample[sampled] = position(points, children[i]);
            weights[sampled] = weighted ? 255 / gradient(points, children[i]).norm() : 1.f;
            ++sampled;
        }
    }
    Ellipse best;
    float best_median = 10000000.f;
    bool found = false;
    int trials = 0;
    while (trials < 70) {
        const auto indices = kernels::rand_5_k(slot.random, static_cast<std::uint32_t>(sampled));
        std::array<Eigen::Vector2f, 5> selected;
        for (int i = 0; i < 5; ++i) {
            selected[i] = sample[indices[i]];
        }
        Ellipse ellipse;
        ++trials;
        if (!ellipse_through_five(selected, ellipse)) {
            continue;
        }
        const float ratio = ellipse.a / ellipse.b;
        if (ratio < 0.04f || ratio > 25) {
            continue;
        }
        slot.distances.clear();
        for (std::size_t i = 0; i < sampled; ++i) {
            slot.distances.push_back(
                distance_to_ellipse(ellipse, sample[i].x(), sample[i].y()) * weights[i]
            );
        }
        const float distance = median(slot.distances);
        if (distance < best_median) {
            trials = 0;
            best = ellipse;
            best_median = distance;
            found = true;
        }
    }
    if (!found) {
        return;
    }
    for (const auto index : children) {
        const auto point = position(points, index);
        float distance = distance_to_ellipse(best, point.x(), point.y());
        if (weighted) {
            distance = distance * 255 / gradient(points, index).norm();
        }
        if (distance < threshold * best_median) {
            filtered.push_back(index);
        }
    }
}

/// Chooses an ellipse initialization once two sampled gradient directions differ enough
bool good_growing_points(EdgePointsHost points, std::span<const std::int32_t> children) {
    auto direction = gradient(points, children.front()).normalized().eval();
    float minimum = 1.1f;
    std::int32_t least_aligned = children.front();
    for (std::size_t i = 1; i < children.size(); ++i) {
        const float product = direction.dot(gradient(points, children[i]).normalized());
        if (product <= 0.25f) {
            return true;
        }
        if (product < minimum) {
            minimum = product;
            least_aligned = children[i];
        }
    }
    direction = gradient(points, least_aligned).normalized();
    for (const auto index : children) {
        if (direction.dot(gradient(points, index).normalized()) <= 0.25f) {
            return true;
        }
    }
    return false;
}

bool grow_hull(const Buffers& level, EdgePointsHost points, CandidateSlot& slot, float width) {
    Ellipse inner, outer;
    if (!ellipse_hull(slot.ellipse, width, inner, outer)) {
        return false;
    }
    constexpr std::array<int, 8> x_offset = {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr std::array<int, 8> y_offset = {0, -1, -1, -1, 0, 1, 1, 1};
    const auto initial_size = slot.outer_points.size();
    for (std::size_t i = 0; i < initial_size; ++i) {
        slot.stack.clear();
        slot.stack.push_back({slot.outer_points[i], 0});
        mark(slot, slot.outer_points[i]);
        while (!slot.stack.empty()) {
            auto& visit = slot.stack.back();
            if (visit.neighbour == 8) {
                slot.stack.pop_back();
                continue;
            }
            const auto point = position(points, visit.point);
            const int neighbour = visit.neighbour++;
            const int x = static_cast<int>(point.x()) + x_offset[neighbour];
            const int y = static_cast<int>(point.y()) + y_offset[neighbour];
            if (x < 0 || y < 0 || x >= static_cast<int>(level.width)
                || y >= static_cast<int>(level.height)) {
                continue;
            }
            const auto index = level.edge_map(y, x);
            if (index < 0 || slot.processed[index] || !in_hull(inner, outer, x, y)) {
                continue;
            }
            if (gradient(points, index).dot(Eigen::Vector2f(inner.cx - x, inner.cy - y)) < 0) {
                slot.outer_points.push_back(index);
                mark(slot, index);
                slot.stack.push_back({index, 0});
            }
        }
    }
    return true;
}

bool grow_ellipse(const Buffers& level, EdgePointsHost points, CandidateSlot& slot, float width) {
    slot.fit.next_circle = 0;
    const bool good_init = good_growing_points(points, slot.filtered_children);
    copy_fit_points(points, slot.filtered_children, slot);
    if (!(good_init ? fit_ellipse(slot.fit_points, slot.fit, slot.ellipse)
                    : fit_circle(slot.fit_points, slot.fit, slot.ellipse))) {
        return false;
    }
    clear_marks(slot);
    slot.outer_points = slot.filtered_children;
    for (const auto index : slot.outer_points) {
        mark(slot, index);
    }
    if (!good_init) {
        std::size_t previous = 0;
        std::size_t current = slot.outer_points.size();
        std::size_t maximum = current;
        slot.best_points = slot.outer_points;
        Ellipse best = slot.ellipse;
        while (current > previous) {
            Ellipse inner, outer;
            if (!ellipse_hull(slot.ellipse, width, inner, outer)) {
                return false;
            }
            const auto count_in_hull = [&] {
                std::size_t count = 0;
                for (const auto index : slot.outer_points) {
                    const auto point = position(points, index);
                    count += in_hull(inner, outer, point.x(), point.y());
                }
                return count;
            };
            previous = count_in_hull();
            if (!grow_hull(level, points, slot, width)) {
                return false;
            }
            // Each saved point set belongs to the ellipse used to grow it, before the next fit
            const Ellipse grown_from = slot.ellipse;
            copy_fit_points(points, slot.outer_points, slot);
            if (!fit_circle(slot.fit_points, slot.fit, slot.ellipse)
                || !ellipse_hull(slot.ellipse, width, inner, outer)) {
                return false;
            }
            current = count_in_hull();
            if (current > maximum) {
                maximum = current;
                slot.best_points = slot.outer_points;
                best = grown_from;
            }
        }
        slot.outer_points = slot.best_points;
        slot.ellipse = best;
        clear_marks(slot);
        for (const auto index : slot.outer_points) {
            mark(slot, index);
        }
    }
    copy_fit_points(points, slot.outer_points, slot);
    if (!fit_ellipse(slot.fit_points, slot.fit, slot.ellipse)) {
        return false;
    }
    std::size_t previous = 0;
    while (slot.outer_points.size() > previous) {
        previous = slot.outer_points.size();
        if (!grow_hull(level, points, slot, width)) {
            return false;
        }
        copy_fit_points(points, slot.outer_points, slot);
        if (!fit_ellipse(slot.fit_points, slot.fit, slot.ellipse)) {
            return false;
        }
    }
    return true;
}

void complete_flow_component(
    const Buffers& level,
    EdgePointsHost points,
    std::span<const std::int32_t> children,
    CandidateSlot& slot,
    const Parameters& params
) {
    if (children.size() < params._minPointsSegmentCandidate) {
        return;
    }
    slot.score = children.size();
    remove_outliers(
        points,
        children,
        params._threshRobustEstimationOfOuterEllipse,
        true,
        slot,
        slot.filtered_children
    );
    if (slot.filtered_children.size() < 5
        || !grow_ellipse(level, points, slot, params._ellipseGrowingEllipticHullWidth)) {
        return;
    }
    if (median_distance(points, slot.outer_points, slot.ellipse, slot)
        > params._thrMedianDistanceEllipse) {
        return;
    }
    const float quality =
        static_cast<float>(slot.outer_points.size()) / ellipse_perimeter(slot.ellipse);
    const float ratio = slot.ellipse.a / slot.ellipse.b;
    slot.accepted = quality <= 1.1 && ratio >= 0.05 && ratio <= 20;
}

/// Walks the alternating links from the outer ring and checks the inner gradient orientations
bool add_flow(
    EdgePointsHost points,
    VoteHost vote,
    std::span<const std::int32_t> children,
    std::span<const std::int32_t> outer_points,
    const Ellipse& ellipse,
    std::size_t circles,
    CandidateSlot& slot
) {
    clear_marks(slot);
    for (const auto index : outer_points) {
        slot.flow_outer_points.push_back(directed_point(points, index));
    }
    std::size_t gradient_out = 0, added = 0;
    bool valid = true;
    for (const auto child : children) {
        int direction = -1;
        auto index = child;
        const auto outer = position(points, child);
        const float a = outer.x() - ellipse.cx, b = outer.y() - ellipse.cy;
        const Eigen::Vector3f line(a, b, -a * ellipse.cx - b * ellipse.cy);
        for (std::size_t ring = 1; ring < circles; ++ring) {
            index = vote.links[2 * index + (direction == -1 ? 0 : 1)];
            if (index < 0 || static_cast<std::uint32_t>(index) >= points.n) {
                valid = false;
                break;
            }
            if (!slot.processed[index]) {
                mark(slot, index);
                const auto point = position(points, index);
                const auto toward_center =
                    Eigen::Vector2f(ellipse.cx - point.x(), ellipse.cy - point.y())
                        .normalized()
                        .eval();
                const float same_side = Eigen::Vector3f(outer.x(), outer.y(), 1.f).dot(line)
                    * Eigen::Vector3f(point.x(), point.y(), 1.f).dot(line);
                if (!in_ellipse(ellipse, point.x(), point.y()) || !(same_side > 0)) {
                    valid = false;
                    break;
                }
                if (ring >= circles - 2) {
                    gradient_out += static_cast<float>(-direction)
                            * gradient(points, index).normalized().dot(toward_center)
                        < -0.5f;
                    ++added;
                }
            }
            direction = -direction;
        }
        if (!valid) {
            break;
        }
    }
    clear_marks(slot);
    if (!valid || static_cast<float>(gradient_out) / static_cast<float>(added) > 0.5f) {
        slot.flow_outer_points.clear();
        return false;
    }
    return true;
}

bool another_segment(
    EdgePointsHost points,
    VoteHost vote,
    const CandidateSlot& partner,
    Ellipse& ellipse,
    CandidateSlot& slot,
    const Parameters& params
) {
    const float reference_distance = median_distance(points, slot.outer_points, ellipse, slot);
    float best_distance = std::numeric_limits<float>::max();
    int trials = 0;
    while (trials < 100) {
        const auto first =
            kernels::rand_5_k(slot.random, static_cast<std::uint32_t>(slot.outer_points.size()));
        const auto second =
            kernels::rand_5_k(slot.random, static_cast<std::uint32_t>(partner.outer_points.size()));
        std::array<Eigen::Vector2f, 8> selected;
        for (int i = 0; i < 4; ++i) {
            selected[i] = position(points, slot.outer_points[first[i]]);
            selected[4 + i] = position(points, partner.outer_points[second[i]]);
        }
        ++trials;
        Ellipse fit, conic;
        if (!fit_ellipse(selected, slot.fit, fit) || !conic.set_conic(fit.conic)) {
            continue;
        }
        const float ratio = conic.a / conic.b;
        if (ratio < 0.12 || ratio > 8) {
            continue;
        }
        const float first_distance = median_distance(points, slot.outer_points, conic, slot);
        const float distance =
            first_distance + median_distance(points, partner.outer_points, conic, slot);
        if (distance < best_distance) {
            best_distance = distance;
            trials = 0;
        }
    }
    if (!(best_distance < 6 * reference_distance)) {
        return false;
    }
    slot.merged_points = slot.outer_points;
    slot.merged_points
        .insert(slot.merged_points.end(), partner.outer_points.begin(), partner.outer_points.end());
    copy_fit_points(points, slot.merged_points, slot);
    Ellipse assembled;
    if (!fit_ellipse(slot.fit_points, slot.fit, assembled)) {
        return false;
    }
    const float quality =
        static_cast<float>(slot.merged_points.size()) / ellipse_perimeter(assembled);
    if (!(quality < 1.1)
        || !(
            median_distance(points, slot.merged_points, assembled, slot)
            < params._thrMedianDistanceEllipse
        )
        || !add_flow(
            points,
            vote,
            partner.filtered_children,
            partner.outer_points,
            assembled,
            params._nCrowns * 2,
            slot
        )) {
        return false;
    }
    ellipse = assembled;
    return true;
}

void make_candidate(
    EdgePointsHost points,
    VoteHost vote,
    std::span<const CandidateSlot> slots,
    CandidateSlot& slot,
    int level,
    const Parameters& params
) {
    Ellipse ellipse = slot.ellipse;
    std::span<const std::int32_t> outer_points = slot.outer_points;
    float quality = static_cast<float>(outer_points.size()) / ellipse_perimeter(ellipse);
    slot.flow_outer_points.clear();
    if (params._searchForAnotherSegment && quality > 0.25 && quality < 0.7) {
        const auto seed = position(points, slot.seed);
        const float flow = vote.flow_length[slot.seed];
        Ellipse area;
        const bool valid_area =
            area.set_parameters(seed.x(), seed.y(), flow * 2.5f, flow * 2.5f, 0);
        const CandidateSlot* partner = nullptr;
        for (const auto& other : slots) {
            if (!other.accepted || &other == &slot || other.label == slot.label) {
                continue;
            }
            const float ratio = vote.flow_length[other.seed] / flow;
            const auto other_seed = position(points, other.seed);
            if (valid_area && ratio > 0.666 && ratio < 1.5
                && in_ellipse(area, other_seed.x(), other_seed.y())
                && (!partner || other.score > partner->score)) {
                partner = &other;
            }
        }
        if (partner && another_segment(points, vote, *partner, ellipse, slot, params)) {
            outer_points = slot.merged_points;
            quality = static_cast<float>(outer_points.size()) / ellipse_perimeter(ellipse);
        }
    }
    if (!add_flow(
            points,
            vote,
            slot.filtered_children,
            slot.outer_points,
            ellipse,
            params._nCrowns * 2,
            slot
        )) {
        return;
    }
    const float scale = std::ldexp(1.f, level);
    Ellipse rescaled;
    if (!rescaled.set_parameters(
            ellipse.cx,
            ellipse.cy,
            ellipse.a * scale,
            ellipse.b * scale,
            ellipse.angle
        )) {
        return;
    }
    const float size = quality * ellipse_perimeter(rescaled);
    if ((quality <= 0.35 && size >= 300.f) || (quality <= 0.45f && size >= 200.f && size < 300.f)
        || (quality <= 0.5f && size >= 100.f && size < 200.f)
        || (quality <= 0.5f && size >= 70.f && size < 100.f)
        || (quality <= 0.96f && size >= 50.f && size < 70.f) || size < 50.f) {
        return;
    }
    const float ratio = ellipse.a / ellipse.b;
    if (ratio > 8.0 || ratio < 0.125) {
        return;
    }
    Ellipse inner, outer;
    if (!ellipse_hull(ellipse, 3.6f, inner, outer)) {
        return;
    }
    float gradient_quality = 0;
    for (const auto index : outer_points) {
        const auto point = position(points, index);
        if (!in_hull(inner, outer, point.x(), point.y())) {
            return;
        }
        gradient_quality += gradient(points, index).norm();
    }
    auto& marker = slot.marker;
    marker.level = level;
    marker.scale = scale;
    marker.quality = gradient_quality * scale;
    marker.center = {ellipse.cx, ellipse.cy};
    // CCTag's constructor shifts the fitted ellipse by half a pixel before rescaling
    // Preserve validation at the original level without retaining that representation
    Ellipse shifted_outer;
    if (!shifted_outer.set_parameters(
            ellipse.cx + 0.5f,
            ellipse.cy + 0.5f,
            ellipse.a,
            ellipse.b,
            ellipse.angle
        )
        || !marker.rescaled_outer_ellipse.set_parameters(
            (ellipse.cx + 0.5f) * scale,
            (ellipse.cy + 0.5f) * scale,
            ellipse.a * scale,
            ellipse.b * scale,
            ellipse.angle
        )) {
        return;
    }
    marker.outer_points.clear();
    for (auto point : slot.flow_outer_points) {
        point.x *= scale;
        point.y *= scale;
        marker.outer_points.push_back(point);
    }
    slot.has_marker = true;
}

/// Collects a horizontal hull interval from the canonical level-zero host view
void collect_interval(
    EdgePointsHost points,
    int width,
    int y,
    float begin,
    float end,
    const Ellipse& inner,
    std::vector<std::int32_t>& collected
) {
    if (!std::isfinite(begin) || !std::isfinite(end) || end <= -1.f || begin >= width) {
        return;
    }
    const int first = static_cast<int>(std::max(0.f, begin));
    const int last = static_cast<int>(std::min(static_cast<float>(width - 1), end));
    std::uint32_t lower = 0, upper = points.n;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        const auto x_mid = points.xy[2 * middle], y_mid = points.xy[2 * middle + 1];
        if (y_mid < y || (y_mid == y && x_mid < first)) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    for (auto index = lower; index < points.n; ++index) {
        const auto point = position(points, index);
        if (point.y() != y || point.x() > last) {
            break;
        }
        if (gradient(points, index).dot(Eigen::Vector2f(inner.cx - point.x(), inner.cy - point.y()))
            < 0) {
            collected.push_back(static_cast<std::int32_t>(index));
        }
    }
}

void refit_at_level_zero(EdgePointsHost points, int width, int height, CandidateSlot& slot) {
    auto& marker = slot.marker;
    if (marker.level == 0) {
        return;
    }
    Ellipse inner, outer;
    if (!ellipse_hull(marker.rescaled_outer_ellipse, marker.scale, inner, outer)) {
        return;
    }
    slot.hull_points.clear();
    const auto collect_row = [&](int y) {
        std::array<float, 2> outside, inside;
        const int out_count = ellipse_line(outer, y, outside);
        const int in_count = ellipse_line(inner, y, inside);
        if (out_count == 2 && in_count == 2) {
            collect_interval(points, width, y, outside[0], inside[0], inner, slot.hull_points);
            collect_interval(points, width, y, inside[1], outside[1], inner, slot.hull_points);
        } else if (out_count == 2 && in_count <= 1) {
            collect_interval(points, width, y, outside[0], outside[1], inner, slot.hull_points);
        } else if (out_count == 1 && in_count == 0) {
            if (outside[0] >= 0 && outside[0] < width) {
                collect_interval(points, width, y, outside[0], outside[0], inner, slot.hull_points);
            }
        } else {
            return false;
        }
        return true;
    };
    const float center_y = marker.rescaled_outer_ellipse.cy;
    if (center_y <= static_cast<float>(std::numeric_limits<int>::min())
        || center_y >= static_cast<float>(std::numeric_limits<int>::max())) {
        return;
    }
    // The legacy visits the centre row twice: downward first, then upward
    for (int y = std::max(static_cast<int>(center_y), 0); y < height && collect_row(y); ++y) {}
    for (int y = std::min(static_cast<int>(center_y), height - 1); y >= 0 && collect_row(y); --y) {}
    remove_outliers(points, slot.hull_points, 20.f, false, slot, slot.merged_points);
    if (slot.merged_points.size() < 5) {
        return;
    }
    copy_fit_points(points, slot.merged_points, slot);
    Ellipse refitted;
    if (!fit_ellipse(slot.fit_points, slot.fit, refitted)) {
        return;
    }
    marker.rescaled_outer_ellipse = refitted;
    marker.center *= marker.scale;
    marker.outer_points.clear();
    for (const auto index : slot.merged_points) {
        marker.outer_points.push_back(directed_point(points, index));
    }
}

} // namespace

void Backend::candidates(Context<Backend>& context, const Parameters& params) {
    context.candidate_levels.resize(context.levels.size());
    std::size_t marker_count = 0;
    for (int index = static_cast<int>(context.levels.size()) - 1; index >= 0; --index) {
        auto& level = context.levels[index];
        auto& candidates = context.candidate_levels[index];
        const auto count =
            std::min(level.loop_one_order.size(), params._maximumNbCandidatesLoopTwo);
        if (candidates.slots.size() < count) {
            candidates.slots.resize(count);
        }
        const EdgePointsHost points = host_edge_points(level);
        const VoteHost vote = host_vote(level);
        const std::span<CandidateSlot> slots(candidates.slots.data(), count);
        for (std::size_t i = 0; i < count; ++i) {
            auto& slot = slots[i];
            clear_marks(slot);
            slot.processed.resize(level.n, 0);
            slot.seed = level.link_seeds[level.loop_one_order[i]];
            slot.accepted = false;
            slot.has_marker = false;
            slot.label = -1;
            slot.filtered_children.clear();
            slot.outer_points.clear();
            kernels::pcg32_seed(
                slot.random,
                271828,
                (std::uint64_t{static_cast<std::uint32_t>(index)} << 32)
                    | static_cast<std::uint32_t>(slot.seed)
            );
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(count); ++i) {
            const auto candidate = level.loop_one_order[i];
            const auto children =
                std::span<const std::int32_t>(level.children_values)
                    .subspan(level.children_offsets[candidate], level.child_counts[candidate]);
            complete_flow_component(level, points, children, slots[i], params);
        }

        // Assign labels in list order after every loop-two result is complete
        for (const auto point : candidates.labelled_points) {
            candidates.segment_label[point] = -1;
        }
        candidates.labelled_points.clear();
        candidates.segment_label.resize(level.n, -1);
        std::int32_t next_label = 0;
        for (auto& slot : slots) {
            if (!slot.accepted) {
                continue;
            }
            for (const auto child : slot.filtered_children) {
                if (candidates.segment_label[child] != -1) {
                    slot.label = candidates.segment_label[child];
                    break;
                }
            }
            if (slot.label == -1) {
                slot.label = next_label++;
            }
            for (const auto child : slot.filtered_children) {
                if (candidates.segment_label[child] == -1) {
                    candidates.labelled_points.push_back(child);
                }
                candidates.segment_label[child] = slot.label;
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(count); ++i) {
            if (slots[i].accepted) {
                make_candidate(points, vote, slots, slots[i], index, params);
            }
        }
    }

    // Refit through level zero's host view after every level's candidate markers are known
    const EdgePointsHost points = host_edge_points(context.levels[0]);
    for (int index = static_cast<int>(context.levels.size()) - 1; index >= 0; --index) {
        auto& slots = context.candidate_levels[index].slots;
        const auto count = std::min(
            context.levels[index].loop_one_order.size(),
            params._maximumNbCandidatesLoopTwo
        );
        for (std::size_t i = 0; i < count; ++i) {
            auto& slot = slots[i];
            if (!slot.has_marker) {
                continue;
            }
            refit_at_level_zero(points, context.width, context.height, slot);
            if (marker_count == context.candidate_markers.size()) {
                context.candidate_markers.emplace_back();
            }
            context.candidate_markers[marker_count++] = slot.marker;
        }
    }
    context.candidate_markers.resize(marker_count);
}

} // namespace cctag::portable::cpu
