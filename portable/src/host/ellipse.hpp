/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_ELLIPSE_HPP
#define CCTAG_PORTABLE_HOST_ELLIPSE_HPP

#include <Eigen/Core>
#include <Eigen/SVD>

#include <array>
#include <span>
#include <vector>

namespace cctag::portable {

/// An ellipse's geometric parameters and symmetric conic matrix
struct Ellipse {
    float cx = 0;
    float cy = 0;
    float a = 0;
    float b = 0;
    float angle = 0;
    Eigen::Matrix3f conic = Eigen::Matrix3f::Zero();

    /// Keeps both representations in sync, returning false for a degenerate ellipse
    bool set_parameters(float x, float y, float semi_a, float semi_b, float rotation);
    bool set_conic(const Eigen::Matrix3f& matrix);
};

/// Retains the fitting matrices between candidates and frames
struct EllipseFitScratch {
    Eigen::MatrixX3f quadratic;
    Eigen::MatrixX3f linear;
    /// Eigen reallocates a dynamic SVD when its row count changes during growing
    /// Retain one workspace per fit in a growth operation, reused in the next frame
    struct CircleFit {
        Eigen::MatrixXf points;
        Eigen::JacobiSVD<Eigen::MatrixXf> svd;
    };
    std::vector<CircleFit> circles;
    std::size_t next_circle = 0;
};

/// Direct least-squares fit with the legacy's centering and ellipse constraint
bool fit_ellipse(
    std::span<const Eigen::Vector2f> points,
    EllipseFitScratch& scratch,
    Ellipse& ellipse
);
/// Algebraic circle fit used to initialize an arc with little gradient variation
bool fit_circle(
    std::span<const Eigen::Vector2f> points,
    EllipseFitScratch& scratch,
    Ellipse& ellipse
);
/// Conic through five points, with the constant coefficient fixed to one
bool ellipse_through_five(const std::array<Eigen::Vector2f, 5>& points, Ellipse& ellipse);

float distance_to_ellipse(const Ellipse& ellipse, float x, float y);
bool in_ellipse(const Ellipse& ellipse, float x, float y);
bool in_hull(const Ellipse& inner, const Ellipse& outer, float x, float y);
bool ellipse_hull(const Ellipse& ellipse, float width, Ellipse& inner, Ellipse& outer);
int ellipse_perimeter(const Ellipse& ellipse);
/// Sorted x intersections with a horizontal line, returning zero, one or two
int ellipse_line(const Ellipse& ellipse, float y, std::array<float, 2>& intersections);

} // namespace cctag::portable

#endif
