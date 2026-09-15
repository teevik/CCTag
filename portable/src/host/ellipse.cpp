/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "host/ellipse.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace cctag::portable {

bool Ellipse::set_parameters(float x, float y, float semi_a, float semi_b, float rotation) {
    if (!(semi_a > 0 && semi_b > 0) || !std::isfinite(x) || !std::isfinite(y)
        || !std::isfinite(semi_a) || !std::isfinite(semi_b) || !std::isfinite(rotation)) {
        return false;
    }
    cx = x;
    cy = y;
    a = semi_a;
    b = semi_b;
    angle = rotation;
    Eigen::Matrix3f transform;
    transform << std::cos(angle), -std::sin(angle), cx, std::sin(angle), std::cos(angle), cy, 0.f,
        0.f, 1.f;
    Eigen::Matrix3f inverse;
    bool invertible;
    transform.computeInverseWithCheck(inverse, invertible);
    if (!invertible) {
        return false;
    }
    Eigen::Matrix3f diagonal = Eigen::Matrix3f::Identity();
    diagonal(0, 0) = 1.f / (a * a);
    diagonal(1, 1) = 1.f / (b * b);
    diagonal(2, 2) = -1.f;
    conic = diagonal * inverse;
    conic = inverse.transpose() * conic;
    return conic.allFinite();
}

bool Ellipse::set_conic(const Eigen::Matrix3f& matrix) {
    const float aa = matrix(0, 0), bb = 2.f * matrix(0, 1), cc = matrix(1, 1);
    const float dd = 2.f * matrix(0, 2), ee = 2.f * matrix(1, 2);
    const float rotation = 0.5f * std::atan2(bb, aa - cc);
    const float cosine = std::cos(rotation), sine = std::sin(rotation);
    const float au = dd * cosine + ee * sine;
    const float av = -dd * sine + ee * cosine;
    const float auu = aa * (cosine * cosine) + cc * (sine * sine) + bb * (sine * cosine);
    const float avv = aa * (sine * sine) + cc * (cosine * cosine) - bb * (sine * cosine);
    if (auu == 0 || avv == 0) {
        return false;
    }
    const float u = -au / (2.f * auu), v = -av / (2.f * avv);
    const float w = matrix(2, 2) - auu * u * u - avv * v * v;
    const float ru = -w / auu, rv = -w / avv;
    if (!(ru > 0 && rv > 0) || !std::isfinite(ru) || !std::isfinite(rv)) {
        return false;
    }
    cx = u * cosine - v * sine;
    cy = u * sine + v * cosine;
    a = std::sqrt(ru);
    b = std::sqrt(rv);
    angle = rotation;
    conic = matrix;
    return std::isfinite(cx) && std::isfinite(cy) && matrix.allFinite();
}

bool fit_ellipse(
    std::span<const Eigen::Vector2f> points,
    EllipseFitScratch& scratch,
    Ellipse& ellipse
) {
    if (points.size() < 5) {
        return false;
    }
    const auto n = static_cast<Eigen::Index>(points.size());
    Eigen::Vector2f offset = Eigen::Vector2f::Zero();
    for (const auto& point : points) {
        offset += point;
    }
    offset /= static_cast<float>(n);
    if (scratch.quadratic.rows() < n) {
        scratch.quadratic.resize(n, 3);
        scratch.linear.resize(n, 3);
    }
    auto quadratic = scratch.quadratic.topRows(n);
    auto linear = scratch.linear.topRows(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Eigen::Vector2f point = points[i] - offset;
        quadratic.row(i) =
            Eigen::Vector3f(point.x() * point.x(), point.x() * point.y(), point.y() * point.y());
        linear.row(i) = Eigen::Vector3f(point.x(), point.y(), 1.f);
    }
    const Eigen::Matrix3f s1 = quadratic.transpose() * quadratic;
    const Eigen::Matrix3f s2 = quadratic.transpose() * linear;
    const Eigen::Matrix3f s3 = linear.transpose() * linear;
    Eigen::Matrix3f inverse;
    bool invertible;
    s3.computeInverseWithCheck(inverse, invertible);
    if (!invertible) {
        return false;
    }
    Eigen::Matrix3f constraint;
    constraint << 0.f, 0.f, 0.5f, 0.f, -1.f, 0.f, 0.5f, 0.f, 0.f;
    const Eigen::Matrix3f t = -s3.inverse() * s2.transpose();
    const Eigen::Matrix3f m = constraint * (s1 + s2 * t);
    if (!m.allFinite()) {
        return false;
    }
    const Eigen::EigenSolver<Eigen::Matrix3f> solver(m);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const Eigen::Matrix3f vectors = solver.eigenvectors().real();
    const float epsilon = std::numeric_limits<float>::epsilon();
    float minimum = std::numeric_limits<float>::max();
    int selected = -1;
    for (int i = 0; i < 3; ++i) {
        const float condition = 4 * vectors(0, i) * vectors(2, i) - vectors(1, i) * vectors(1, i);
        if (condition > epsilon && condition < minimum) {
            selected = i;
            minimum = condition;
        }
    }
    if (selected == -1) {
        return false;
    }
    Eigen::Matrix<float, 6, 1> coefficients;
    coefficients.head<3>() = vectors.col(selected);
    coefficients.tail<3>() = t * vectors.col(selected);
    float determinant = coefficients(0) * coefficients(2) - coefficients(1) * coefficients(1) / 4;
    determinant = determinant > epsilon ? 1.f / determinant : 0;
    const float scale = std::sqrt(determinant / 4);
    if (scale < epsilon) {
        return false;
    }
    coefficients *= scale;
    const float aa = coefficients(0), bb = coefficients(1), cc = coefficients(2);
    const float dd = coefficients(3), ee = coefficients(4);
    float ff = coefficients(5);
    const Eigen::Vector2f center =
        Eigen::Vector2f(-dd * cc + ee * bb / 2, -aa * ee + dd * bb / 2) * 2;
    ff += aa * center.x() * center.x() + bb * center.x() * center.y() + cc * center.y() * center.y()
        + dd * center.x() + ee * center.y();
    if (std::abs(ff) < epsilon) {
        return false;
    }
    Eigen::Matrix2f s;
    s << aa, bb / 2, bb / 2, cc;
    s /= -ff;
    if (!s.allFinite()) {
        return false;
    }
    const Eigen::JacobiSVD<Eigen::Matrix2f> svd(s, Eigen::ComputeFullU);
    const auto& values = svd.singularValues();
    const auto& u = svd.matrixU();
    return ellipse.set_parameters(
        center.x() + offset.x(),
        center.y() + offset.y(),
        std::sqrt(1.f / values(0)),
        std::sqrt(1.f / values(1)),
        std::numbers::pi_v<float> - std::atan2(u(0, 1), u(1, 1))
    );
}

bool fit_circle(
    std::span<const Eigen::Vector2f> points,
    EllipseFitScratch& scratch,
    Ellipse& ellipse
) {
    if (points.size() < 5) {
        return false;
    }
    const auto n = static_cast<Eigen::Index>(points.size());
    if (scratch.next_circle == scratch.circles.size()) {
        scratch.circles.emplace_back();
    }
    auto& circle = scratch.circles[scratch.next_circle++];
    circle.points.resize(n, 4);
    for (Eigen::Index i = 0; i < n; ++i) {
        const float x = points[i].x(), y = points[i].y();
        circle.points.row(i) = Eigen::Vector4f(x, y, 1.f, x * x + y * y);
    }
    circle.svd.compute(circle.points, Eigen::ComputeThinV);
    const auto& v = circle.svd.matrixV();
    const float x = -0.5f * v(0, 3) / v(3, 3);
    const float y = -0.5f * v(1, 3) / v(3, 3);
    const float radius = std::sqrt(x * x + y * y - v(2, 3) / v(3, 3));
    return ellipse.set_parameters(x, y, radius, radius, 0);
}

bool ellipse_through_five(const std::array<Eigen::Vector2f, 5>& points, Ellipse& ellipse) {
    Eigen::Matrix<float, 5, 5> a;
    for (int i = 0; i < 5; ++i) {
        const float x = points[i].x(), y = points[i].y();
        a.row(i) << x * x, 2.f * x * y, y * y, 2.f * x, 2.f * y;
    }
    if (a.determinant() == 0.f) {
        return false;
    }
    const Eigen::Matrix<float, 5, 1> coefficients =
        a.lu().solve(Eigen::Matrix<float, 5, 1>::Constant(-1.f));
    if (!(coefficients(0) * coefficients(2) - coefficients(1) * coefficients(1) > 0)) {
        return false;
    }
    Eigen::Matrix3f conic;
    conic << coefficients(0), coefficients(1), coefficients(3), coefficients(1), coefficients(2),
        coefficients(4), coefficients(3), coefficients(4), 1.f;
    return ellipse.set_conic(conic);
}

float distance_to_ellipse(const Ellipse& ellipse, float x, float y) {
    const auto& q = ellipse.conic;
    const float u = x * q(0, 0) + y * q(0, 1) + q(0, 2);
    const float v = x * q(0, 1) + y * q(1, 1) + q(1, 2);
    Eigen::Matrix<float, 6, 1> point, coefficients;
    point << x * x, 2 * x * y, 2 * x, y * y, 2 * y, 1.f;
    coefficients << q(0, 0), q(0, 1), q(0, 2), q(1, 1), q(1, 2), q(2, 2);
    const float product = point.dot(coefficients);
    return product * product / (u * u + v * v);
}

bool in_ellipse(const Ellipse& ellipse, float x, float y) {
    const Eigen::Vector3f point(x, y, 1.f), center(ellipse.cx, ellipse.cy, 1.f);
    return point.dot(ellipse.conic * point) * point.dot(ellipse.conic * center) > 0;
}

bool in_hull(const Ellipse& inner, const Ellipse& outer, float x, float y) {
    const Eigen::Vector3f point(x, y, 1.f);
    return point.dot(inner.conic * point) * point.dot(outer.conic * point) < 0;
}

bool ellipse_hull(const Ellipse& ellipse, float width, Ellipse& inner, Ellipse& outer) {
    return inner.set_parameters(
               ellipse.cx,
               ellipse.cy,
               std::max(ellipse.a - width, 0.001f),
               std::max(ellipse.b - width, 0.001f),
               ellipse.angle
           )
        && outer.set_parameters(
            ellipse.cx,
            ellipse.cy,
            ellipse.a + width,
            ellipse.b + width,
            ellipse.angle
        );
}

int ellipse_perimeter(const Ellipse& ellipse) {
    const float sine = std::sin(ellipse.angle), cosine = std::cos(ellipse.angle);
    const float a1 = -ellipse.b * sine - ellipse.b * cosine;
    const float b1 = -ellipse.a * cosine + ellipse.a * sine;
    const float a2 = -ellipse.b * sine + ellipse.b * cosine;
    const float b2 = -ellipse.a * cosine - ellipse.a * sine;
    const float t1 = std::atan2(-a1, b1), t2 = std::atan2(-a2, b2);
    const auto point = [&](float theta) -> Eigen::Vector2f {
        // The legacy uses double trigonometry here, then rounds the float coordinates
        const float x = ellipse.a * std::cos(static_cast<double>(theta));
        const float y = ellipse.b * std::sin(static_cast<double>(theta));
        const float px = x * std::cos(static_cast<double>(ellipse.angle))
            - y * std::sin(static_cast<double>(ellipse.angle)) + ellipse.cx;
        const float py = x * std::sin(static_cast<double>(ellipse.angle))
            + y * std::cos(static_cast<double>(ellipse.angle)) + ellipse.cy;
        return {std::round(px), std::round(py)};
    };
    const Eigen::Vector2f p11 = point(t1), p12 = point(t1 + std::numbers::pi_v<float>);
    const Eigen::Vector2f p22 = point(t2 + std::numbers::pi_v<float>);
    const float perimeter =
        2.f * ((p22 - p11).cwiseAbs().maxCoeff() + (p12 - p22).cwiseAbs().maxCoeff());
    if (!std::isfinite(perimeter)
        || perimeter >= static_cast<float>(std::numeric_limits<int>::max())) {
        return 0;
    }
    return static_cast<int>(perimeter);
}

int ellipse_line(const Ellipse& ellipse, float y, std::array<float, 2>& intersections) {
    const auto& q = ellipse.conic;
    const float a = q(0, 0), b = 2 * (y * q(0, 1) + q(0, 2));
    const float c = q(1, 1) * (y * y) + 2 * y * q(2, 1) + q(2, 2);
    const float discriminant = b * b / 4.f - a * c;
    if (discriminant > 0.f) {
        const float root = std::sqrt(discriminant);
        intersections = {(-b / 2.f - root) / a, (-b / 2.f + root) / a};
        return 2;
    }
    if (discriminant == 0.f) {
        intersections[0] = -b / (2.f * a);
        return 1;
    }
    return 0;
}

} // namespace cctag::portable

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::ellipse {

using namespace boost::ut;

inline suite<"ellipse"> ellipse_suite = [] {
    "ellipse fitting recovers a translated rotated conic"_test = [] {
        // Points from the parametric ellipse equation, independent of the fitter
        std::array<Eigen::Vector2f, 32> points;
        constexpr float cx = 12.f, cy = -7.f, a = 5.f, b = 13.f, angle = 0.37f;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const double theta = 2 * std::numbers::pi * i / points.size();
            const double x = a * std::cos(theta), y = b * std::sin(theta);
            points[i] = Eigen::Vector2f(
                cx + x * std::cos(angle) - y * std::sin(angle),
                cy + x * std::sin(angle) + y * std::cos(angle)
            );
        }
        EllipseFitScratch scratch;
        Ellipse ellipse;
        expect(fit_ellipse(points, scratch, ellipse)) << fatal;
        expect(std::abs(ellipse.cx - cx) < 1e-3f);
        expect(std::abs(ellipse.cy - cy) < 1e-3f);
        expect(std::abs(ellipse.a - a) < 1e-3f);
        expect(std::abs(ellipse.b - b) < 1e-3f);
        expect(std::abs(std::remainder(ellipse.angle - angle, std::numbers::pi_v<float>)) < 1e-3f);
        for (const auto& point : points) {
            expect(distance_to_ellipse(ellipse, point.x(), point.y()) < 1e-5f);
        }
    };

    "circle fitting recovers a partial circular arc"_test = [] {
        std::array<Eigen::Vector2f, 20> points;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const double angle = 0.1 + 0.6 * i / (points.size() - 1);
            points[i] = Eigen::Vector2f(20 + 8 * std::cos(angle), 30 + 8 * std::sin(angle));
        }
        EllipseFitScratch scratch;
        Ellipse ellipse;
        expect(fit_circle(points, scratch, ellipse)) << fatal;
        expect(std::abs(ellipse.cx - 20.f) < 0.01f);
        expect(std::abs(ellipse.cy - 30.f) < 0.01f);
        expect(std::abs(ellipse.a - 8.f) < 0.01f);
        expect(eq(ellipse.a, ellipse.b));
    };

    "ellipse fitting rejects too few or collinear points"_test = [] {
        const std::array<Eigen::Vector2f, 5> points = {{{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}}};
        EllipseFitScratch scratch;
        Ellipse ellipse;
        expect(!fit_ellipse(std::span(points).first(4), scratch, ellipse));
        expect(!fit_ellipse(points, scratch, ellipse));
        expect(!ellipse_through_five(points, ellipse));
    };
};

} // namespace cctag::portable::tests::ellipse
#endif
