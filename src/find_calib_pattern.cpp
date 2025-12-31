/*
 * File: find_calib_pattern.cpp
 * Author: Neucrede <neucrede@sina.com>
 */

/*
BSD 2-Clause License

Copyright (c) 2018, Neucrede <neucrede@sina.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <list>
#include <stdexcept>
#include <memory>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "find_calib_pattern.hpp"

extern bool FitEllipseSubPixel(const cv::Mat& gradX, const cv::Mat& gradY, 
    const std::vector<cv::Point>& points, cv::RotatedRect& ellipse); // fit_ellipse_subpixel.cpp

template <typename Tp1, typename Tp2>
static inline double PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB);

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const cv::Point_<Tp2>& pt0, const cv::Point_<Tp3>& ptLineA, 
        const cv::Point_<Tp3>& ptLineB);

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0, 
        const cv::Point_<Tp3>& ptLineA, const cv::Point_<Tp3>& ptLineB);

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0, const std::vector<int>& exclusions = std::vector<int>());

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0);

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices);

template <typename Tp1, typename Tp2>
static bool SortEllipsesAndCenterPoints(cv::Size patSize, const std::vector<cv::Point_<Tp1>>& cornerPoints, 
    const std::vector<cv::RotatedRect>& ellipses, std::vector<cv::RotatedRect>& sortedEllipses,
    const std::vector<cv::Point_<Tp2>>& centerPoints, std::vector<cv::Point_<Tp2>>& sortedCenterPoints);

static bool HierarchicalClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    std::vector<int> &patternPointIndices);

static bool ScanlineClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    const cv::Size imageSize, double distThresh, std::vector<int> &patternPointIndices,
    std::vector<cv::Point2d>* pCornerPoints = nullptr);

static bool ExtractContoursHalconCalibBoard(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered,
        bool inverseThresh = false);



bool FindCheckerPattern(const cv::Mat& img, std::vector<cv::Point2d>& sortedCorners,
    cv::Size patSize, int thresh, bool inverseThresh, bool subPixel, const cv::Mat& mask)
{
    if (img.empty()) {
        throw std::invalid_argument("img is empty.");
    }

    if ((patSize.width < 5) || (patSize.height < 5)) {
        throw std::invalid_argument("Pattern size can't be smaller than 5x5.");
    }

    cv::Mat imgGray;
    if (img.channels() != 1) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        imgGray = img.clone();
    }

    if (!mask.empty()) {
        if (mask.type() != CV_8UC1) {
            throw std::invalid_argument("Bad mask data type. Must be CV_8UC1.");
        }

        imgGray = imgGray & mask;
    }

    cv::Mat imgMono;
    if (thresh < 0) {
        cv::threshold(imgGray, imgMono, -1, 255, 
            (inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) + cv::THRESH_OTSU);
    }
    else {
        cv::threshold(imgGray, imgMono, thresh, 255, 
            inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    }

    std::vector<cv::Point2f> corners;   // single precision required by cornerSubPix().
    bool found = cv::findChessboardCorners(imgMono, patSize, corners, 0);
    if (!found) {
        return false;
    }

    if (subPixel) {
        cv::cornerSubPix(imgGray, corners, cv::Size(7, 7), cv::Size(3, 3), 
            cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 0.01));
    }

    const int rows = patSize.height, cols = patSize.width;
    sortedCorners.clear();
    sortedCorners.reserve(rows * cols);
    for (int r = rows - 1; r >= 0; --r) {
        for (int c = 0; c < cols; ++c) {
            sortedCorners.push_back(cv::Point2d(corners[r * cols + c]));
        }
    }
    
    return true;
}

bool FindCirclesGridPattern(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize, int thresh, bool inverseThresh, bool subPixel, const cv::Mat& mask,
        const std::vector<cv::Point2d>& cornerPointsHint, std::vector<cv::RotatedRect>* sortedEllipses)
{
    if (img.empty()) {
        throw std::invalid_argument("img is empty.");
    }

    if ((patSize.width < 5) || (patSize.height < 5)) {
        throw std::invalid_argument("Pattern size can't be smaller than 5x5.");
    }

    cv::Mat imgGray;
    if (img.channels() != 1) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        imgGray = img.clone();
    }

    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 1.5);
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1) {
            throw std::invalid_argument("Bad mask data type. Must be CV_8UC1.");
        }

        imgGray = imgGray & mask;
    }

    cv::Mat imgMono;
    if (thresh < 0) {
        cv::threshold(imgGray, imgMono, -1, 255, 
            (inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) + cv::THRESH_OTSU);
    }
    else {
        cv::threshold(imgGray, imgMono, thresh, 255, 
            inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    }

    // Extract contours from binarized image.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgMono, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    const size_t total = patSize.width * patSize.height;
    const int numContours = contours.size();
    if (numContours < total) {
        return false;
    }

    // Compute image gradients in horizontal and vertical directions.
    cv::Mat gradX, gradY;
    cv::Sobel(imgGray, gradX, CV_32F, 1, 0);
    cv::Sobel(imgGray, gradY, CV_32F, 0, 1);

    // Fit ellipses blindly.
    std::vector<cv::RotatedRect> ellipses;
    ellipses.reserve(total);
    {
        cv::RotatedRect ellipse;
        for (int idx = 0; idx != numContours; ++idx) {
            cv::Moments moments = cv::moments(contours[idx]);
            double area = moments.m00;
            
            // Filter by area
            {
                const double areaThresh = img.rows * img.cols / patSize.area();
                if (area > areaThresh) {
                    continue;
                }
            }
            
            // Filter by circularity
            {
                double perim = cv::arcLength(contours[idx], true);
                if (4.0 * M_PI * area / (perim * perim + 1.0) < 0.75) {
                    continue;
                }
            }
            
            // Filter by convexity
            {
                std::vector<cv::Point> hull;
                cv::convexHull(contours[idx], hull);
                double hullArea = cv::contourArea(hull);
                if (area / (hullArea + 1.0) < 0.9) {
                    continue;
                }
            }
            
            if (subPixel) {
                if (!FitEllipseSubPixel(gradX, gradY, contours[idx], ellipse)) {
                    continue;
                }
            }
            else if (contours[idx].size() >= 6) {
                ellipse = cv::fitEllipseDirect(contours[idx]);
            }
            else {
                continue;
            }
            
            if (ellipse.boundingRect().area() >= 64) {
                ellipses.push_back(ellipse);
            }
        }
    }

    if (ellipses.size() < total) {
        return false;
    }
    
    // Sweep away overly small blobbs.
    {
        double maxEllipseArea = 0;
        for (const cv::RotatedRect& ellipse : ellipses) {
            double area = M_PI * ellipse.size.area();
            if (area > maxEllipseArea) {
                maxEllipseArea = area;
            }
        }
        
        double area0 = 0.5 * maxEllipseArea;
        
        std::vector<cv::RotatedRect> goodEllipses;
        goodEllipses.reserve(total);
        for (const cv::RotatedRect& ellipse : ellipses) {
            double area = M_PI * ellipse.size.area();
            if (area >= area0) {
                goodEllipses.push_back(ellipse);
            }
        }
        
        ellipses = std::move(goodEllipses);
    }
    
    const int numEllipses = ellipses.size();
    if (numEllipses < total) {
        return false;
    }

    std::vector<cv::Point2d> cornerPointsHintDup = cornerPointsHint;
    
    // Find largest cluster of center points.
    std::vector<int> blobIndices;
    {
        std::vector<cv::Point2d> points;
        points.reserve(numEllipses);
        for (const cv::RotatedRect& ellipse : ellipses) {
            points.push_back(ellipse.center);
        }

        // if (!HierarchicalClustering(points, patSize, blobIndices)) {
        if (!ScanlineClustering(points, patSize, cv::Size(img.cols, img.rows), -1,
            blobIndices, cornerPointsHintDup.size() < 4 ? &cornerPointsHintDup : nullptr)) 
        {
            return false;
        }
    }

    // Compute standard deviations of minor and major lengths of ellipses.
    std::vector<double> minorLengths, majorLengths;
    minorLengths.reserve(total);
    majorLengths.reserve(total);
    double meanMinorLength = 0.0f, meanMajorLength = 0.0f;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        meanMinorLength += minorLength;

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        meanMajorLength += majorLength;
    }

    meanMinorLength /= (double)(total);
    meanMajorLength /= (double)(total);

    double stddevMajorLength = 0.0f, stddevMinorLength = 0.0f;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        stddevMinorLength += std::pow(minorLength - meanMinorLength, 2);

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        stddevMajorLength += std::pow(majorLength - meanMajorLength, 2);
    }

    stddevMinorLength = std::sqrt(stddevMinorLength / (double)(total));
    stddevMajorLength = std::sqrt(stddevMajorLength / (double)(total));
    
    // The range of valid major and minor lengths.
    const double 
        lowMinLen = meanMinorLength - std::max(0.5f * meanMinorLength, 3.0f * stddevMinorLength),
        uppMinLen = meanMinorLength + std::max(0.5f * meanMinorLength, 3.0f * stddevMinorLength),
        lowMajLen = meanMajorLength - std::max(0.5f * meanMajorLength, 3.0f * stddevMajorLength),
        uppMajLen = meanMajorLength + std::max(0.5f * meanMajorLength, 3.0f * stddevMajorLength);

    // Filter ellipses by their major and minor lengths using 3ss.
    std::vector<cv::Point2d> centerPoints;
    centerPoints.reserve(total);
    std::vector<cv::RotatedRect> ellipsesFiltered;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        if ((minorLength < lowMinLen) || (minorLength > uppMinLen)) {
            continue;
        }

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        if ((majorLength < lowMajLen) || (majorLength > uppMajLen)) {
            continue;
        }

        ellipsesFiltered.push_back(ellipse);
        centerPoints.push_back(ellipse.center);
    }

    if (centerPoints.size() != total) {
        return false;
    }

    // If 4 corner points are given in the order { origin, rear X, diagonal, rearY }.
    if (cornerPointsHintDup.size() == 4) {
        std::vector<cv::RotatedRect> _sortedEllipses;
        if (!SortEllipsesAndCenterPoints(patSize, cornerPointsHintDup, ellipsesFiltered, _sortedEllipses,
            centerPoints, sortedCenterPoints)) 
        {
            return false;
        }

        if (sortedEllipses) {
            *sortedEllipses = std::move(_sortedEllipses);
        }

        return true;
    }
    // Otherwise...

    // Find four corners among center points.
    std::vector<int> cornerIndices;
    if (!FindFourCorners(centerPoints, cornerIndices)) {
        return false;
    }

    // Sort them in counter-clockwise order.
    {
        std::vector<cv::Point2f> points;
        points.reserve(4);
        for (int i : cornerIndices) {
            points.push_back(cv::Point2f(centerPoints[i]));
        }

        std::vector<int> hull;
        cv::convexHull(points, hull, true, false);

        std::vector<int> cornerIndices2;
        cornerIndices2.reserve(4);
        for (int i : hull) {
            cornerIndices2.push_back(cornerIndices[i]);
        }

        cornerIndices = std::move(cornerIndices2);
    }

    // Find bottom left corner.
    int idxBottomLeft = 0;
    const cv::Point2d ptImageBottomLeft(0.0, (double)(img.rows));
    double minDist = 1.0e9;
    for (int i = 0; i != 4; ++i) {
        int idx = cornerIndices[i];
        const cv::Point2d& pt = centerPoints[idx];
        double dist = std::hypot(pt.x - ptImageBottomLeft.x, pt.y - ptImageBottomLeft.y);
        if (dist < minDist) {
            idxBottomLeft = i;
            minDist = dist;
        }
    }

    // Sort in the order { origin = bottom left corner, rear X, diagonal, rearY }.
    std::vector<cv::Point2d> cornerPoints;
    cornerPoints.reserve(4);
    for (int i = idxBottomLeft; i != idxBottomLeft + 4; ++i) {
        cornerPoints.push_back(centerPoints[cornerIndices[i % 4]]);
    }

    std::vector<cv::RotatedRect> _sortedEllipses;
    if (!SortEllipsesAndCenterPoints(patSize, cornerPoints, ellipsesFiltered, _sortedEllipses,
        centerPoints, sortedCenterPoints)) 
    {
        return false;
    }

    if (sortedEllipses) {
        *sortedEllipses = std::move(_sortedEllipses);
    }

    return true;
}

bool FindHalconCalibBoard(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize, int thresh, bool subPixel, std::vector<cv::RotatedRect>* sortedEllipses)
{
    if (img.empty()) {
        throw std::invalid_argument("img is empty.");
    }

    if ((patSize.width < 5) || (patSize.height < 5)) {
        throw std::invalid_argument("Pattern size can't be smaller than 5x5.");
    }

    cv::Mat imgGray;
    if (img.channels() != 1) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        imgGray = img.clone();
    }

    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 1.5);

    const size_t total = patSize.width * patSize.height;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> outerContour, innerContour;
    std::vector<int> blobIndicesFiltered;
    if (!ExtractContoursHalconCalibBoard(imgGray, thresh, total, contours, outerContour, innerContour,
            blobIndicesFiltered))
    {
        return false;
    }

    // Compute image gradients in horizontal and vertical directions.
    cv::Mat gradX, gradY;
    cv::Sobel(imgGray, gradX, CV_32F, 1, 0);
    cv::Sobel(imgGray, gradY, CV_32F, 0, 1);

    // * Fit ellipses.
    // * Compute means and standard deviations of major and minor axis lengths.
    std::vector<cv::RotatedRect> ellipses;
    ellipses.reserve(total);
    std::vector<double> minorLengths, majorLengths;
    minorLengths.reserve(total);
    majorLengths.reserve(total);
    double meanMinorLength = 0.0f, meanMajorLength = 0.0f;
    cv::RotatedRect ellipse;
    for (int idx : blobIndicesFiltered) {
        if (subPixel) {
            if (FitEllipseSubPixel(gradX, gradY, contours[idx], ellipse)) {
                ellipses.push_back(ellipse);
            }
            else {
                return false;
            }
        }
        else {
            ellipse = cv::fitEllipseDirect(contours[idx]);
            if (ellipse.boundingRect().area() == 0) {
                return false;
            }
            else {
                ellipses.push_back(ellipse);
            }
        }

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        meanMinorLength += minorLength;

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        meanMajorLength += majorLength;
    }

    meanMinorLength /= (double)(total);
    meanMajorLength /= (double)(total);

    double stddevMajorLength = 0.0f, stddevMinorLength = 0.0f;
    for (const cv::RotatedRect& ellipse : ellipses) {
        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        stddevMinorLength += std::pow(minorLength - meanMinorLength, 2);

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        stddevMajorLength += std::pow(majorLength - meanMajorLength, 2);
    }

    stddevMinorLength = std::sqrt(stddevMinorLength / (double)(total));
    stddevMajorLength = std::sqrt(stddevMajorLength / (double)(total));
    
    const double 
        lowMinLen = meanMinorLength - std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        uppMinLen = meanMinorLength + std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        lowMajLen = meanMajorLength - std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength),
        uppMajLen = meanMajorLength + std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength);

    // * Filter ellipses by their major and minor lengths using 3ss.
    // * Compute center of ellipses.
    std::vector<cv::Point2d> centerPoints;
    centerPoints.reserve(total);
    std::vector<cv::RotatedRect> ellipsesFiltered;
    for (const cv::RotatedRect& ellipse : ellipses) {
        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        if ((minorLength < lowMinLen) || (minorLength > uppMinLen)) {
            return false;
        }

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        if ((majorLength < lowMajLen) || (majorLength > uppMajLen)) {
            return false;
        }

        ellipsesFiltered.push_back(ellipse);
        centerPoints.push_back(ellipse.center);
    }

    // Map corner points to rect grids.
    std::vector<cv::Point> rectifiedOuterPoints = { 
        {0, 0}, {patSize.width, 0}, {patSize.width, patSize.height}, {0, patSize.width} };
    cv::Mat H = cv::findHomography(outerContour, rectifiedOuterPoints, 0);
    std::vector<cv::Point2d> rectifiedInnerPoints;
    std::vector<cv::Point2d> innerContourDbl;
    innerContourDbl.reserve(innerContour.size());
    for (auto pt : innerContour) {
        innerContourDbl.push_back(cv::Point2d(pt));
    }
    cv::perspectiveTransform(innerContourDbl, rectifiedInnerPoints, H);

    // Find a outer corner nearest to the chamfer.
    int idx0 = 0;
    double maxDist = 0;
    for (int j = 0; j != 4; ++j) {
        const cv::Point& pt = rectifiedOuterPoints[j];
        
        int idx = FindNearestPoint(rectifiedInnerPoints, pt);
        if (idx < 0) {
            return false;
        }

        const cv::Point2d& pt1 = rectifiedInnerPoints[idx];
        double dist = std::hypot(pt.x - pt1.x, pt.y - pt1.y);
        if (dist > maxDist) {
            idx0 = j;
            maxDist = dist;
        }
    }
    const cv::Point& ptOuter0 = outerContour[idx0];
    const int idxOuter0 = idx0;

    // Let the point nearest to the chamfer be the origin.
    idx0 = FindNearestPoint(centerPoints, ptOuter0);
    if (idx0 < 0) {
        return false;
    }
    const cv::Point2d& ptOrigin = centerPoints[idx0];

    // Find 2 outer contour points falls on X and Y axes respectively. #1 --> X, #2 --> Y.
    cv::Point ptOuter1, ptOuter2;
    {
        const cv::Point& ptOuterA = outerContour[(idxOuter0 + 1) % 4];
        const cv::Point& ptOuterB = outerContour[(idxOuter0 + 3) % 4];
        const cv::Vec2i vecA(ptOuterA - ptOuter0), vecB(ptOuterB - ptOuter0);
        int crossProduct = vecA[0] * vecB[1] - vecA[1] * vecB[0];
        if (crossProduct > 0) {
            ptOuter1 = ptOuterB;
            ptOuter2 = ptOuterA;
        }
        else {
            ptOuter1 = ptOuterA;
            ptOuter2 = ptOuterB;
        }
    }

    // Find four corner points.
    std::vector<int> cornerIndices;
    if (!FindFourCorners(centerPoints, cornerIndices)) {
        return false;
    }

    // Sort corner indices in the order { origin, rear X, diagonal, rearY }
    std::vector<int> sortedCornerIndices = {
        FindNearestPoint(centerPoints, cornerIndices, ptOrigin),
        FindNearestPoint(centerPoints, cornerIndices, ptOuter1),
        FindNearestPoint(centerPoints, cornerIndices, outerContour[(idxOuter0 + 2) % 4]),
        FindNearestPoint(centerPoints, cornerIndices, ptOuter2) };
    std::vector<cv::Point2d> cornerPoints;
    cornerPoints.reserve(4);
    for (int i = 0; i != 4; ++i) {
        cornerPoints.push_back(centerPoints[sortedCornerIndices[i]]);
    }

    std::vector<cv::RotatedRect> _sortedEllipses;
    if (!SortEllipsesAndCenterPoints(patSize, cornerPoints, ellipsesFiltered, _sortedEllipses,
        centerPoints, sortedCenterPoints)) 
    {
        return false;
    }

    if (sortedEllipses) {
        *sortedEllipses = std::move(_sortedEllipses);
    }

    return true;
}

template <typename Tp1, typename Tp2>
static inline double PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB)
{
    double xP = pt.x - ptLineA.x, yP = pt.y - ptLineA.y, xV = ptLineB.x - ptLineA.x,
          yV = ptLineB.y - ptLineA.y;
    double vecLen = std::hypot(xV, yV) + 1.0e-9f;
    double crossProd = xP * yV - xV * yP;

    return std::abs(crossProd) / vecLen;
}

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const cv::Point_<Tp2>& pt0, const cv::Point_<Tp3>& ptLineA, 
        const cv::Point_<Tp3>& ptLineB)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[i];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        dist += PointLineDistance(pt, ptLineA, ptLineB);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0, 
        const cv::Point_<Tp3>& ptLineA, const cv::Point_<Tp3>& ptLineB)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[indices[i]];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        dist += PointLineDistance(pt, ptLineA, ptLineB);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0, const std::vector<int>& exclusions)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        bool skip = false;
        
        for (int idx : exclusions) {
            if (i == idx) {
                skip = true;
                break;
            }
        }
        
        if (skip) {
            continue;
        }
        
        const cv::Point_<Tp1>& pt = points[i];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != indices.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[indices[i]];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        if (dist < minDist) {
            idx = indices[i];
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices)
{
    std::vector<cv::Point2f> points32f;
    points32f.reserve(points.size());
    for (const auto& pt : points) {
        points32f.push_back(cv::Point2f(pt));
    }

    std::vector<int> hull;
    cv::convexHull(points32f, hull);
    if (hull.size() < 4) {
        return false;
    }

    // Compute cosine of angles formed by adjacent 3 vertices.
    std::vector<double> angleCosines;
    angleCosines.reserve(hull.size());
    for (int i = 0; i != hull.size(); ++i) {
        const int K = hull.size();
        cv::Vec2d vec1 = cv::Point2d(points[hull[(i + 1) % K]] - points[hull[i]]);
        cv::Vec2d vec2 = cv::Point2d(points[hull[(i - 1 + K) % K]] - points[hull[i]]);
        double cosAngle = std::abs((double)(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))));
        angleCosines.push_back(cosAngle);
    }

    // Sort angleCosines in ascending order. After sorting, the first 4 vertices 
    // with sharpest angles are considered corners.
    std::vector<int> sortedAngleCosineIndices(angleCosines.size());
    std::iota(sortedAngleCosineIndices.begin(), sortedAngleCosineIndices.end(), 0);
    std::sort(sortedAngleCosineIndices.begin(), sortedAngleCosineIndices.end(), 
            [&angleCosines] (int lhs, int rhs) -> bool {
                return angleCosines[lhs] < angleCosines[rhs];
            }
    );

    cornerIndices.clear();
    cornerIndices.reserve(4);
    for (int i = 0; i != 4; ++i) {
        int idx = hull[sortedAngleCosineIndices[i]];
        cornerIndices.push_back(idx);
    }
    std::sort(cornerIndices.begin(), cornerIndices.end());

    return true;
}

template <typename Tp1, typename Tp2>
static bool SortEllipsesAndCenterPoints(cv::Size patSize, const std::vector<cv::Point_<Tp1>>& cornerPoints, 
    const std::vector<cv::RotatedRect>& ellipses, std::vector<cv::RotatedRect>& sortedEllipses,
    const std::vector<cv::Point_<Tp2>>& centerPoints, std::vector<cv::Point_<Tp2>>& sortedCenterPoints)
{
    // Rectify centre points.
    std::vector<cv::Point2d> rectifiedPoints;
    std::vector<cv::Point2d> srcPoints;
    srcPoints.reserve(4);
    for (int i = 0; i != 4; ++i) {
        srcPoints.push_back(cornerPoints[i]);
    }

    double len1 = cv::norm(srcPoints[1] - srcPoints[0]),
           len2 = cv::norm(srcPoints[2] - srcPoints[0]);
    double stride = (double)(len1 + len2) / (double)(2 * (patSize.width + patSize.height - 2)) + 1.0f;
    std::vector<cv::Point2d> destPoints = {
        {0, 0}, 
        {(double)(patSize.width - 1) * stride, 0},
        {(double)(patSize.width - 1) * stride, (double)(patSize.height - 1) * stride},
        {0, (double)(patSize.height - 1) * stride} };

    cv::Mat H = cv::findHomography(srcPoints, destPoints, 0);
    cv::perspectiveTransform(centerPoints, rectifiedPoints, H);

    // Sort ellipses and centre points in ascending dictionary order.
    sortedCenterPoints.clear();
    sortedCenterPoints.reserve(patSize.width * patSize.height);
    sortedEllipses.clear();
    sortedEllipses.reserve(patSize.width * patSize.height);
    std::vector<int> indices;
    indices.reserve(patSize.width * patSize.height);
    for (int r = 0; r != patSize.height; ++r) {
        for (int c = 0; c != patSize.width; ++c) {
            cv::Point2d ptIdeal((double)(c) * stride, (double)(r) * stride);
            cv::Point2d ptA(0.0, (double)(r) * stride);
            cv::Point2d ptB((double)(patSize.width - 1) * stride, (double)(r) * stride);
            int idx = FindNearestPoint(rectifiedPoints, ptIdeal, ptA, ptB);
            
            sortedCenterPoints.push_back(centerPoints[idx]);
            sortedEllipses.push_back(ellipses[idx]);
            indices.push_back(idx);
        }
    }

    int N = patSize.width;
    cv::Point2d pt0 = rectifiedPoints[indices[0]], pt1 = rectifiedPoints[indices[1]],
                ptN = rectifiedPoints[indices[N - 1]];
    double dist01 = std::hypot(pt0.x - pt1.x, pt0.y - pt1.y);
    double dist0N = std::hypot(pt0.x - ptN.x, pt0.y - ptN.y);

    return ((dist0N > (double)(N - 2) * dist01) && (dist0N < 1.2 * (double)(N - 1) * dist01));
}

// *** DEPRECATED ***
// Use ScanlineClustering() instead.
//
static bool HierarchicalClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    std::vector<int> &patternPointIndices)
{
    assert(false && "deprecated");
    
/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

    int j, n = (int)points.size();
    size_t pn = static_cast<size_t>(patternSz.area());

    patternPointIndices.clear();
    patternPointIndices.reserve(pn);

    if (pn >= points.size())
    {
        if (pn == points.size()) {
            patternPointIndices.resize(pn);
            std::iota(patternPointIndices.begin(), patternPointIndices.end(), 0);
            return true;
        }
        else {
            return false;
        }
    }

    cv::Mat dists(n, n, CV_64FC1, cv::Scalar(0));
    cv::Mat distsMask(dists.size(), CV_8UC1, cv::Scalar(0));
    for(int i = 0; i < n; i++)
    {
        for(j = i+1; j < n; j++)
        {
            dists.at<double>(i, j) = (double)cv::norm(points[i] - points[j]);
            distsMask.at<uchar>(i, j) = 255;
            distsMask.at<uchar>(j, i) = 255;//distsMask.at<uchar>(i, j);
            dists.at<double>(j, i) = dists.at<double>(i, j);
        }
    }

    std::vector<std::list<size_t> > clusters(points.size());
    for(size_t i=0; i<points.size(); i++)
    {
        clusters[i].push_back(i);
    }

    int patternClusterIdx = 0;
    while(clusters[patternClusterIdx].size() < pn)
    {
        cv::Point minLoc;
        cv::minMaxLoc(dists, 0, 0, &minLoc, 0, distsMask);
        int minIdx = std::min(minLoc.x, minLoc.y);
        int maxIdx = std::max(minLoc.x, minLoc.y);

        distsMask.row(maxIdx).setTo(0);
        distsMask.col(maxIdx).setTo(0);
        cv::Mat tmpRow = dists.row(minIdx);
        cv::Mat tmpCol = dists.col(minIdx);
        cv::min(dists.row(minLoc.x), dists.row(minLoc.y), tmpRow);
        tmpRow = tmpRow.t();
        tmpRow.copyTo(tmpCol);

        clusters[minIdx].splice(clusters[minIdx].end(), clusters[maxIdx]);
        patternClusterIdx = minIdx;
    }

    if(clusters[patternClusterIdx].size() < static_cast<size_t>(patternSz.area()))
    {
        return false;
    }

    for(std::list<size_t>::iterator it = clusters[patternClusterIdx].begin(); 
        it != clusters[patternClusterIdx].end(); ++it)
    {
        patternPointIndices.push_back(*it);
    }

    return true;
}

/* *****************************************************************************/

#undef __DEBUG_SCANLINE_CLUSTERING__

#if (defined(NDEBUG) || !defined(_WIN32))
    #undef __DEBUG_SCANLINE_CLUSTERING__
#endif

#ifdef __DEBUG_SCANLINE_CLUSTERING__
    #include <opencv2/highgui.hpp>
#endif

static bool ScanlineClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    const cv::Size imageSize, double distThresh, std::vector<int> &patternPointIndices,
    std::vector<cv::Point2d>* pCornerPoints)
{
    const size_t N = points.size();
    const int rows = patternSz.height, cols = patternSz.width;
    const size_t M = rows * cols;
    const int dimLo = std::min(rows, cols), dimHi = std::max(rows, cols);
    
    // Compute centre of mass of the random ordered point set.
    double centreOfMass[2] = { 0, 0 };
    for (const cv::Point2d& pt : points) {
        centreOfMass[0] += pt.x;
        centreOfMass[1] += pt.y;
    }
    centreOfMass[0] /= (double)(N);
    centreOfMass[1] /= (double)(N);
    
    // Take the point nearest to centre of mass as origin. 
    int idxOrigin = FindNearestPoint(points, cv::Point2d(centreOfMass[0], centreOfMass[1]));
    const cv::Point2d ptOrigin = points[idxOrigin];
    
#ifdef __DEBUG_SCANLINE_CLUSTERING__
    cv::Mat img(imageSize, CV_8UC3, cv::Scalar::all(0));
    for (size_t i = 0; i != N; ++i) {
        cv::drawMarker(img, cv::Point(points[i]), cv::Scalar(0, 255, 0));
    }
    cv::imshow("Image", img);
    cv::waitKey(10);
#endif
    
    // Sort points by their distance from ptOrigin, in ascending order.
    std::vector<int> sortedPointIndices;        // index to `points`
    sortedPointIndices.reserve(N - 1);
    for (size_t i = 0; i != N; ++i) {
        if (i != idxOrigin) {
            sortedPointIndices.push_back(i);
        }
    }
    std::sort(sortedPointIndices.begin(), sortedPointIndices.end(), 
        [&points, ptOrigin] (int lhs, int rhs) -> bool {
            return std::hypot(points[lhs].x - ptOrigin.x, points[lhs].y - ptOrigin.y)
                   < std::hypot(points[rhs].x - ptOrigin.x, points[rhs].y - ptOrigin.y);
        }
    );
    
    // distThresh := 1/5 nearest distance.
    if (distThresh < 0) {
        distThresh = std::hypot(points[sortedPointIndices[0]].x - ptOrigin.x,
            points[sortedPointIndices[0]].y - ptOrigin.y) / 5.0;
    }
    
    // Compute histogram of the distance from other points to either of 4
    // nearest points.
    const size_t K = 4;
    int distHist[K];
    memset(distHist, 0, K * sizeof(int));
    std::vector<std::vector<int>> hintIndices;      // index to `points`
    hintIndices.resize(4);
    for (size_t k = 0; k != K; ++k) {
        cv::Point2d ptk = points[sortedPointIndices[k]];
        for (size_t j = 0; j != N - 1; ++j) {
            if (j == k) {
                continue;
            }
            
            double dist = PointLineDistance(points[sortedPointIndices[j]], ptOrigin, ptk);
            if (dist < distThresh) {
                ++distHist[k];
                hintIndices[k].push_back(sortedPointIndices[j]);
            }
        }
    }
    
    // Find a pair of points, where the angle between vectors
    // (ptj - ptOrigin) and (ptk - ptOrigin) is closest to 90 degrees == pi/2 radians.
    int bestPair[2] = {0, 1};   // index to `distHist` and `hintIndices`
    double minAngleDiff = 1.0e99;   
    for (size_t k = 0; k != K; ++k) {
        cv::Point2d ptk = points[sortedPointIndices[k]];
        for (size_t j = k + 1; j != K; ++j) {
            cv::Point2d ptj = points[sortedPointIndices[j]];
            
            double angle_j = std::atan2(ptj.y - ptOrigin.y, ptj.x - ptOrigin.x);
            double angle_k = std::atan2(ptk.y - ptOrigin.y, ptk.x - ptOrigin.x);
            double angle = angle_j - angle_k;
            
            // Wrap angle value into range [0, pi].
            int nWrap = angle / (2.0 * M_PI);
            angle -= (double)(nWrap) * (2.0 * M_PI);
            if (angle < 0) {
                angle += M_PI;
            }
            
            double angleDiff = std::abs(angle - 0.5 * M_PI);
            if (angleDiff < minAngleDiff) {
                minAngleDiff = angleDiff;
                bestPair[0] = k;
                bestPair[1] = j;
            }
        }
    }

    // Test if vector (points[bestPair[0]] - ptOrigin) is flatter than
    // vector (points[bestPair[1]] - ptOrigin).
    cv::Point2d pt0 = points[sortedPointIndices[bestPair[0]]], 
                pt1 = points[sortedPointIndices[bestPair[1]]];
    double angle0 = std::atan2(std::abs(pt0.y - ptOrigin.y), std::abs(pt0.x - ptOrigin.x));
    double angle1 = std::atan2(std::abs(pt1.y - ptOrigin.y), std::abs(pt1.x - ptOrigin.x));
    // swap them if not.
    if (angle0 > angle1) {
        std::swap(bestPair[0], bestPair[1]);
    }
    
    // Horizontal axis-points sorted in ascending order by X coordinates.
    std::vector<int>& horzAxispointIndices = hintIndices[bestPair[0]];
    horzAxispointIndices.push_back(idxOrigin);
    horzAxispointIndices.push_back(sortedPointIndices[bestPair[0]]);
    std::sort(horzAxispointIndices.begin(), horzAxispointIndices.end(),
        [&points](int lhs, int rhs) -> bool {
            return points[lhs].x < points[rhs].x;
        }
    );
    
    // Vertical axis-points sorted in ascending order by Y coordinates.
    std::vector<int>& vertAxispointIndices = hintIndices[bestPair[1]];
    vertAxispointIndices.push_back(idxOrigin);
    vertAxispointIndices.push_back(sortedPointIndices[bestPair[1]]);
    std::sort(vertAxispointIndices.begin(), vertAxispointIndices.end(),
        [&points](int lhs, int rhs) -> bool {
            return points[lhs].y < points[rhs].y;
        }
    );
    
#ifdef __DEBUG_SCANLINE_CLUSTERING__
    for (size_t i = 0; i != horzAxispointIndices.size(); ++i) {
        cv::drawMarker(img, cv::Point(points[horzAxispointIndices[i]]), cv::Scalar(0, 255, 0),
            cv::MARKER_STAR);
        cv::drawMarker(img, cv::Point(points[horzAxispointIndices[i]]), cv::Scalar(0, 255, 0),
            cv::MARKER_DIAMOND);
        cv::imshow("Image", img);
        cv::waitKey(10);
    }
    
    for (size_t i = 0; i != vertAxispointIndices.size(); ++i) {
        cv::drawMarker(img, cv::Point(points[vertAxispointIndices[i]]), cv::Scalar(0, 255, 0),
            cv::MARKER_STAR);
        cv::imshow("Image", img);
        cv::waitKey(10);
    }
#endif
    
    // Number of vertical axis-points above horizontal axis. Note that the 
    // positive Y axis always points downward in image coordinate system.
    // Use int type for variable `nAboveHorzBaseline` to avoid error prone comparison to 
    // unsigned values below.
    int nAboveHorzBaseline = 0;     
    for (size_t i = 0; i != vertAxispointIndices.size(); ++i) {
        if (vertAxispointIndices[i] != idxOrigin) {
            ++nAboveHorzBaseline;
        }
        else {
            break;
        }
    }

    // Scanlines jumping around horizontal axis. See figure below.
    //
    //                               +--- Order of scan
    //                               |
    //                               V
    // **********+***********        n
    //          ...                 ...
    // **********+***********        5
    // **********+***********        3
    // **********+***********        1
    // ++++++++++++++++++++++        0       <--- horizontal axis points
    // **********+***********        2
    // **********+***********        4
    // **********+***********       ...
    //          ...                 ...
    // **********+***********       ...
    //
    std::vector<int> scanSeq;
    scanSeq.reserve(vertAxispointIndices.size());
    scanSeq.push_back(nAboveHorzBaseline);
    for (int k = 1;; ++k) {
        bool quit = true;
        
        // try move downward
        if (nAboveHorzBaseline + k < vertAxispointIndices.size()) {
            scanSeq.push_back(nAboveHorzBaseline + k);
            quit = false;
        }
        
        // try move upward
        if (nAboveHorzBaseline - k >= 0) {
            scanSeq.push_back(nAboveHorzBaseline - k);
            quit = false;
        }
        
        // quit if index out of bound
        if (quit) {
            break;
        }
    }
    
    // Index was used if true.
    std::vector<bool> indexUsed;
    indexUsed.resize(N, false);
    for (int idx : horzAxispointIndices) {
        indexUsed[idx] = true;
    }
    for (int idx : vertAxispointIndices) {
        indexUsed[idx] = true;
    }

    
    // Vector `scanlines` stores sorted point indices of each scanline.
    std::vector<std::vector<int>> scanlines;
    scanlines.reserve(dimHi);
    scanlines.push_back(horzAxispointIndices);

    // Process each scanline.
    const size_t numScanlines = scanSeq.size();    
    for (size_t s = 1; s != numScanlines; ++s) {
        std::vector<int>& prevLine = scanlines[s - 1];
        
        int idxPrevBase = -1;      // `index to prevLine`
        for (size_t i = 0; i != prevLine.size(); ++i) {
            if (prevLine[i] == vertAxispointIndices[scanSeq[s - 1]]) {
                idxPrevBase = i;
                break;
            }
        }
        
        // Find from previous scanline the index of the point next to 
        // the base point of previous scanline either to the left or right
        // what so ever.
        int idxNextToPrevBase = -1;   // index to `points`
        if (idxPrevBase < 0) {
            return false;
        }
        else if (idxPrevBase + 1 < prevLine.size()) {
            idxNextToPrevBase = prevLine[idxPrevBase + 1];
        }
        else if (idxPrevBase - 1 >= 0) {
            idxNextToPrevBase = prevLine[idxPrevBase - 1];
        }
        else {
            return false;
        }
        
        // Index of the basepoint of current scanline.
        int idxCurBase = vertAxispointIndices[scanSeq[s]];
        
        cv::Point2d ptPrevBase = points[prevLine[idxPrevBase]];
        cv::Point2d ptNextToPrevBase = points[idxNextToPrevBase];
        cv::Point2d ptCurBase = points[idxCurBase];
        
        // Find second point for current scanline. See figure below.
        //             
        //                  
        //                  
        //        Current scanline L2 in parallel to L1          
        //                        \
        //                         \
        // ptCurBase ---->  *................*  <--- ptSecond in search of
        //                   \       *                  := Point closest to L2.
        //                    \  * 
        //                     \
        //                      \
        // ptPrevBase ------->   *-------------*   <---- ptNextToPrevBase     
        //                               \
        //                                \
        //                               Previous scanline L1
        //   
        //
        //
        double lineParallel[2] = {
            ptNextToPrevBase.x - ptPrevBase.x,
            ptNextToPrevBase.y - ptPrevBase.y
        };
        double minDist = 1.0e99;
        int idxNearest = -1;
        for (size_t i = 0; i != N; ++i) {
            if (indexUsed[i]) {
                continue;
            }
            
            cv::Point ptCur = points[i];
            double x0 = ptCur.x - ptCurBase.x, y0 = ptCur.y - ptCurBase.y;

            // Omitted dividing the cross product result by vector length of lineParallel,
            // since the scaling is unimportant.
            double dist = std::abs(x0 * lineParallel[1] - y0 * lineParallel[0]);
            
            if (dist < minDist) {
                minDist = dist;
                idxNearest = i;
            }
        }
        
        if (idxNearest < 0) {
            return false;
        }
        
        cv::Point2d ptSecond = points[idxNearest];
        indexUsed[idxNearest] = true;
        
        // Find remaining points of current scanline.
        std::vector<int> curLine;
        curLine.reserve(dimHi);
        curLine.push_back(idxCurBase);
        curLine.push_back(idxNearest);
        for (size_t i = 0; i != N; ++i) {
            if (indexUsed[i]) {
                continue;
            }
            
            double dist = PointLineDistance(points[i], ptSecond, ptCurBase);
            if (dist < distThresh) {
                curLine.push_back(i);
                indexUsed[i] = true;
            }
        }
        
        // Sort them in ascending order by X coords.
        std::sort(curLine.begin(), curLine.end(),
            [&points](int lhs, int rhs) -> bool {
                return points[lhs].x < points[rhs].x;
            }
        );
        
#ifdef __DEBUG_SCANLINE_CLUSTERING__
        for (size_t i = 0; i != curLine.size(); ++i) {
            cv::drawMarker(img, cv::Point(points[curLine[i]]), cv::Scalar(0, 255, 0),
                cv::MARKER_DIAMOND);
            cv::imshow("Image", img);
            cv::waitKey(10);
        }
#endif

        // Check missing points or extreme perspective deform.
        double maxDist = -1.0, minDist1 = 1.0e99, meanDist = 0.0;
        for (size_t i = 1; i != curLine.size(); ++i) {
            const cv::Point& ptPrev = points[curLine[i - 1]];
            const cv::Point& ptCur = points[curLine[i]];
            double dist = std::hypot(ptPrev.x - ptCur.x, ptPrev.y - ptCur.y);
            if (dist > maxDist) {
                maxDist = dist;
            }
            if (dist < minDist1) {
                minDist1 = dist;
            }
            meanDist += dist;
        }
        meanDist /= (double)(curLine.size());
        if ((maxDist > 1.5 * meanDist) || (minDist1 < 0.75 * meanDist)) {
            return false;
        }
        
        scanlines.push_back(std::move(curLine));
    }
    
    // Sort in ascending order by Y coordinate.
    std::vector<std::vector<int>> sortedScanlines;
    sortedScanlines.resize(numScanlines);
    for (size_t s = 0; s != numScanlines; ++s) {
        sortedScanlines[scanSeq[s]].swap(scanlines[s]);
    }
    
    // Find candidate scanlines.
    //
    // `scanlineBasepointIndices` stores index to `sortedScanlines`. 
    // Positive `scanlineBasepointIndices` element if corresponding ordered-scanline is a candidate.
    std::vector<int> scanlineBasepointIndices(numScanlines, -1); // --> sortedScanlines
    // Used by next step.
    // Number of points to the left of basepoint: 0, 2, 4, ...; 
    // Nubmer of points to the right of basepoint: 1, 3, 5, ...; 
    std::vector<int> scanlinePointsCount(2 * numScanlines, 0);
    for (size_t s = 0; s != numScanlines; ++s) {
        const std::vector<int>& curLine = sortedScanlines[s];
        int& idxCurBase = scanlineBasepointIndices[s];
        
        if (curLine.size() < dimLo) {
            continue;
        }
        
        for (size_t j = 0; j != curLine.size(); ++j) {
            if (vertAxispointIndices[s] == curLine[j]) {
                idxCurBase = j;
                break;
            }
        }
        
        size_t n = 1;
        for (int j = 1; n <= dimHi; ++j) {
            bool quit = true;
            
            // move toward positive direction along horizontal axis
            if (idxCurBase + j < curLine.size()) {
                ++scanlinePointsCount[2 * s + 1];
                ++n;
                quit = false;
            }
            
            // move toward negative direction along horizontal axis
            if (idxCurBase - j >= 0) {
                ++scanlinePointsCount[2 * s];
                ++n;
                quit = false;
            }
            
            // if index out of bound
            if (quit) {
                break;
            }
        }
        
        if (n < dimLo) {
            idxCurBase = -1;
        }
    }

    // Locate rect grid points centered at `ptOrigin`, as far as possible.
    auto fScan = 
        [&scanlinePointsCount, numScanlines] 
        (int dimA, int dimB, int& idxFirstScanline, int& nLeft) -> bool
    {
        const int s0 = numScanlines / 2, ds = dimB / 2;
        bool found = false;

        for (int j = 0; true; ++j) {
            int quit = 0;

            for (int q = 0; q <= 1; ++q) {
                int sj = (q == 0) ? s0 - ds - j : s0 - ds + j;
                if ((sj >= 0) && (sj + dimB <= numScanlines)) {
                    // both initialised as a BIGGGGggg number thanks to two's complement magic.
                    size_t minNumLeft = -1, minNumRight = -1; 
                    for (int k = sj; k < sj + dimB; ++k) {
                        size_t n;

                        n = scanlinePointsCount[2 * k];
                        if (n < minNumLeft) {
                            minNumLeft = n;
                        }

                        n = scanlinePointsCount[2 * k + 1];
                        if (n < minNumRight) {
                            minNumRight = n; 
                        }
                    }

                    found = (1 + (int)(minNumLeft) + (int)(minNumRight) >= dimA);

                    if (found) {
                        idxFirstScanline = sj;
                        nLeft = std::min((int)minNumLeft, (dimA - 1) / 2);
                        if (dimA - 1 - nLeft > minNumRight) {
                            nLeft = minNumLeft;
                        }
                        quit = 2;
                    }
                }
                else {
                    ++quit;
                }

                if (found || (/* dont't scan the same range twice */ j == 0)) {
                    break;
                }
            }

            if (quit >= 2) {
                break;
            }
        }

        return found;
    };
    
    int idxFirstScanline = -1, numLeft = -1;
    bool found1 = false, found2 = false;
    found1 = fScan(cols, rows, idxFirstScanline, numLeft);
    if (!found1) {
        found2 = fScan(rows, cols, idxFirstScanline, numLeft);
    }
    if (!(found1 || found2)) {
        return false;
    }
    
    patternPointIndices.clear();
    patternPointIndices.reserve(M);

    const int s0 = found1 ? (idxFirstScanline + rows - 1) : (idxFirstScanline + cols - 1);
    const int dim0 = found1 ? cols : rows;
    for (int s = s0; s >= idxFirstScanline; --s) {
        int idxCurBase = scanlineBasepointIndices[s];
        std::vector<int>& curScanline = sortedScanlines[s];
        for (int i = 0; i < dim0; ++i) {
            patternPointIndices.push_back(curScanline[idxCurBase - numLeft + i]);
        }
    }
    
#ifdef __DEBUG_SCANLINE_CLUSTERING__
        for (int i : patternPointIndices) {
            cv::drawMarker(img, cv::Point(points[i]), cv::Scalar(0, 255, 0),
                cv::MARKER_SQUARE);
            cv::imshow("Image", img);
            cv::waitKey(10);
        }
#endif

    if (pCornerPoints) {
        pCornerPoints->clear();
        pCornerPoints->reserve(4);

        if (found1) {
            //
            //   4------3   -
            //   |      |   | rows
            //   1------2   -
            //
            //   |------|
            //     cols
            //
            pCornerPoints->push_back(points[patternPointIndices[0]]);
            pCornerPoints->push_back(points[patternPointIndices[cols - 1]]);
            pCornerPoints->push_back(points[patternPointIndices[rows * cols - 1]]);
            pCornerPoints->push_back(points[patternPointIndices[(rows - 1) * cols]]);
        }
        else {
            //
            //   1------4  -
            //   |      |  | cols
            //   2------3  -
            //
            //   |------|
            //     rows
            //
            pCornerPoints->push_back(points[patternPointIndices[(cols - 1) * rows]]);
            pCornerPoints->push_back(points[patternPointIndices[0]]);
            pCornerPoints->push_back(points[patternPointIndices[rows - 1]]);
            pCornerPoints->push_back(points[patternPointIndices[rows * cols - 1]]);
        }
    }
    
    return true;
}

/* *****************************************************************************/

static bool ExtractContoursHalconCalibBoard(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered,
        bool inverseThresh)
{
    cv::Mat imgMono;
    if (thresh < 0) {
        cv::threshold(imgGray, imgMono, -1, 255, 
            (inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) + cv::THRESH_OTSU);
    }
    else {
        cv::threshold(imgGray, imgMono, thresh, 255, 
            inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    }

    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgMono, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    const int numContours = contours.size();

    // Find inner and outer contours.
    int idxInner = -1;
    for (int i = 0; i != numContours; ++i) {
        // Count number of child contours.
        int n = 0;
        for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0], ++n);

        // Inner contour should contain `total` circles and must have
        // a valid parent contour.
        if ((n < total) || (n > total + 5) || (hierarchy[i][3] < 0)) {
            continue;
        }

        // Try approximate parent contour.
        // The outer contour must be a quadrilateral.
        std::vector<cv::Point> outerContourApprox;
        double eps = cv::arcLength(contours[hierarchy[i][3]], true) / 32.0;
        cv::approxPolyDP(contours[hierarchy[i][3]], outerContourApprox, eps, true);

        std::vector<int> outerCornerIndices;
        if (!FindFourCorners(outerContourApprox, outerCornerIndices)) {
            continue;
        }
        else {
            outerContour.clear();
            outerContour.reserve(4);
            for (int idx : outerCornerIndices) {
                outerContour.push_back(outerContourApprox[idx]);
            }
        }

        // Try approximate current contour.
        std::vector<cv::Point> approxPointsInner;
        cv::approxPolyDP(contours[i], approxPointsInner, 3, true);

        // The inner border of a Halcon calibration board pattern is a rectangle
        // with one of its corner chamfered hence ideal total number of vertices is 5.
        if (approxPointsInner.size() < 5) {
            continue;
        }
        else {
            idxInner = i;
            innerContour = std::move(approxPointsInner);
            break;
        }
    }

    if (idxInner < 0) {
        return false;
    }

    // Compute blob areas.
    std::vector<int> blobIndices;
    std::vector<double> blobAreas;
    blobIndices.reserve(total);
    blobAreas.reserve(total);
    for (int i = 0; i != numContours; ++i) {
        if (hierarchy[i][3] != idxInner) {
            continue;
        }
        
        const std::vector<cv::Point>& contour = contours[i];
        
        cv::Moments moments = cv::moments(contour);
        double area = moments.m00;
        
        // Filter by area
        {
            const double areaThresh = imgGray.rows * imgGray.cols / total;
            if (area > areaThresh) {
                continue;
            }
        }
            
        // Filter by circularity
        {
            double perim = cv::arcLength(contour, true);
            if (4.0 * M_PI * area / (perim * perim + 1.0) < 0.75) {
                continue;
            }
        }
        
        // Filter by convexity
        {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            double hullArea = cv::contourArea(hull);
            if (area / (hullArea + 1.0) < 0.9) {
                continue;
            }
        }
        
        if (contour.size() >= 9) {
            blobIndices.push_back(i);
            blobAreas.push_back(cv::contourArea(contour));
        }
    }

    // Compute mean and standard deviation of blob areas.
    const int numBlobs = blobIndices.size();
    double meanArea = 0.0f;
    for (double area : blobAreas) {
        meanArea += area;
    }
    meanArea = meanArea / (double)(numBlobs);

    double stddevArea = 0.0f;
    for (double area : blobAreas) {
        stddevArea += std::pow(area - meanArea, 2);
    }
    stddevArea = std::sqrt(stddevArea / (double)(numBlobs));

    // Filter blobs by area using 3-sigma rule of thumb.
    double area1 = meanArea - std::max(0.25f * meanArea, 3.0f * stddevArea), 
          area2 = meanArea + std::max(0.25f * meanArea, 3.0f * stddevArea);
    blobIndicesFiltered.clear();
    blobIndicesFiltered.reserve(numBlobs);
    for (int i = 0; i != numBlobs; ++i) {
        int idx = blobIndices[i];
        double area = blobAreas[i];
        if ((area >= area1) && (area <= area2)) {
            blobIndicesFiltered.push_back(idx);
        }
    }

    return (blobIndicesFiltered.size() == total);
}
