#include "dualmaskblender.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <cmath>
#include <algorithm>

namespace {
static const float WEIGHT_EPS = 1e-5f;

// Helper function to create Laplacian pyramid
void createLaplacePyr(cv::InputArray img, int num_levels, std::vector<cv::UMat> &pyr) {
    pyr.resize(num_levels + 1);

    if (img.depth() == CV_8U) {
        if (num_levels == 0) {
            img.getUMat().convertTo(pyr[0], CV_16S);
            return;
        }

        cv::UMat downNext;
        cv::UMat current = img.getUMat();
        cv::pyrDown(img, downNext);

        for (int i = 1; i < num_levels; ++i) {
            cv::UMat lvl_up;
            cv::UMat lvl_down;

            cv::pyrDown(downNext, lvl_down);
            cv::pyrUp(downNext, lvl_up, current.size());
            cv::subtract(current, lvl_up, pyr[i-1], cv::noArray(), CV_16S);

            current = downNext;
            downNext = lvl_down;
        }

        {
            cv::UMat lvl_up;
            cv::pyrUp(downNext, lvl_up, current.size());
            cv::subtract(current, lvl_up, pyr[num_levels-1], cv::noArray(), CV_16S);

            downNext.convertTo(pyr[num_levels], CV_16S);
        }
    } else {
        pyr[0] = img.getUMat();
        for (int i = 0; i < num_levels; ++i)
            cv::pyrDown(pyr[i], pyr[i + 1]);
        cv::UMat tmp;
        for (int i = 0; i < num_levels; ++i) {
            cv::pyrUp(pyr[i + 1], tmp, pyr[i].size());
            cv::subtract(pyr[i], tmp, pyr[i]);
        }
    }
}

// Helper function to restore image from Laplacian pyramid
void restoreImageFromLaplacePyr(std::vector<cv::UMat> &pyr) {
    if (pyr.empty())
        return;
    cv::UMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i) {
        cv::pyrUp(pyr[i], tmp, pyr[i - 1].size());
        cv::add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}

// Helper function to normalize using weight map
void normalizeUsingWeightMap(cv::InputArray _weight, cv::InputOutputArray _src) {
    cv::Mat src = _src.getMat();
    cv::Mat weight = _weight.getMat();

    CV_Assert(src.type() == CV_16SC3);

    if (weight.type() == CV_32FC1) {
        for (int y = 0; y < src.rows; ++y) {
            cv::Point3_<short> *row = src.ptr<cv::Point3_<short>>(y);
            const float *weight_row = weight.ptr<float>(y);

            for (int x = 0; x < src.cols; ++x) {
                row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
                row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
                row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
            }
        }
    } else {
        CV_Assert(weight.type() == CV_16SC1);

        for (int y = 0; y < src.rows; ++y) {
            const short *weight_row = weight.ptr<short>(y);
            cv::Point3_<short> *row = src.ptr<cv::Point3_<short>>(y);

            for (int x = 0; x < src.cols; ++x) {
                int w = weight_row[x] + 1;
                row[x].x = static_cast<short>((row[x].x << 8) / w);
                row[x].y = static_cast<short>((row[x].y << 8) / w);
                row[x].z = static_cast<short>((row[x].z << 8) / w);
            }
        }
    }
}

} // anonymous namespace


DualMaskMultiBandBlender::DualMaskMultiBandBlender(int num_bands, int weight_type)
    : actual_num_bands_(0), num_bands_(0), weight_type_(weight_type) {
    CV_Assert(weight_type == CV_32F || weight_type == CV_16S);
    setNumBands(num_bands);
}

void DualMaskMultiBandBlender::setNumBands(int num_bands) {
    CV_Assert(num_bands >= 1 && num_bands <= 50);
    actual_num_bands_ = num_bands;
}

void DualMaskMultiBandBlender::prepare(cv::Rect dst_roi) {
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
    num_bands_ = std::min(actual_num_bands_, static_cast<int>(std::ceil(std::log(max_len) / std::log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    // Prepare base destination
    dst_roi_ = dst_roi;
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(cv::Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(cv::Scalar::all(0));

    // Prepare pyramids
    dst_pyr_laplace_.clear();
    dst_band_weights_.clear();

    dst_pyr_laplace_.resize(num_bands_ + 1);
    dst_pyr_laplace_[0] = dst_;

    dst_band_weights_.resize(num_bands_ + 1);
    dst_band_weights_[0].create(dst_roi.size(), weight_type_);
    dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i) {
        dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
            (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
        dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
            (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
        dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));
        dst_band_weights_[i].setTo(0);
    }
}

void DualMaskMultiBandBlender::feed(cv::InputArray _img, cv::InputArray _weight_mask, 
                                     cv::InputArray _blend_mask, cv::Point tl) {
    cv::UMat img = _img.getUMat();

    CV_Assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    CV_Assert(_weight_mask.type() == CV_8U);
    CV_Assert(_blend_mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    cv::Point tl_new(std::max(dst_roi_.x, tl.x - gap),
                     std::max(dst_roi_.y, tl.y - gap));
    cv::Point br_new(std::min(dst_roi_.br().x, tl.x + img.cols + gap),
                     std::min(dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_)
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = std::max(br_new.y - dst_roi_.br().y, 0);
    int dx = std::max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    cv::UMat img_with_border;
    cv::copyMakeBorder(_img, img_with_border, top, bottom, left, right, cv::BORDER_REFLECT);

    std::vector<cv::UMat> src_pyr_laplace;
    createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);

    // Create the weight map Gaussian pyramid (from weight_mask)
    cv::UMat weight_map;
    std::vector<cv::UMat> weight_pyr_gauss(num_bands_ + 1);

    if (weight_type_ == CV_32F) {
        _weight_mask.getUMat().convertTo(weight_map, CV_32F, 1./255.);
    } else {
        _weight_mask.getUMat().convertTo(weight_map, CV_16S);
        cv::UMat add_mask;
        cv::compare(_weight_mask, 0, add_mask, cv::CMP_NE);
        cv::add(weight_map, cv::Scalar::all(1), weight_map, add_mask);
    }

    cv::copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, cv::BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        cv::pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    // Create the blend map Gaussian pyramid (from blend_mask)
    cv::UMat blend_map;
    std::vector<cv::UMat> blend_pyr_gauss(num_bands_ + 1);

    if (weight_type_ == CV_32F) {
        _blend_mask.getUMat().convertTo(blend_map, CV_32F, 1./255.);
    } else {
        _blend_mask.getUMat().convertTo(blend_map, CV_16S);
        cv::UMat add_mask;
        cv::compare(_blend_mask, 0, add_mask, cv::CMP_NE);
        cv::add(blend_map, cv::Scalar::all(1), blend_map, add_mask);
    }

    cv::copyMakeBorder(blend_map, blend_pyr_gauss[0], top, bottom, left, right, cv::BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        cv::pyrDown(blend_pyr_gauss[i], blend_pyr_gauss[i + 1]);

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    // Key difference: use blend_mask for pixel blending, weight_mask for accumulation
    for (int i = 0; i <= num_bands_; ++i) {
        cv::Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
        
        cv::Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(cv::ACCESS_READ);
        cv::Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc).getMat(cv::ACCESS_RW);
        cv::Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(cv::ACCESS_READ);
        cv::Mat _blend_pyr_gauss = blend_pyr_gauss[i].getMat(cv::ACCESS_READ);
        cv::Mat _dst_band_weights = dst_band_weights_[i](rc).getMat(cv::ACCESS_RW);
        
        if (weight_type_ == CV_32F) {
            for (int y = 0; y < rc.height; ++y) {
                const cv::Point3_<short>* src_row = _src_pyr_laplace.ptr<cv::Point3_<short>>(y);
                cv::Point3_<short>* dst_row = _dst_pyr_laplace.ptr<cv::Point3_<short>>(y);
                const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                const float* blend_row = _blend_pyr_gauss.ptr<float>(y);
                float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                for (int x = 0; x < rc.width; ++x) {
                    // Blend pixels using blend_mask
                    dst_row[x].x += static_cast<short>(src_row[x].x * blend_row[x]);
                    dst_row[x].y += static_cast<short>(src_row[x].y * blend_row[x]);
                    dst_row[x].z += static_cast<short>(src_row[x].z * blend_row[x]);
                    // Accumulate weights using weight_mask
                    dst_weight_row[x] += weight_row[x];
                }
            }
        } else {
            for (int y = 0; y < rc.height; ++y) {
                const cv::Point3_<short>* src_row = _src_pyr_laplace.ptr<cv::Point3_<short>>(y);
                cv::Point3_<short>* dst_row = _dst_pyr_laplace.ptr<cv::Point3_<short>>(y);
                const short* weight_row = _weight_pyr_gauss.ptr<short>(y);
                const short* blend_row = _blend_pyr_gauss.ptr<short>(y);
                short* dst_weight_row = _dst_band_weights.ptr<short>(y);

                for (int x = 0; x < rc.width; ++x) {
                    // Blend pixels using blend_mask
                    dst_row[x].x += short((src_row[x].x * blend_row[x]) >> 8);
                    dst_row[x].y += short((src_row[x].y * blend_row[x]) >> 8);
                    dst_row[x].z += short((src_row[x].z * blend_row[x]) >> 8);
                    // Accumulate weights using weight_mask
                    dst_weight_row[x] += weight_row[x];
                }
            }
        }

        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }
}

void DualMaskMultiBandBlender::blend(cv::OutputArray dst, cv::OutputArray dst_mask) {
    cv::Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height);
    cv::UMat dst_band_weights_0;

    for (int i = 0; i <= num_bands_; ++i)
        normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);

    restoreImageFromLaplacePyr(dst_pyr_laplace_);

    dst_ = dst_pyr_laplace_[0](dst_rc);
    dst_band_weights_0 = dst_band_weights_[0];

    dst_pyr_laplace_.clear();
    dst_band_weights_.clear();

    cv::compare(dst_band_weights_0(dst_rc), WEIGHT_EPS, dst_mask_, cv::CMP_GT);

    // Final blend
    cv::UMat mask;
    cv::compare(dst_mask_, 0, mask, cv::CMP_EQ);
    dst_.setTo(cv::Scalar::all(0), mask);
    dst.assign(dst_);
    dst_mask.assign(dst_mask_);
    dst_.release();
    dst_mask_.release();
}
