#ifndef DUALMASKBLENDER_H
#define DUALMASKBLENDER_H

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief Multi-band blender with separate weight and blend masks
 * 
 * This class extends the concept of OpenCV's MultiBandBlender to support
 * two separate masks:
 * - weight_mask: Used to compute weight pyramids (e.g., PC_ masks with large feathering)
 * - blend_mask: Used to blend pixels in Laplacian pyramids (e.g., Voronoi masks with sharp edges)
 * 
 * This separation allows combining the advantages of both mask types:
 * - Smooth, homogeneous blending from large weight masks
 * - No ghosting artifacts from sharp blend masks
 */
class DualMaskMultiBandBlender {
public:
    /**
     * @brief Constructor
     * @param num_bands Number of bands in the multi-band pyramid (default: 5)
     * @param weight_type Data type for weights: CV_32F or CV_16S (default: CV_32F)
     */
    explicit DualMaskMultiBandBlender(int num_bands = 5, int weight_type = CV_32F);

    /**
     * @brief Prepares the blender for the given destination ROI
     * @param dst_roi Destination region of interest (full canvas size)
     */
    void prepare(cv::Rect dst_roi);

    /**
     * @brief Feeds an image with two separate masks into the blender
     * @param img Input image (CV_16SC3 or CV_8UC3)
     * @param weight_mask Mask for computing weights (CV_8U, 0-255)
     * @param blend_mask Mask for blending pixels (CV_8U, 0-255)
     * @param tl Top-left corner of the image in canvas coordinates
     */
    void feed(cv::InputArray img, cv::InputArray weight_mask, cv::InputArray blend_mask, cv::Point tl);

    /**
     * @brief Blends all fed images and produces the final result
     * @param dst Output blended image
     * @param dst_mask Output mask indicating valid pixels
     */
    void blend(cv::OutputArray dst, cv::OutputArray dst_mask);

    /**
     * @brief Sets the number of bands
     * @param num_bands Number of bands (1-50)
     */
    void setNumBands(int num_bands);

    /**
     * @brief Gets the current number of bands
     * @return Number of bands
     */
    int numBands() const { return actual_num_bands_; }

private:
    int actual_num_bands_;  // User-specified number of bands
    int num_bands_;         // Actual number of bands used (may be less due to image size)
    int weight_type_;       // CV_32F or CV_16S
    
    cv::Rect dst_roi_;      // Current ROI (padded to be divisible by 2^num_bands)
    cv::Rect dst_roi_final_;// Final ROI (user-requested)
    
    cv::UMat dst_;          // Base destination image
    cv::UMat dst_mask_;     // Base destination mask
    
    std::vector<cv::UMat> dst_pyr_laplace_;    // Destination Laplacian pyramid
    std::vector<cv::UMat> dst_band_weights_;   // Accumulated weights for each band
};

#endif // DUALMASKBLENDER_H
