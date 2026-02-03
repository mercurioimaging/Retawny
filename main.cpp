#include "ortholoader.h"

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <iostream>
#include <sys/resource.h>

using namespace std;

QImage bgrMatToQImage(const cv::Mat &bgr);
cv::Mat qImageToBgrMat(const QImage &source);
cv::Mat buildCoverageMask(const QImage &source, const QImage &loadedMask = QImage(), bool sharp = false);


int main(int argc, char *argv[]) {
	if (argc < 3 || argc > 4) {
		cerr << "Usage: " << argv[0] << " <input_folder> <output.jpg> [num_bands]" << endl;
		cerr << "  <input_folder>: Folder containing TIFF files" << endl;
		cerr << "  <output.jpg>: Output JPEG image path" << endl;
		cerr << "  [num_bands]: Optional number of bands for MultiBandBlender (default: 14)" << endl;
		return 1;
	}

	QString folder = QString::fromUtf8(argv[1]);
	QString outputPath = QString::fromUtf8(argv[2]);
	
	int numBands = 14; // Default value
	if (argc == 4) {
		bool ok = false;
		numBands = QString::fromUtf8(argv[3]).toInt(&ok);
		if (!ok || numBands < 0 || numBands > 50) {
			cerr << "Invalid num_bands value. Must be between 0 and 50." << endl;
			return 1;
		}
	}



	OrthoLoader loader;

	QString errorMessage;
	if(!loader.loadFromDirectory(folder, &errorMessage)) {
		cerr << "Loading failed: " << qPrintable(errorMessage) << endl;
		return 1;
	}

	QVector<OrthoLoader::Tile> &tiles = loader.tiles();
	if (tiles.size() < 2) {
		cerr << "Not enough tiles: Need at least two tiles to blend." << endl;
		return 1;
	}

	const QSize canvasSize = loader.canvasSize();
	if (!canvasSize.isValid() || canvasSize.isEmpty()) {
		cerr << "Invalid canvas: Cannot blend because the canvas size is invalid." << endl;
		return 1;
	}

	// Check for large images and disable OpenCL to avoid VRAM exhaustion
	const long long totalPixels = static_cast<long long>(canvasSize.width()) * canvasSize.height();
	const long long estimatedMB = (totalPixels * 3) / (1024 * 1024); // 3 bytes per pixel for BGR
	if (estimatedMB > 1000) {
		cout << "Large image detected (" << canvasSize.width() << "x" << canvasSize.height() 
		     << ", ~" << estimatedMB << " MB)" << endl;
		cout << "Disabling OpenCL to avoid VRAM exhaustion, using CPU instead..." << endl;
		cv::ocl::setUseOpenCL(false);
	}

	const cv::Rect roi(0, 0, canvasSize.width(), canvasSize.height());
	cv::detail::MultiBandBlender blender(false, numBands);
	blender.prepare(roi);

	bool fedAny = false;
	for (OrthoLoader::Tile &tile : tiles) {
		// Load tile into memory
		if (!loader.loadTile(&tile, &errorMessage)) {
			cerr << "Failed to load tile: " << qPrintable(errorMessage) << endl;
			continue;
		}
		if (tile.image.isNull())
			continue;

		cv::Mat bgr = qImageToBgrMat(tile.image);
		if (bgr.empty())
			continue;

		// Load mask if available
		loader.loadMask(&tile, nullptr);
		
		// Build coverage mask with feathering (using loaded mask if available)
		cv::Mat mask = buildCoverageMask(tile.image, tile.mask, false);
		
		// Unload mask to free memory
		loader.unloadMask(&tile);
		
		if (mask.empty() || cv::countNonZero(mask) == 0) {
			loader.unloadTile(&tile);
			continue;
		}

		cv::Scalar avgColor = cv::mean(bgr, mask);

		// Fill only pixels where mask is zero with average color
		cv::Mat zeroMask = (mask == 0);
		bgr.setTo(avgColor, zeroMask);

		cv::Mat img16s;
		bgr.convertTo(img16s, CV_16SC3);
		blender.feed(img16s, mask, cv::Point(tile.x, tile.y));
		fedAny = true;
		
		// Unload tile to free memory
		loader.unloadTile(&tile);
	}

	if (!fedAny) {
		cerr << "Blending failed: No valid pixels were submitted to the blender." << endl;
		return 1;
	}

	cv::Mat blended, blendedMask;
	blender.blend(blended, blendedMask);
	if (blended.empty()) {
		cerr << "Blending failed: OpenCV returned an empty result." << endl;
		return 1;
	}

	cv::Mat blended8u;
	blended.convertTo(blended8u, CV_8UC3);
	QImage blendedImage = bgrMatToQImage(blended8u);
	if (blendedImage.isNull()) {
		cerr << "Blending failed: Could not convert the blended image for display." << endl;
		return 1;
	}

	// Save the output image
	if (!blendedImage.save(outputPath)) {
		cerr << "Failed to save output image to: " << qPrintable(outputPath) << endl;
		return 1;
	}
	
	// Report peak memory usage
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) == 0) {
		const long peakMB = usage.ru_maxrss / (1024 * 1024);
		cout << "Peak memory usage: " << peakMB << " MB" << endl;
	}
	
	return 0;
}



cv::Mat qImageToBgrMat(const QImage &source) {
	if (source.isNull())
		return cv::Mat();

	QImage converted = source.convertToFormat(QImage::Format_ARGB32);
	cv::Mat wrapper(converted.height(), converted.width(), CV_8UC4, const_cast<uchar *>(converted.bits()), converted.bytesPerLine());
	cv::Mat bgr;
	cv::cvtColor(wrapper, bgr, cv::COLOR_BGRA2BGR);
	return bgr.clone();
}


QImage bgrMatToQImage(const cv::Mat &bgr) {
	if (bgr.empty() || bgr.type() != CV_8UC3)
		return QImage();

	cv::Mat rgb;
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
	QImage image(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
	return image.copy();
}

cv::Mat buildCoverageMask(const QImage &source, const QImage &loadedMask, bool sharp) {
	if (source.isNull())
		return cv::Mat();

	cv::Mat mask;
	
	// Use loaded mask if available (black = usable, white = masked)
	if (!loadedMask.isNull()) {
		QImage maskConverted = loadedMask.convertToFormat(QImage::Format_ARGB32);
		mask = cv::Mat(maskConverted.height(), maskConverted.width(), CV_8UC1, cv::Scalar(0));
		for (int y = 0; y < maskConverted.height(); ++y) {
			const QRgb *scan = reinterpret_cast<const QRgb *>(maskConverted.constScanLine(y));
			uchar *maskRow = mask.ptr<uchar>(y);
			for (int x = 0; x < maskConverted.width(); ++x) {
				const QRgb pixel = scan[x];
				int gray = qGray(pixel);
				// Black (low values) = usable -> 255, White (high values) = masked -> 0
				maskRow[x] = (gray < 128) ? 255 : 0;
			}
		}
	} else {
		// Fallback: detect magenta pixels in source image
		QImage converted = source.convertToFormat(QImage::Format_ARGB32);
		mask = cv::Mat(converted.height(), converted.width(), CV_8UC1, cv::Scalar(0));
		for (int y = 0; y < converted.height(); ++y) {
			const QRgb *scan = reinterpret_cast<const QRgb *>(converted.constScanLine(y));
			uchar *maskRow = mask.ptr<uchar>(y);
			for (int x = 0; x < converted.width(); ++x) {
				const QRgb pixel = scan[x];
				maskRow[x] = (qRed(pixel) == 255 && qGreen(pixel) == 0 && qBlue(pixel) == 255) ? 0 : 255;
			}
		}
	}
	constexpr double kMaskFeatherRadiusPixels = 512.0;

	if (sharp || kMaskFeatherRadiusPixels <= 1.0)
		return mask;

	// Distance transform from masked regions (magenta pixels)
	cv::Mat distanceFromMask;
	cv::distanceTransform(mask, distanceFromMask, cv::DIST_L2, 3);

	// Distance transform from image borders
	cv::Mat borderMask(mask.rows, mask.cols, CV_8UC1, cv::Scalar(255));
	// Set border pixels to 0
	borderMask.row(0).setTo(0);
	borderMask.row(borderMask.rows - 1).setTo(0);
	borderMask.col(0).setTo(0);
	borderMask.col(borderMask.cols - 1).setTo(0);

	cv::Mat distanceFromBorder;
	cv::distanceTransform(borderMask, distanceFromBorder, cv::DIST_L2, 3);

	// Combine: minimum distance to either masked region or border
	cv::Mat combinedDistance;
	cv::min(distanceFromMask, distanceFromBorder, combinedDistance);

	// Normalize by feather radius
	combinedDistance /= kMaskFeatherRadiusPixels;
	cv::min(combinedDistance, 1.0, combinedDistance);

	// Convert to 8-bit mask
	cv::Mat feathered;
	combinedDistance.convertTo(feathered, CV_8UC1, 255.0);

	// Zero out original masked pixels
	feathered.setTo(0, mask == 0);

	return feathered;
}
