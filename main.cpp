#include "ortholoader.h"

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <iostream>
#include <sys/resource.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

QImage bgrMatToQImage(const cv::Mat &bgr);
cv::Mat qImageToBgrMat(const QImage &source);
cv::Mat buildCoverageMask(const QImage &source, const QImage &loadedMask = QImage(), double featherRadius = 512.0, bool sharp = false);


int main(int argc, char *argv[]) {
	if (argc < 3 || argc > 5) {
		cerr << "Usage: " << argv[0] << " <input_folder> <output.png> [num_bands] [feather_radius]" << endl;
		cerr << "  <input_folder>: Folder containing TIFF files" << endl;
		cerr << "  <output.png>: Output PNG image path" << endl;
		cerr << "  [num_bands]: Optional number of bands for MultiBandBlender (default: 14)" << endl;
		cerr << "  [feather_radius]: Optional feathering radius in pixels (default: 512.0)" << endl;
		return 1;
	}

	auto startTime = high_resolution_clock::now();

	QString folder = QString::fromUtf8(argv[1]);
	QString outputPath = QString::fromUtf8(argv[2]);
	
	int numBands = 14; // Default value
	if (argc >= 4) {
		bool ok = false;
		numBands = QString::fromUtf8(argv[3]).toInt(&ok);
		if (!ok || numBands < 0 || numBands > 50) {
			cerr << "Invalid num_bands value. Must be between 0 and 50." << endl;
			return 1;
		}
	}
	
	double featherRadius = 512.0; // Default value
	if (argc == 5) {
		bool ok = false;
		featherRadius = QString::fromUtf8(argv[4]).toDouble(&ok);
		if (!ok || featherRadius < 0.0) {
			cerr << "Invalid feather_radius value. Must be >= 0." << endl;
			return 1;
		}
	}
	
	cout << "=== ReTawny V2 ===" << endl;
	cout << "Parameters:" << endl;
	cout << "  Input folder: " << qPrintable(folder) << endl;
	cout << "  Output file: " << qPrintable(outputPath) << endl;
	cout << "  Num bands: " << numBands << endl;
	cout << "  Feather radius: " << featherRadius << " pixels" << endl;
	cout << endl;



	cout << "[1/5] Loading tiles metadata..." << endl;
	auto t1 = high_resolution_clock::now();
	
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
	
	auto t2 = high_resolution_clock::now();
	cout << "  Loaded " << tiles.size() << " tiles in " 
	     << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;
	cout << endl;

	cout << "[2/5] Preparing blender..." << endl;
	auto t3 = high_resolution_clock::now();
	
	const QSize canvasSize = loader.canvasSize();
	if (!canvasSize.isValid() || canvasSize.isEmpty()) {
		cerr << "Invalid canvas: Cannot blend because the canvas size is invalid." << endl;
		return 1;
	}

	// Check for large images and disable OpenCL to avoid VRAM exhaustion
	const long long totalPixels = static_cast<long long>(canvasSize.width()) * canvasSize.height();
	const long long estimatedMB = (totalPixels * 3) / (1024 * 1024); // 3 bytes per pixel for BGR
	
	// MultiBandBlender creates pyramids that consume much more memory
	// Rough estimate: each band level ~= 1.33x original size, total ~= numBands * 1.5
	const long long estimatedBlendingMB = estimatedMB * numBands * 2;
	
	cout << "  Canvas size: " << canvasSize.width() << "x" << canvasSize.height() 
	     << " (~" << estimatedMB << " MB, ~" << estimatedBlendingMB << " MB with " << numBands << " bands)" << endl;
	     
	// Disable OpenCL if estimated VRAM usage > 4GB or if more than 5 bands
	if (estimatedBlendingMB > 4096 || numBands > 5) {
		cout << "  Disabling OpenCL to avoid VRAM exhaustion (using CPU)" << endl;
		cv::ocl::setUseOpenCL(false);
	} else {
		cout << "  Using OpenCL if available" << endl;
	}

	const cv::Rect roi(0, 0, canvasSize.width(), canvasSize.height());
	cv::detail::MultiBandBlender blender(false, numBands);
	blender.prepare(roi);
	
	auto t4 = high_resolution_clock::now();
	cout << "  Blender ready in " << duration_cast<milliseconds>(t4 - t3).count() << " ms" << endl;
	cout << endl;

	cout << "[3/5] Processing and feeding tiles..." << endl;
	auto t5 = high_resolution_clock::now();
	
	bool fedAny = false;
	int tileIndex = 0;
	for (OrthoLoader::Tile &tile : tiles) {
		tileIndex++;
		cout << "  Tile " << tileIndex << "/" << tiles.size() << ": " << qPrintable(tile.name) << "..." << flush;
		auto tileStart = high_resolution_clock::now();
		
		// Load tile into memory
		if (!loader.loadTile(&tile, &errorMessage)) {
			cerr << " FAILED: " << qPrintable(errorMessage) << endl;
			return 1;
		}
		if (tile.image.isNull()){
		   cerr << " FAILED: null image" << endl;
			return 1;
		}
		cv::Mat bgr = qImageToBgrMat(tile.image);

		if (bgr.empty()){
			loader.unloadTile(&tile);
			cerr << " FAILED: empty BGR" << endl;
			return 1;
		}

		// Load mask if available
		loader.loadMask(&tile, nullptr);
		
		// Build coverage mask with feathering (using loaded mask if available)
		cv::Mat mask = buildCoverageMask(tile.image, tile.mask, featherRadius, false);
		
		// Unload mask and tile to free memory
		loader.unloadMask(&tile);
		loader.unloadTile(&tile);
		
		if (mask.empty() || cv::countNonZero(mask) == 0) {
			cerr << " FAILED: empty or zero mask" << endl;
			return 1;
		}
		
		cv::Scalar avgColor = cv::mean(bgr, mask);

		// Fill only pixels where mask is zero with average color
		cv::Mat zeroMask = (mask == 0);
		bgr.setTo(avgColor, zeroMask);

		cv::Mat img16s;
		bgr.convertTo(img16s, CV_16SC3);
		blender.feed(img16s, mask, cv::Point(tile.x, tile.y));
		fedAny = true;
		
		auto tileEnd = high_resolution_clock::now();
		cout << " OK (" << duration_cast<milliseconds>(tileEnd - tileStart).count() << " ms)" << endl;
	}
	
	auto t6 = high_resolution_clock::now();
	cout << "  All tiles processed in " << duration_cast<seconds>(t6 - t5).count() << " seconds" << endl;
	cout << endl;

	if (!fedAny) {
		cerr << "Blending failed: No valid pixels were submitted to the blender." << endl;
		return 1;
	}

	cout << "[4/5] Blending..." << endl;
	auto t7 = high_resolution_clock::now();
	
	cv::Mat blended, blendedMask;
	blender.blend(blended, blendedMask);
	if (blended.empty()) {
		cerr << "Blending failed: OpenCV returned an empty result." << endl;
		return 1;
	}

	auto t8 = high_resolution_clock::now();
	cout << "  Blending completed in " << duration_cast<seconds>(t8 - t7).count() << " seconds" << endl;
	cout << endl;

	cout << "[5/5] Saving output..." << endl;
	auto t9 = high_resolution_clock::now();
	
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
	
	auto t10 = high_resolution_clock::now();
	cout << "  Output saved in " << duration_cast<milliseconds>(t10 - t9).count() << " ms" << endl;
	cout << endl;
	
	// Report statistics
	auto endTime = high_resolution_clock::now();
	auto totalSeconds = duration_cast<seconds>(endTime - startTime).count();
	
	struct rusage usage;
	cout << "=== Statistics ===" << endl;
	cout << "Total time: " << totalSeconds << " seconds (" 
	     << totalSeconds / 60 << "m " << totalSeconds % 60 << "s)" << endl;
	if (getrusage(RUSAGE_SELF, &usage) == 0) {
		const long peakMB = usage.ru_maxrss / (1024 * 1024);
		cout << "Peak memory usage: " << peakMB << " MB" << endl;
	}
	cout << endl;
	
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

cv::Mat buildCoverageMask(const QImage &source, const QImage &loadedMask, double featherRadius, bool sharp) {
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

	if (sharp || featherRadius <= 1.0)
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
	combinedDistance /= featherRadius;
	cv::min(combinedDistance, 1.0, combinedDistance);

	// Convert to 8-bit mask
	cv::Mat feathered;
	combinedDistance.convertTo(feathered, CV_8UC1, 255.0);

	// Zero out original masked pixels
	feathered.setTo(0, mask == 0);

	return feathered;
}
