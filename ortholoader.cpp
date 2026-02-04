#include "ortholoader.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QStringList>
#include <QXmlStreamReader>
#include <QRegularExpression>
#include <QImageReader>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
QStringList imageExtensions() {
	return {QStringLiteral("tif"), QStringLiteral("tiff"),
	        QStringLiteral("TIF"), QStringLiteral("TIFF")};
}

constexpr double kRotationTolerance = 0;
constexpr double kResolutionTolerance = 0;

bool nearlyZero(double value, double tolerance) {
	return std::abs(value) <= tolerance;
}
}

bool OrthoLoader::loadFromDirectory(const QString &directoryPath, QString *errorMessage) {
	tiles_.clear();
	canvasSize_ = QSize();
	pixelWidth_ = 0.0;
	pixelHeight_ = 0.0;
	hasReference_ = false;

	if (directoryPath.isEmpty()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("No directory selected.");
		return false;
	}

	QDir dir(directoryPath);
	if (!dir.exists()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Directory does not exist: %1").arg(directoryPath);
		return false;
	}

	// Check for Orthophotomosaic.tfw reference file
	const QString referenceTfwPath = dir.absoluteFilePath(QStringLiteral("Orthophotomosaic.tfw"));
	const QString mtdOrthoPath = dir.absoluteFilePath(QStringLiteral("MTDOrtho.xml"));
	
	if (QFileInfo::exists(referenceTfwPath) && QFileInfo::exists(mtdOrthoPath)) {
		if (!parseTfw(referenceTfwPath, &referenceTfw_, errorMessage))
			return false;
		if (!ensureRotationIsZero(referenceTfw_, QStringLiteral("Orthophotomosaic.tfw"), errorMessage))
			return false;
		if (!parseMTDOrtho(mtdOrthoPath, &referenceCanvasSize_, errorMessage))
			return false;
		hasReference_ = true;
		pixelWidth_ = std::abs(referenceTfw_.scaleX);
		pixelHeight_ = std::abs(referenceTfw_.scaleY);
	}

	const QStringList tfwFiles = dir.entryList({QStringLiteral("*.tfw"), QStringLiteral("*.TFW")}, QDir::Files | QDir::Readable);
	if (tfwFiles.isEmpty()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("No TFW files found in %1").arg(directoryPath);
		return false;
	}

	for (const QString &tfwFileName : tfwFiles) {
		// Skip the reference file
		if (tfwFileName == QStringLiteral("Orthophotomosaic.tfw"))
			continue;
		const QString tfwPath = dir.absoluteFilePath(tfwFileName);
		TfwRecord record;
		if (!parseTfw(tfwPath, &record, errorMessage))
			return false;

		if (!ensureRotationIsZero(record, tfwFileName, errorMessage))
			return false;

		if (!ensureResolutionConsistency(record, tfwFileName, errorMessage))
			return false;

		const QString imagePath = resolveImagePath(dir, tfwFileName);
		if (imagePath.isEmpty()) {
			// Skip TFW files without matching images (like Orthophotomosaic.tfw)
			continue;
		}

		Tile tile;
		tile.name = QFileInfo(imagePath).fileName();
		tile.imagePath = imagePath;
		tile.maskPath = resolveMaskPath(imagePath);
		
		// Load image dimensions without loading full image
		QImageReader reader(imagePath);
		if (!reader.canRead()) {
			if (errorMessage)
				*errorMessage = QStringLiteral("Cannot read image dimensions from %1").arg(imagePath);
			return false;
		}
		QSize size = reader.size();
		if (!size.isValid() || size.isEmpty()) {
			if (errorMessage)
				*errorMessage = QStringLiteral("Invalid image dimensions in %1").arg(imagePath);
			return false;
		}
		tile.width = size.width();
		tile.height = size.height();
		
		if (!computeTileOffset(record, &tile, tfwFileName, errorMessage))
			return false;

		tiles_.push_back(tile);
	}

	if (!finalizeTiles(errorMessage))
		return false;

	// Unload all tiles after computing offsets
	for (Tile &tile : tiles_) {
		unloadTile(&tile);
	}

	return true;
}

bool OrthoLoader::parseTfw(const QString &filePath, TfwRecord *record, QString *errorMessage) const {
	if (!record)
		return false;

	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Unable to open %1").arg(filePath);
		return false;
	}

	QTextStream stream(&file);
	double values[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	int valueCount = 0;

	while (!stream.atEnd() && valueCount < 6) {
		const QString line = stream.readLine().trimmed();
		if (line.isEmpty())
			continue;

		bool ok = false;
		const double number = line.toDouble(&ok);
		if (!ok) {
			if (errorMessage)
				*errorMessage = QStringLiteral("Invalid numeric value in %1: %2").arg(filePath, line);
			return false;
		}

		values[valueCount++] = number;
	}

	if (valueCount != 6) {
		if (errorMessage)
			*errorMessage = QStringLiteral("TFW file %1 does not contain 6 values.").arg(filePath);
		return false;
	}

	record->scaleX = values[0];
	record->rotationY = values[1];
	record->rotationX = values[2];
	record->scaleY = values[3];
	record->translateX = values[4];
	record->translateY = values[5];

	return true;
}

bool OrthoLoader::ensureRotationIsZero(const TfwRecord &record, const QString &tfwFile, QString *errorMessage) const {
	if (nearlyZero(record.rotationX, kRotationTolerance) &&
	    nearlyZero(record.rotationY, kRotationTolerance))
		return true;

	if (errorMessage)
		*errorMessage = QStringLiteral("Expected zero rotation in %1").arg(tfwFile);
	return false;
}

bool OrthoLoader::ensureResolutionConsistency(const TfwRecord &record, const QString &tfwFile, QString *errorMessage) {
	const double width = std::abs(record.scaleX);
	const double height = std::abs(record.scaleY);
	if (width <= 0.0 || height <= 0.0) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Invalid pixel size in %1").arg(tfwFile);
		return false;
	}

	if (pixelWidth_ == 0.0 && pixelHeight_ == 0.0) {
		pixelWidth_ = width;
		pixelHeight_ = height;
		return true;
	}

	if (!nearlyZero(pixelWidth_ - width, kResolutionTolerance) ||
	    !nearlyZero(pixelHeight_ - height, kResolutionTolerance)) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Tile %1 uses a different resolution").arg(tfwFile);
		return false;
	}

	return true;
}

bool OrthoLoader::computeTileOffset(const TfwRecord &record, Tile *tile, const QString &tfwFile, QString *errorMessage) const {
	if (!tile)
		return false;

	if (pixelWidth_ <= 0.0 || pixelHeight_ <= 0.0) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Missing resolution metadata before processing %1").arg(tfwFile);
		return false;
	}

	const double rawX = record.translateX / pixelWidth_;
	const double rawY = -record.translateY / pixelHeight_;
	const int offsetX = static_cast<int>(std::lround(rawX));
	const int offsetY = static_cast<int>(std::lround(rawY));

	tile->x = offsetX;
	tile->y = offsetY;
	return true;
}

QString OrthoLoader::resolveImagePath(const QDir &directory, const QString &tfwFile) const {
	const QFileInfo tfwInfo(tfwFile);
	const QString baseName = tfwInfo.completeBaseName();

	for (const QString &extension : imageExtensions()) {
		const QString candidate = directory.absoluteFilePath(baseName + QLatin1Char('.') + extension);
		if (QFileInfo::exists(candidate))
			return candidate;
	}

	return QString();
}

QString OrthoLoader::resolveMaskPath(const QString &imagePath) const {
	const QFileInfo imageInfo(imagePath);
	const QString fileName = imageInfo.fileName();
	const QString dirPath = imageInfo.absolutePath();
	
	// Try to find mask by replacing Ort_ with PC_
	if (fileName.startsWith(QStringLiteral("Ort_"), Qt::CaseInsensitive)) {
		QString maskFileName = fileName;
		maskFileName.replace(0, 4, QStringLiteral("PC_"));
		const QString maskPath = QDir(dirPath).absoluteFilePath(maskFileName);
		if (QFileInfo::exists(maskPath))
			return maskPath;
	}
	
	// No mask found - will use magenta detection fallback
	return QString();
}

bool OrthoLoader::loadTile(Tile *tile, QString *errorMessage) {
	if (!tile) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Invalid tile pointer");
		return false;
	}

	if (tile->imagePath.isEmpty()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Tile has no image path");
		return false;
	}

	// Load image
	QImage image(tile->imagePath);
	if (image.isNull()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Failed to load image %1").arg(tile->imagePath);
		return false;
	}
	tile->image = image.convertToFormat(QImage::Format_ARGB32);

	return true;
}

void OrthoLoader::unloadTile(Tile *tile) {
	if (!tile)
		return;
	
	tile->image = QImage();
}

bool OrthoLoader::loadMask(Tile *tile, QString *errorMessage) {
	if (!tile) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Invalid tile pointer");
		return false;
	}

	// Try to load generated Voronoi mask first
	if (!tile->generatedMaskPath.isEmpty() && QFileInfo::exists(tile->generatedMaskPath)) {
		QImage mask(tile->generatedMaskPath);
		if (!mask.isNull()) {
			tile->mask = mask.convertToFormat(QImage::Format_ARGB32);
			return true;
		}
	}

	// Fallback: load PC_ mask if available
	if (!tile->maskPath.isEmpty() && QFileInfo::exists(tile->maskPath)) {
		QImage mask(tile->maskPath);
		if (!mask.isNull()) {
			tile->mask = mask.convertToFormat(QImage::Format_ARGB32);
			return true;
		}
	}

	// No mask available
	return false;
}

void OrthoLoader::unloadMask(Tile *tile) {
	if (!tile)
		return;
	
	tile->mask = QImage();
}

bool OrthoLoader::parseMTDOrtho(const QString &filePath, QSize *canvasSize, QString *errorMessage) const {
	if (!canvasSize)
		return false;

	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Unable to open %1").arg(filePath);
		return false;
	}

	QXmlStreamReader xml(&file);
	while (!xml.atEnd()) {
		xml.readNext();
		if (xml.isStartElement() && xml.name() == QStringLiteral("NombrePixels")) {
			const QString text = xml.readElementText().trimmed();
			const QStringList parts = text.split(QRegularExpression(QStringLiteral("\\s+")));
			if (parts.size() != 2) {
				if (errorMessage)
					*errorMessage = QStringLiteral("Invalid NombrePixels format in %1").arg(filePath);
				return false;
			}
			bool okWidth = false, okHeight = false;
			const int width = parts[0].toInt(&okWidth);
			const int height = parts[1].toInt(&okHeight);
			if (!okWidth || !okHeight || width <= 0 || height <= 0) {
				if (errorMessage)
					*errorMessage = QStringLiteral("Invalid pixel dimensions in %1").arg(filePath);
				return false;
			}
			*canvasSize = QSize(width, height);
			return true;
		}
	}

	if (xml.hasError()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("XML parsing error in %1: %2").arg(filePath, xml.errorString());
		return false;
	}

	if (errorMessage)
		*errorMessage = QStringLiteral("NombrePixels not found in %1").arg(filePath);
	return false;
}

bool OrthoLoader::generateVoronoiMasks(double overlapMargin, QString *errorMessage) {
	if (tiles_.isEmpty()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("No tiles loaded.");
		return false;
	}

	if (overlapMargin < 0.0) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Invalid overlap margin: must be >= 0.");
		return false;
	}

	// Load PC_ masks for all tiles
	QVector<cv::Mat> pcMasks;
	pcMasks.reserve(tiles_.size());
	
	for (int i = 0; i < tiles_.size(); ++i) {
		Tile &tile = tiles_[i];
		cv::Mat pcMask;
		
		// Load PC_ mask if available
		if (!tile.maskPath.isEmpty() && QFileInfo::exists(tile.maskPath)) {
			pcMask = cv::imread(tile.maskPath.toStdString(), cv::IMREAD_GRAYSCALE);
			if (pcMask.empty() || pcMask.cols != tile.width || pcMask.rows != tile.height) {
				if (errorMessage)
					*errorMessage = QStringLiteral("Failed to load or invalid PC_ mask: %1").arg(tile.maskPath);
				return false;
			}
		} else {
			// No PC_ mask: all pixels are valid (all black = usable)
			pcMask = cv::Mat(tile.height, tile.width, CV_8UC1, cv::Scalar(0));
		}
		
		pcMasks.push_back(pcMask);
	}

	// Precompute tile centers in canvas coordinates
	struct TileCenter {
		double x;
		double y;
		int index;
	};
	QVector<TileCenter> centers;
	centers.reserve(tiles_.size());
	
	for (int i = 0; i < tiles_.size(); ++i) {
		const Tile &tile = tiles_[i];
		TileCenter center;
		center.x = tile.x + tile.width / 2.0;
		center.y = tile.y + tile.height / 2.0;
		center.index = i;
		centers.push_back(center);
	}

	// Generate mask for each tile
	for (int tileIdx = 0; tileIdx < tiles_.size(); ++tileIdx) {
		Tile &tile = tiles_[tileIdx];
		const cv::Mat &currentPcMask = pcMasks[tileIdx];

		// Create mask with same size as tile
		cv::Mat voronoiMask(tile.height, tile.width, CV_8UC1, cv::Scalar(0));

		// For each pixel in the tile
		for (int localY = 0; localY < tile.height; ++localY) {
			uchar *maskRow = voronoiMask.ptr<uchar>(localY);
			const uchar *pcMaskRow = currentPcMask.ptr<uchar>(localY);
			
			for (int localX = 0; localX < tile.width; ++localX) {
				// Check if pixel is valid in PC_ mask (black = valid, white = masked)
				if (pcMaskRow[localX] > 128) {
					// Pixel is masked in PC_ (white), skip it
					maskRow[localX] = 0;
					continue;
				}
				
				// Canvas coordinates
				const double canvasX = tile.x + localX;
				const double canvasY = tile.y + localY;

				// Find distances to all tile centers, considering only valid pixels in their PC_ masks
				double minDist = std::numeric_limits<double>::max();
				double secondMinDist = std::numeric_limits<double>::max();
				int closestIdx = -1;

				for (const TileCenter &center : centers) {
					// Check if this canvas pixel falls within the other tile's bounds
					const Tile &otherTile = tiles_[center.index];
					const int otherLocalX = static_cast<int>(canvasX - otherTile.x);
					const int otherLocalY = static_cast<int>(canvasY - otherTile.y);
					
					// Skip if pixel is outside other tile's bounds
					if (otherLocalX < 0 || otherLocalX >= otherTile.width ||
					    otherLocalY < 0 || otherLocalY >= otherTile.height) {
						continue;
					}
					
					// Check if pixel is valid in other tile's PC_ mask
					const cv::Mat &otherPcMask = pcMasks[center.index];
					if (otherPcMask.at<uchar>(otherLocalY, otherLocalX) > 128) {
						// Pixel is masked (white) in other tile, skip this tile
						continue;
					}
					
					// Calculate distance to center
					const double dx = canvasX - center.x;
					const double dy = canvasY - center.y;
					const double dist = std::sqrt(dx * dx + dy * dy);

					if (dist < minDist) {
						secondMinDist = minDist;
						minDist = dist;
						closestIdx = center.index;
					} else if (dist < secondMinDist) {
						secondMinDist = dist;
					}
				}

				// Distance from frontier: positive = towards our center, negative = towards other center
				const double distToFrontier = (secondMinDist - minDist) / 2.0;
				
				// Include pixels up to overlapMargin BEYOND the Voronoi frontier
				// This means accepting pixels even when we're NOT the closest, if we're close enough
				const bool weAreClosest = (closestIdx == tileIdx);
				
				// Calculate how far we are from the frontier
				// If we're closest: distToFrontier is positive (good)
				// If we're not closest: we want to check if secondMinDist - ourDist < 2*overlapMargin
				double distanceFromFrontier;
				if (weAreClosest) {
					distanceFromFrontier = distToFrontier;
				} else {
					// We're not closest, but we might still be within overlap range
					distanceFromFrontier = -distToFrontier; // Invert
				}
				
				// Accept if within overlapMargin of frontier
				if (distanceFromFrontier >= -overlapMargin) {
					if (distanceFromFrontier >= overlapMargin) {
						// Far from frontier: full ownership
						maskRow[localX] = 255;
					} else {
						// Near frontier: gradient
						// At frontier (dist=0): 255
						// At -overlapMargin: 0
						// At +overlapMargin: 255
						const double ratio = (distanceFromFrontier + overlapMargin) / (2.0 * overlapMargin);
						maskRow[localX] = static_cast<uchar>(ratio * 255.0);
					}
				}
				// else: pixel too far, leave at 0
			}
		}

		// Save the mask
		QFileInfo imageInfo(tile.imagePath);
		const QString baseName = imageInfo.completeBaseName();
		const QString maskFileName = baseName + QStringLiteral("_voronoi_mask.tif");
		const QString maskPath = imageInfo.absolutePath() + QDir::separator() + maskFileName;

		if (!cv::imwrite(maskPath.toStdString(), voronoiMask)) {
			if (errorMessage)
				*errorMessage = QStringLiteral("Failed to save Voronoi mask: %1").arg(maskPath);
			return false;
		}

		tile.generatedMaskPath = maskPath;
	}

	return true;
}

bool OrthoLoader::finalizeTiles(QString *errorMessage) {
	if (tiles_.isEmpty()) {
		if (errorMessage)
			*errorMessage = QStringLiteral("No TIFF images were loaded.");
		return false;
	}

	if (pixelWidth_ <= 0.0 || pixelHeight_ <= 0.0) {
		if (errorMessage)
			*errorMessage = QStringLiteral("Invalid pixel size metadata.");
		return false;
	}

	if (hasReference_) {
		// Use Orthophotomosaic.tfw as reference for georeferencing
		// Calculate reference origin in pixel coordinates
		const double refOriginX = referenceTfw_.translateX / pixelWidth_;
		const double refOriginY = -referenceTfw_.translateY / pixelHeight_;
		const int refX = static_cast<int>(std::lround(refOriginX));
		const int refY = static_cast<int>(std::lround(refOriginY));

		// Adjust all tiles relative to reference origin
		for (Tile &tile : tiles_) {
			tile.x -= refX;
			tile.y -= refY;
		}

		// Use full canvas size from MTDOrtho.xml for correct georeferencing
		if (referenceCanvasSize_.isValid() && !referenceCanvasSize_.isEmpty()) {
			canvasSize_ = referenceCanvasSize_;
		} else {
			// Fallback: calculate tight bounding box of tiles
			int minX = std::numeric_limits<int>::max();
			int minY = std::numeric_limits<int>::max();
			int maxX = std::numeric_limits<int>::min();
			int maxY = std::numeric_limits<int>::min();
			
			for (const Tile &tile : tiles_) {
				minX = std::min(minX, tile.x);
				minY = std::min(minY, tile.y);
				maxX = std::max(maxX, tile.x + tile.width);
				maxY = std::max(maxY, tile.y + tile.height);
			}

			// Shift tiles to start at (0,0) while preserving relative positions
			for (Tile &tile : tiles_) {
				tile.x -= minX;
				tile.y -= minY;
			}

			canvasSize_ = QSize(maxX - minX, maxY - minY);
		}
	} else {
		// Fallback to original behavior (fit all tiles)
		int minX = std::numeric_limits<int>::max();
		int minY = std::numeric_limits<int>::max();
		for (const Tile &tile : tiles_) {
			minX = std::min(minX, tile.x);
			minY = std::min(minY, tile.y);
		}

		int canvasWidth = 0;
		int canvasHeight = 0;
		for (Tile &tile : tiles_) {
			tile.x -= minX;
			tile.y -= minY;
			canvasWidth = std::max(canvasWidth, tile.x + tile.width);
			canvasHeight = std::max(canvasHeight, tile.y + tile.height);
		}

		canvasSize_ = QSize(canvasWidth, canvasHeight);
	}
	return true;
}
