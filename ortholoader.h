#ifndef ORTHOLOADER_H
#define ORTHOLOADER_H

#include <QImage>
#include <QSize>
#include <QVector>

#include <QString>

class QDir;

class OrthoLoader {
public:
	struct Tile {
		QString name;
		QString imagePath;  // Path to the image file
		QString maskPath;   // Path to the mask file (if exists)
		QString generatedMaskPath; // Path to generated Voronoi mask (if exists)
		QImage image;       // Loaded image (empty when unloaded)
		QImage mask;        // Loaded mask (empty when unloaded)
		int x = 0; // offset in pixels from composite origin along X
		int y = 0; // offset in pixels from composite origin along Y
		int width = 0;  // width in pixels
		int height = 0; // height in pixels
	};

	bool loadFromDirectory(const QString &directoryPath, QString *errorMessage = nullptr);
	bool generateVoronoiMasks(double overlapMargin = 20.0, QString *errorMessage = nullptr);
	
	bool loadTile(Tile *tile, QString *errorMessage = nullptr);
	void unloadTile(Tile *tile);
	bool loadMask(Tile *tile, QString *errorMessage = nullptr);
	void unloadMask(Tile *tile);

	bool empty() const { return tiles_.isEmpty(); }
	QVector<Tile> &tiles() { return tiles_; }
	const QVector<Tile> &tiles() const { return tiles_; }
	QSize canvasSize() const { return canvasSize_; }
	double pixelWidth() const { return pixelWidth_; }
	double pixelHeight() const { return pixelHeight_; }


private:
	struct TfwRecord {
		double scaleX = 0.0;
		double rotationY = 0.0;
		double rotationX = 0.0;
		double scaleY = 0.0;
		double translateX = 0.0;
		double translateY = 0.0;
	};

	bool parseTfw(const QString &filePath, TfwRecord *record, QString *errorMessage) const;
	bool parseMTDOrtho(const QString &filePath, QSize *canvasSize, QString *errorMessage) const;
	bool ensureRotationIsZero(const TfwRecord &record, const QString &tfwFile, QString *errorMessage) const;
	bool ensureResolutionConsistency(const TfwRecord &record, const QString &tfwFile, QString *errorMessage);
	bool computeTileOffset(const TfwRecord &record, Tile *tile, const QString &tfwFile, QString *errorMessage) const;
	QString resolveImagePath(const QDir &directory, const QString &tfwFile) const;
	QString resolveMaskPath(const QString &imagePath) const;
	bool finalizeTiles(QString *errorMessage);

	QVector<Tile> tiles_;
	QSize canvasSize_;
	double pixelWidth_ = 0.0;
	double pixelHeight_ = 0.0;
	TfwRecord referenceTfw_;
	bool hasReference_ = false;
	QSize referenceCanvasSize_;
};

#endif // ORTHOLOADER_H
