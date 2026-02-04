#!/bin/sh
# Build ReTawny V2 - retire AGL du Makefile (SDK macOS récents n'ont plus ce framework)
cd "$(dirname "$0")"
qmake retawny.pro
sed -i '' 's/-framework AGL -framework OpenGL/-framework OpenGL/g' Makefile
sed -i '' 's|-I/Applications[^ ]*AGL.framework/Headers/||g' Makefile
make
