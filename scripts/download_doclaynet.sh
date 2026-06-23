#!/bin/bash
echo "Downloading DocLayNet Core Dataset (COCO format)..."
echo "This is a 30GB download. Please be patient."

# Create data directory
mkdir -p DocLayNet_core
cd DocLayNet_core

# Download
wget -c https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip

# Unzip
echo "Unzipping dataset..."
unzip -q DocLayNet_core.zip

echo "Download and extraction complete!"
