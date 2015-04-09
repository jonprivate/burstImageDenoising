function [features, validPoints] = getFeatures(image)
baseImage = rgb2gray(image);
POINTS = detectSURFFeatures(baseImage);
[features, validPoints] = extractFeatures(baseImage, POINTS);