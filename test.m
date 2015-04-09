close all; clear all;
% first, import two images
imageNum = 10;
imageSet = cell(1, imageNum);
for i = 1 : imageNum
    imageSet{i} = imread([num2str(i - 1), '.jpg']);
end

ref = 5;
refImage = imageSet{ref}; % get reference image
refPyramid = getPyramids(refImage); % get pyramid for reference image
[refFeatures, refPoints] = getFeatures(refPyramid{1}); % get features and valid points of reference image

baseHomographySet = zeros(size(refPyramid{1}, 1), size(refPyramid{1}, 2), 2, length(imageSet));
baseImageSet = cell(1, length(imageSet));
for i = 1 : length(imageSet)
    if i == ref
        baseImageSet{i} = refPyramid{1};
        continue;
    end
    pyramid = getPyramids(imageSet{i});
    homographyFlowPyramid = getHomographyFlowPyramidWithRefFeatures(refPyramid, refFeatures, refPoints, pyramid);
    baseHomographySet(:,:,:,i) = homographyFlowPyramid{1};
    baseImageSet{i} = pyramid{1};
    disp(['homography ', num2str(i), ' complete']);
end

baseGrayScaleImageSet = zeros(size(refPyramid{1}, 1), size(refPyramid{1}, 2), length(imageSet));
for i = 1 : length(imageSet)
    baseGrayScaleImageSet(:, :, i) = rgb2gray(baseImageSet{i});
end
baseMedianImage = median(baseGrayScaleImageSet, 3); % get median image
baseIntegralMedImage = integralImage(baseMedianImage);
baseIntegralImageSet = zeros(size(refPyramid{1}, 1) + 1, size(refPyramid{1}, 2) + 1, length(imageSet));
for i = 1 : length(imageSet)
    baseIntegralImageSet(:, :, i) = integralImage(baseGrayScaleImageSet(:, :, i));
end
baseRefConsistentPixelMap = zeros(size(baseImageSet{1}, 1), size(baseImageSet{1}, 2), length(baseImageSet));
baseMedConsistentPixelMap = zeros(size(baseImageSet{1}, 1), size(baseImageSet{1}, 2), length(baseImageSet));
tau = 10; % threshold for selecting consistent pixels
halfWidth = 2; halfHeight = 2;
rows = size(baseRefConsistentPixelMap, 1);
cols = size(baseRefConsistentPixelMap, 2);
for r = 1 : rows
    for c = 1 : cols
        % first get the pixel from the reference image
        sR = max(1, r - halfHeight);
        sC = max(1, c - halfWidth);
        eR = min(rows, r + halfHeight);
        eC = min(cols, c + halfWidth);
        refPix = baseIntegralImageSet(eR+1,eC+1,ref) - baseIntegralImageSet(eR+1,sC,ref) - baseIntegralImageSet(sR,eC+1,ref) + baseIntegralImageSet(sR,sC,ref);
        medPix = baseIntegralMedImage(eR+1,eC+1) - baseIntegralMedImage(eR+1,sC) - baseIntegralMedImage(sR,eC+1) + baseIntegralMedImage(sR,sC);
        for i = 1 : length(imageSet)
            if i == ref
                baseRefConsistentPixelMap(r,c,i) = 1;
                continue;
            end
            % get positions according to homography flow
            ri = floor(r + baseHomographySet(r,c,2,i));
            ci = floor(c + baseHomographySet(r,c,1,i));
            if ri < 1 || ri > rows || ci < 1 || ci > cols
                continue;
            end
            sR = max(1, ri - halfHeight);
            sC = max(1, ci - halfWidth);
            eR = min(rows, ri + halfHeight);
            eC = min(cols, ci + halfWidth);
            iPix = baseIntegralImageSet(eR+1,eC+1,ref) - baseIntegralImageSet(eR+1,sC,ref) - baseIntegralImageSet(sR,eC+1,ref) + baseIntegralImageSet(sR,sC,ref);
            if abs(refPix - iPix) < tau
                baseRefConsistentPixelMap(r,c,i) = 1;
            end
            if abs(medPix - iPix) < tau
                baseMedConsistentPixelMap(r,c,i) = 1;
            end
        end
    end
end

% combine median consistent pixels and reference consistent pixels
baseConsistentPixelMap = zeros(size(baseRefConsistentPixelMap));
reliableNumber = floor(imageNum / 2);
consistentPixelNumMap = sum(baseMedConsistentPixelMap, 3);
consistentPixelNumMap = consistentPixelNumMap > reliableNumber;
% perform majority filter
consistentPixelNumMap = bwmorph(consistentPixelNumMap, 'majority');

for r = 1 : rows
    for c = 1 : cols
        % case 1: union of the two
        if baseMedConsistentPixelMap(r,c,ref) == 1
            baseConsistentPixelMap(r,c,:) = baseRefConsistentPixelMap(r,c,:) | baseMedConsistentPixelMap(r,c,:);
        % case 2: judge if median based is reliable
        elseif consistentPixelNumMap(r,c) == 1
            % median based result is reliable
            baseConsistentPixelMap(r,c,:) = baseMedConsistentPixelMap(r,c,:);
        else
            % median based result is not reliable
            baseConsistentPixelMap(r,c,:) = baseRefConsistentPixelMap(r,c,:);
        end
    end
end

for i = 1 : size(baseConsistentPixelMap,3)
    baseConsistentPixelMap(:, :, i) = bwmorph(baseConsistentPixelMap(:, :, i), 'majority');
end
