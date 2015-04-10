close all; clear all;
% % import burst of images
% imageNum = 10;
% imageSet = cell(1, imageNum);
% for i = 1 : imageNum
%     imageSet{i} = imread([num2str(i - 1), '.jpg']);
% end
% % set the reference image to be the 5th one
% ref = 5;
% refImage = imageSet{ref}; % get reference image
% refPyramid = getPyramids(refImage); % get pyramid for reference image
% [refFeatures, refPoints] = getFeatures(refPyramid{1}); % get features and valid points of reference image
% 
% % homography flows for all base images (base: coarsest level of the pyramid)
% baseHomographySet = zeros(size(refPyramid{1}, 1), size(refPyramid{1}, 2), 2, length(imageSet));
% % all base images
% baseImageSet = cell(1, length(imageSet));
% for i = 1 : length(imageSet)
%     if i == ref
%         baseImageSet{i} = refPyramid{1};
%         continue;
%     end
%     pyramid = getPyramids(imageSet{i});
%     homographyFlowPyramid = getHomographyFlowPyramidWithRefFeatures(refPyramid, refFeatures, refPoints, pyramid);
%     baseHomographySet(:,:,:,i) = homographyFlowPyramid{1};
%     baseImageSet{i} = pyramid{1};
%     disp(['homography ', num2str(i), ' complete']);
% end
% 
% % take grayscale base images to compute pixel difference
% baseGrayScaleImageSet = zeros(size(refPyramid{1}, 1), size(refPyramid{1}, 2), length(imageSet));
% for i = 1 : length(imageSet)
%     baseGrayScaleImageSet(:, :, i) = rgb2gray(baseImageSet{i});
% end
% % compute median image of all base images (for median value based consistent pixel selection)
% baseConsistentImageSet = getConsistentImageSet(baseGrayScaleImageSet, baseHomographySet);
% baseMedianImage = median(baseConsistentImageSet, 3);
% baseIntegralMedImage = integralImage(baseMedianImage); % integral image of the median image
% % integral images of all base images
% baseIntegralImageSet = zeros(size(refPyramid{1}, 1) + 1, size(refPyramid{1}, 2) + 1, length(imageSet));
% for i = 1 : length(imageSet)
%     baseIntegralImageSet(:, :, i) = integralImage(baseGrayScaleImageSet(:, :, i));
% end

% % set of consistent pixel indexes: reference based and median based
% baseRefConsistentPixelMap = zeros(size(baseImageSet{1}, 1), size(baseImageSet{1}, 2), length(baseImageSet));
% baseMedConsistentPixelMap = zeros(size(baseImageSet{1}, 1), size(baseImageSet{1}, 2), length(baseImageSet));
% % set of potential consistent pixel indexes
% basePotentialConsistentPixelMap = zeros(size(baseImageSet{1}, 1), size(baseImageSet{1}, 2), length(baseImageSet), 2);
% tau = 10; % threshold for selecting consistent pixels
% halfWidth = 2; halfHeight = 2;
% rows = size(baseRefConsistentPixelMap, 1);
% cols = size(baseRefConsistentPixelMap, 2);
% for r = 1 : rows
%     for c = 1 : cols
%         % first get the pixel from the reference image
%         sR = max(1, r - halfHeight);
%         sC = max(1, c - halfWidth);
%         eR = min(rows, r + halfHeight);
%         eC = min(cols, c + halfWidth);
%         pixNum = (eR - sR + 1) * (eC - sC + 1);
%         refPix = baseIntegralImageSet(eR+1,eC+1,ref) - baseIntegralImageSet(eR+1,sC,ref) - baseIntegralImageSet(sR,eC+1,ref) + baseIntegralImageSet(sR,sC,ref);
%         refPix = refPix / pixNum;
%         medPix = baseIntegralMedImage(eR+1,eC+1) - baseIntegralMedImage(eR+1,sC) - baseIntegralMedImage(sR,eC+1) + baseIntegralMedImage(sR,sC);
%         medPix = medPix / pixNum;
%         for i = 1 : length(imageSet)
%             if i == ref
%                 baseRefConsistentPixelMap(r,c,i) = 1;
%                 continue;
%             end
%             % get positions according to homography flow
%             ri = floor(r + baseHomographySet(r,c,2,i));
%             ci = floor(c + baseHomographySet(r,c,1,i));
%             basePotentialConsistentPixelMap(r,c,i,1) = ri;
%             basePotentialConsistentPixelMap(r,c,i,2) = ci;
%             if ri < 1 || ri > rows || ci < 1 || ci > cols
%                 continue;
%             end
%             sR = max(1, ri - halfHeight);
%             sC = max(1, ci - halfWidth);
%             eR = min(rows, ri + halfHeight);
%             eC = min(cols, ci + halfWidth);
%             pixNum = (eR - sR + 1) * (eC - sC + 1);
%             iPix = baseIntegralImageSet(eR+1,eC+1,ref) - baseIntegralImageSet(eR+1,sC,ref) - baseIntegralImageSet(sR,eC+1,ref) + baseIntegralImageSet(sR,sC,ref);
%             iPix = iPix / pixNum;
%             if abs(refPix - iPix) < tau
%                 baseRefConsistentPixelMap(r,c,i) = 1;
%             end
%             if abs(medPix - iPix) < tau
%                 baseMedConsistentPixelMap(r,c,i) = 1;
%             end
%         end
%     end
% end
% 
% % combine median consistent pixels and reference consistent pixels
% baseConsistentPixelMap = zeros(size(baseRefConsistentPixelMap));
% reliableNumber = floor(imageNum / 2);
% consistentPixelNumMap = sum(baseMedConsistentPixelMap > 0, 3);
% consistentPixelNumMap = consistentPixelNumMap > reliableNumber;
% % perform majority filter
% consistentPixelNumMap = bwmorph(consistentPixelNumMap, 'majority');
% 
% for r = 1 : rows
%     for c = 1 : cols
%         % case 1: union of the two
%         if baseMedConsistentPixelMap(r,c,ref) == 1
%             baseConsistentPixelMap(r,c,:) = baseRefConsistentPixelMap(r,c,:) | baseMedConsistentPixelMap(r,c,:);
%         % case 2: judge if median based is reliable
%         elseif consistentPixelNumMap(r,c) == 1
%             % median based result is reliable
%             baseConsistentPixelMap(r,c,:) = baseMedConsistentPixelMap(r,c,:);
%         else
%             % median based result is not reliable
%             baseConsistentPixelMap(r,c,:) = baseRefConsistentPixelMap(r,c,:);
%         end
%     end
% end
% 
% for i = 1 : size(baseConsistentPixelMap,3)
%     baseConsistentPixelMap(:, :, i) = bwmorph(baseConsistentPixelMap(:, :, i), 'majority');
% end

load current.mat
%% fusion stage
% first estimate noise
fineRefImage = refPyramid{length(refPyramid)};
fineGrayScaleRefImage = rgb2gray(fineRefImage);
nontextureMap = imresize(edge(baseMedianImage), size(fineGrayScaleRefImage), 'near');
inds = find(nontextureMap);
sigma2 = computeSigma2FromDiffVector(fineGrayScaleRefImage(inds) - imresize(baseMedianImage, size(fineGrayScaleRefImage), 'bilinear'));
% second perform temporal fusion
baseConsistentPixelMapR = basePotentialConsistentPixelMap(:,:,:,1) .* baseConsistentPixelMap;
baseConsistentPixelMapC = basePotentialConsistentPixelMap(:,:,:,2) .* baseConsistentPixelMap;
for level = 1 : length(refPyramid)
    rows = size(refPyramid{level}, 1);
    cols = size(refPyramid{level}, 2);
    % reuse the consistent pixel map for all levels
    if level > 1
        levelConsistentPixelMapR = imresize(baseConsistentPixelMapR, [rows, cols], 'near');
        levelConsistentPixelMapC = imresize(baseConsistentPixelMapC, [rows, cols], 'near');
    else
        levelConsistentPixelMapR = baseConsistentPixelMapR;
        levelConsistentPixelMapC = baseConsistentPixelMapC;
    end
    % get the set of all frames at this level
    levelImageSet = [];
    for i = 1 : length(imageSet)
        if i == ref
            levelImageSet = cat(4, levelImageSet, refPyramid{level});
            continue;
        else
            if level < length(refPyramid)
                ithImage = imresize(imageSet{i},[rows,cols], 'bilinear');
            else
                ithImage = imageSet{i};
            end
            levelImageSet = cat(4, levelImageSet, ithImage);
        end
    end
    levelMeanImage = mean(levelImageSet, 4);
    % use consistent image to compute mean value and variance
    levelGrayScaleImageSet = zeros(size(levelImageSet,1), size(levelImageSet,2), size(levelImageSet,4));
    for i = 1 : length(imageSet)
        levelGrayScaleImageSet(:,:,i) = rgb2gray(levelImageSet(:,:,:,i));
    end
    % get level homography set
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    levelConsistentImageSet = getConsistentImageSet(levelGrayScaleImageSet, levelHomographySet);
    levelConsistentPixelMap = levelConsistentPixelMapR > 0;
    levelConsistentImageSet = levelConsistentImageSet .* levelConsistentPixelMap;
    meanImage = mean(levelConsistentImageSet, 3);
    sigmac2Map = sum((levelConsistentImageSet - repmat(meanImage, [1 1 size(levelConsistentImageSet, 3)])) .^ 2, 3) ./ sum(levelConsistentPixelMap, 3) - sigma2;
    sigmac2Map = (sigmac2Map > 0) * sigmac2Map;
    sigmac2Map = sigmac2Map ./ (sigmac2Map + sigma2);
    sigmac2Map = repmat(sigmac2Map, [1 1 3]);
    levelConsistentPixelMap = reshape(size(levelConsistentPixelMap,1), size(levelConsistentPixelMap,2), 1, size(levelConsistentPixelMap,3));
    levelConsistentPixelMap = repmat(levelConsistentPixelMap, [1,1,3,1]);
    meanImage = sum(levelImageSet .* levelConsistentPixelMap, 4) ./ sum(levelConsistentPixelMap, 4);
    refPyramid{level} = meanImage + sigmac2Map .* (refPyramid{level} - meanImage);
end

% compute sigma for the finest level reference image
differenceImage = [];
rs = [1 2 2 3];
cs = [2 1 3 2];
for i = 1 : 4
    h = zeros(3,3);
    h(2,2) = 1;
    h(rs(i), cs(i)) = -1;
    differenceImage = cat(3, differenceImage, abs(conv2(fineGrayScaleRefImage, h, 'same')));
end
differenceImage = max(differenceImage, [], 3);
differenceImage = 1 ./ (1 + exp(-5 * differenceImage / (sqrt(sigma2) - 3)));
for level = 2 : length(refPyramid)
    levelSpatiallyFilteredImage = refPyramid{level};
    [rows, cols, ~] = size(levelSpatiallyFilteredImage);
    halfWidth = 2; halfHeight = 2;
    levelGrayScaleImage = rgb2gray(levelSpatiallyFilteredImage);
    levelGrayScaleGradientImageSet = zeros(rows, cols, 2);
    hy = [-1 -2 -1; 0 0 0; 1 2 1];
    hx = [-1 0 1; -2 0 2; -1 0 1];
    levelGrayScaleGradientImageSet(:,:,1) = conv2(levelGrayScaleImage, hx, 'same'); % vertical
    levelGrayScaleGradientImageSet(:,:,2) = conv2(levelGrayScaleImage, hy, 'same'); % horizontal
    levelGrayScaleGradientImageSet = abs(atan2(levelGrayScaleGradientImageSet(:,:,2), levelGrayScaleGradientImageSet(:,:,1)));
    levelGrayScaleGradientImageSet = levelGrayScaleGradientImageSet / pi * 180;
    for r = 1 + halfHeight : rows - halfHeight
        for c = 1 + halfWidth : cols - halfWidth
            if levelGrayScaleGradientImageSet(r,c) >= 0 && levelGrayScaleGradientImageSet(r,c) < 22.5
                % vertical
                levelSpatiallyFilteredImage(r,c,:) = (levelSpatiallyFilteredImage(r-2,c,:) + levelSpatiallyFilteredImage(r-1,c,:)...
                    + levelSpatiallyFilteredImage(r,c,:) + levelSpatiallyFilteredImage(r+1,c,:) + levelSpatiallyFilteredImage(r+2,c,:)) / 5;
            end
            if levelGrayScaleGradientImageSet(r,c) >= 22.5 && levelGrayScaleGradientImageSet(r,c) <= 67.5
                % main diagonal
                levelSpatiallyFilteredImage(r,c,:) = (levelSpatiallyFilteredImage(r-2,c-2,:) + levelSpatiallyFilteredImage(r-1,c-1,:)...
                    + levelSpatiallyFilteredImage(r,c,:) + levelSpatiallyFilteredImage(r+1,c+1,:) + levelSpatiallyFilteredImage(r+2,c+2,:)) / 5;
            end
            if levelGrayScaleGradientImageSet(r,c) > 67.5 && levelGrayScaleGradientImageSet(r,c) <= 112.5
                % horizontal
                levelSpatiallyFilteredImage(r,c,:) = (levelSpatiallyFilteredImage(r,c-2,:) + levelSpatiallyFilteredImage(r,c-1,:)...
                    + levelSpatiallyFilteredImage(r,c,:) + levelSpatiallyFilteredImage(r,c+1,:) + levelSpatiallyFilteredImage(r,c+2,:)) / 5;
            end
            if levelGrayScaleGradientImageSet(r,c) >= 0 && levelGrayScaleGradientImageSet(r,c) < 22.5
                % second diagonal
                levelSpatiallyFilteredImage(r,c,:) = (levelSpatiallyFilteredImage(r-2,c+2,:) + levelSpatiallyFilteredImage(r-1,c+1,:)...
                    + levelSpatiallyFilteredImage(r,c,:) + levelSpatiallyFilteredImage(r+1,c-1,:) + levelSpatiallyFilteredImage(r+2,c-2,:)) / 5;
            end
        end
    end
    refPyramid{level} = differenceImage .* levelSpatiallyFilteredImage...
        + (1 - differenceImage) .* imresize(refPyramid{level - 1}, [size(refPyramid{level},1), size(refPyramid{level},2)], 'bilinear');
end
