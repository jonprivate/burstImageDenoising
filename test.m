function test()
close all; clear all;
% first, import two images
image1 = imread('./Baby/5.jpg');
image2 = imread('./Baby/6.jpg');

% second, compute pyramids for both the two images, group all layers of
% each pyramid into a cell called pyramid
pyramid1 = getPyramids(image1);
pyramid2 = getPyramids(image2);
baseImage1 = rgb2gray(pyramid1{1});
% t = [0.9 0.2 0; 0.2 1.2 0; 0 0 1];
% tt = maketform('projective', double(t));
% baseImage2 = imtransform(baseImage1, tt);
% figure; imshow(baseImage2);
baseImage2 = rgb2gray(pyramid2{1});

% third, detect feature points and perform matching on lowest layer
% use the points and matching correspondence to compute homography for each
% image node
POINTS1 = detectSURFFeatures(baseImage1);
POINTS2 = detectSURFFeatures(baseImage2);
% imshow(pyramid1{1});
% hold on
% plot(POINTS1(1 : 100));
[features1, validPoints1] = extractFeatures(baseImage1, POINTS1);
[features2, validPoints2] = extractFeatures(baseImage2, POINTS2);
[indexPairs,matchmetric] = matchFeatures(features1,features2);
matchedPts1 = validPoints1(indexPairs(:,1));
matchedPts2 = validPoints2(indexPairs(:,2));
figure; showMatchedFeatures(baseImage1, baseImage2, matchedPts1, matchedPts2);
title('Matched SURF points, including outliers');
[tform,inlierPts1,inlierPts2] = estimateGeometricTransform(matchedPts1, matchedPts2, 'projective');
figure; showMatchedFeatures(baseImage1, baseImage2, inlierPts1, inlierPts2);
title('Matched inlier points');

featuredPyramid1 = getFeaturedPyramid(pyramid1, inlierPts1);
featuredPyramid2 = getFeaturedPyramidWithRef(pyramid2, inlierPts2, featuredPyramid1);
homographyPyramid = getHomographyPyramid(pyramid1, inlierPts1, inlierPts2);

T = maketform('projective', double(tform.T));
Ir = imtransform(baseImage1, T);
figure; imshow(Ir); title('Recovered image');
% forth, refine homography
[~, layerNum] = size(homographyPyramid);
lambda = 0.1;
for level = 2:layerNum
    homographyLevel = homographyPyramid{level};
    [rows, cols] = size(homographyLevel);
    num = rows * cols;
    
    R = []; %zeros(num * 9, 1);
    D = zeros(num, num * 9);
    for r = 1 : rows
        for c = 1 : cols
            homography = homographyLevel(r,c).homographies;
            R = [R; homography(:)];
            cur_ind = (r - 1) * cols + c;
            count = 0;
            if r - 1 >= 1
                ind = (r - 2) * cols + c;
                D(cur_ind, (ind - 1) * 9 + 1: ind * 9) = -1;
                count = count + 1;
            end
            if r + 1 <= rows
                ind = r * cols + c;
                D(cur_ind, (ind - 1) * 9 + 1: ind * 9) = -1;
                count = count + 1;
            end
            if c - 1 >= 1
                ind = (r - 1) * cols + c - 1;
                D(cur_ind, (ind - 1) * 9 + 1: ind * 9) = -1;
                count = count + 1;
            end
            if c + 1 <= cols
                ind = (r - 1) * cols + c + 1;
                D(cur_ind, (ind - 1) * 9 + 1: ind * 9) = -1;
                count = count + 1;
            end
            D(cur_ind, (cur_ind - 1) * 9 + 1: cur_ind * 9) = count;
        end
    end
    
    A = eye(num * 9, num * 9) + lambda * (D' * D);
    b = 2 * R;
    
    H = A^(-1) * b;
    for r = 1 : rows
        for c = 1 : cols
            ind = (r - 1) * cols + c;
            homographyLevel(r,c).homographies = reshape(H((ind - 1) * 9 + 1: ind * 9), 3, 3);
        end
    end
end

% fifth, discretize homography to create homography flow
homographyFLow = discretizeHomography(tform.T, size(baseImage1, 1), size(baseImage2, 2));

end

%%
function pyramid = getPyramids(image)
minRows = 400;
minCols = 400;
[rows, cols, ~] = size(image);
layerNum = 1;
while rows > minRows && cols > minCols
    layerNum = layerNum + 1;
    rows = ceil(rows / 2);
    cols = ceil(cols / 2);
end
pyramid = cell(1, layerNum);
current = image;
pyramid{layerNum} = current;
for level = layerNum - 1 : -1 : 1
    current = impyramid(current, 'reduce');
    pyramid{level} = current;
end

end

function homographyFLow = discretizeHomography(homography, rows, cols)
% The matrix T uses the convention:
% [x y 1] = [u v 1] * T
% where T has the form:
% [a b c;...
%  d e f;...
%  g h i];
homographyFLow = zeros(rows, cols, 2);
for x = 1 : cols
    for y = 1 : rows
        pt = [x, y, 1];
        pt_ = pt * homography;
        pt_ = pt_ / pt_(3);
        homographyFLow(y, x, 1) = pt_(1) - pt(1);
        homographyFLow(y, x, 2) = pt_(2) - pt(2);
    end
end
end

function flag = inRange(range, pt)
flag = false;
if pt(1) >= range(1,1) && pt(1) < range(1,2) && pt(2) >= range(2,1) && pt(2) < range(2,2)
    flag = true;
end
end

function featuredPyramid = getFeaturedPyramid(pyramid, pts)
[~, layerNum] = size(pyramid);
featuredPyramid = cell(1, layerNum);
for level = 1:layerNum
    rows = 2 ^ (level - 1);
    cols = rows;
    multi = rows;
    nodes = cell(rows, cols);
    % node: range, featurePoints, homography
    nodePixelNumVert = floor(size(pyramid{level}, 1) / multi);
    nodePixelNumHori = floor(size(pyramid{level}, 2) / multi);
    newPts = pts;
    newPts.Scale = newPts.Scale * multi;
    newPts.Location = newPts.Location * multi;
    for r = 1 : rows
        for c = 1 : cols
            range = [1 + (c - 1) * nodePixelNumHori, c * nodePixelNumHori; 1 + (r - 1) * nodePixelNumVert, r * nodePixelNumVert];
            inds = [];
            for i = 1 : length(newPts)
                if inRange(range, newPts(i).Location)
                    inds = [inds, i];
                end
            end
            nodePts = newPts(inds);
            node = ImageNode(range, nodePts, inds);
            nodes(r, c) = {node};
        end
    end
    featuredPyramid(level) = {nodes};
end
end

function featuredPyramid = getFeaturedPyramidWithRef(pyramid, pts, refFeaturedPyramid)
[~, layerNum] = size(pyramid);
featuredPyramid = cell(1, layerNum);
for level = 1:layerNum
    rows = 2 ^ (level - 1);
    cols = rows;
    multi = rows;
    nodes = cell(rows, cols);
    refNodes = refFeaturedPyramid{level};
    % node: range, featurePoints, homography
    nodePixelNumVert = floor(size(pyramid{level}, 1) / multi);
    nodePixelNumHori = floor(size(pyramid{level}, 2) / multi);
    newPts = pts;
    newPts.Scale = newPts.Scale * multi;
    newPts.Location = newPts.Location * multi;
    for r = 1 : rows
        for c = 1 : cols
            range = [1 + (c - 1) * nodePixelNumHori, c * nodePixelNumHori; 1 + (r - 1) * nodePixelNumVert, r * nodePixelNumVert];
            refNode = refNodes{r,c};
            node = ImageNode(range, newPts(refNode.inds), refNode.inds);
            nodes(r,c) = {node};
        end
    end
    featuredPyramid(level) = {nodes};
end
end

function homographyPyramid = getHomographyPyramid(pyramid, matchedPoints1, matchedPoints2)
featuredPyramid1 = getFeaturedPyramid(pyramid, matchedPoints1);
featuredPyramid2 = getFeaturedPyramidWithRef(pyramid, matchedPoints2, featuredPyramid1);
[~, layerNum] = size(pyramid);
homographyPyramid = cell(1, layerNum);
for level = 1:layerNum
    nodes1 = featuredPyramid1{level};
    nodes2 = featuredPyramid2{level};
    [rows, cols] = size(nodes1);
    homographies = cell(rows, cols);
    for r = 1:rows
        for c = 1:cols
            [tform,inlierPts1,inlierPts2] = estimateGeometricTransform(nodes1{r,c}.pts, nodes2{r,c}.pts, 'projective');
            if length(inlierPts1) < 40 && level > 1% minimum number of points
                r_upper = ceil(r / 2);
                c_upper = ceil(c / 2);
                temp = homographyPyramid{level - 1};
                homographies(r,c) = {temp(r_upper, c_upper).homographies};
            else
                homographies(r,c) = {tform.T};
            end
        end
    end
    pointNum = length(inlierPts1);
    current = struct('homographies', homographies, 'pointNumber', pointNum);
    homographyPyramid(level) = {current};
end
end