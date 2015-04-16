function consistentImageSet = getConsistentImageSet(grayScaleImageSet, homographySet)
consistentImageSet = zeros(size(grayScaleImageSet));
rows = size(grayScaleImageSet, 1);
cols = size(grayScaleImageSet, 2);
for i = 1 : size(grayScaleImageSet, 3)
    if isempty(homographySet{i})
        consistentImageSet(:,:,i) = grayScaleImageSet(:,:,i);
        continue;
    end
    homographyFlow = floor(homographySet{i});
    for r = 1 : rows
        for c = 1 : cols
            ri = min(max(1, r + homographyFlow(r,c,2)), rows);
            ci = min(max(1, c + homographyFlow(r,c,1)), cols);
            consistentImageSet(r,c,i) = grayScaleImageSet(ri,ci,i);
        end
    end
end