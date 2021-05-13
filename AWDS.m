function score = AWDS(img1, img2)

[M, N] = size(img1);
img1 = double(img1);
img2 = double(img2);

lpf = ones(2,2);
lpf = lpf/sum(lpf(:));
% sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
% ksize = 1 + floor(6*sigma)

dx = [1 0 -1; 2 0 -2; 1 0 -1]/4;
dy = dx';

c = 0.0025 * 65535;
a = -10;
ref_GM0 = sqrt(conv2(img1, dx, 'same').^2 + conv2(img1, dy, 'same').^2);
dis_GM0 = sqrt(conv2(img2, dx, 'same').^2 + conv2(img2, dy, 'same').^2);
mu1 = imgaussfilt(ref_GM0, 4.1);
mu2 = imgaussfilt(dis_GM0, 4.1);
weight_map = max(mu1,mu2);
qualityMap = (2 * ref_GM0.* dis_GM0 + c) ./ ( ref_GM0.^2 + dis_GM0.^2 + c);
mean0 = sum(qualityMap.*weight_map)/sum(weight_map);

ref_b = imgaussfilt(ref_GM0, 1.1);
ref_GMB = sqrt(conv2(ref_b, dx, 'same').^2 + conv2(ref_b, dy, 'same').^2);
mu1 = imgaussfilt(ref_GM0, 1.1);
mu2 = imgaussfilt(ref_GMB, 1.1);
weight_map = max(mu1,mu2);
qualityMap = ((2-a) * ref_GM0.* ref_GMB + c) ./ ( ref_GM0.^2 + ref_GMB.^2 - a * ref_GM0.* ref_GMB + c);
GDoG = sum(qualityMap.*weight_map)/sum(weight_map);
GDoG = 1 / (1 + exp(2*(97.49502237 * GDoG - 90.12996552)));

img1 = imfilter(img1, lpf, 'symmetric', 'same');
img1 = img1(1:2:end, 1:2:end);
img2 = imfilter(img2, lpf, 'symmetric', 'same');
img2 = img2(1:2:end, 1:2:end);
ref_GM1 = sqrt(conv2(img1, dx, 'same').^2 + conv2(img1, dy, 'same').^2);
dis_GM1 = sqrt(conv2(img2, dx, 'same').^2 + conv2(img2, dy, 'same').^2);
mu1 = imgaussfilt(ref_GM1, 4.1);
mu2 = imgaussfilt(dis_GM1, 4.1);
weight_map = max(mu1,mu2);
qualityMap = ((2-a) * ref_GM1.* dis_GM1 + c) ./ ( ref_GM1.^2 + dis_GM1.^2 - a * ref_GM1.* dis_GM1 + c);
mean1 = sum(qualityMap.*weight_map, 'all')/sum(weight_map, 'all');

score = mean0 * (1 - GDoG) + GDoG * (mean1^4);
