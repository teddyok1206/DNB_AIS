root = "/Volumes/SAMSUNG/JPSS-2_VIIRS/VJ202DNB"
files = dir(fullfile(root, "**", "*.nc"));
N = numel(files);

P = 0:1:100;
Q = nan(N, numel(P));

nValid = zeros(N,1);
fname = strings(N,1);

for i = 1:10 %N
    fpath = fullfile(files(i).folder, files(i).name);
    fname(i) = files(i).name;
    x = ncread(fpath, "/observation_data/DNB_observations");
    x = double(x(:));

    % 예: 0은 NoData, 또는 NaN 제거 등
    x = x(isfinite(x));         % NaN/Inf 제거
    % x = x(x ~= 0);            % 필요시 NoData 제거
    nValid(i) = numel(x);

    if nValid(i) > 0
        Q(i,:) = prctile(x, P);
    end
end

save("/Users/jungtaeuk/Desktop/SATGEO/DNB_AIS/[5]_DNB_AIS - (STEP 3)/percentiles.mat", "P","Q","nValid","fname", "-v7.3");