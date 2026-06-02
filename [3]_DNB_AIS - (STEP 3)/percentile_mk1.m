root = "/Volumes/SAMSUNG/JPSS-2_VIIRS/VJ202DNB"
files = dir(fullfile(root, "**", "*.nc"));

figure; hold on; grid on
ax = gca;
set(gca, 'YScale', 'log')

p_all =  0:1:100;

for k = 1:10 %numel(files)
    fpath = fullfile(files(k).folder, files(k).name);
    X = ncread(fpath, "/observation_data/DNB_observations");
    X = double(X);
    X = X(:);
    X = X(~isnan(X) & X > 0);

    if isempty(X)
        continue
    end

    % 왼쪽: RAW (log)
    yyaxis left
    plot(p_all, prctile(X, p_all), '-', 'LineWidth', 0.5);
    ax.YAxis(1).Scale = 'log';
    ylabel('Radiance')

    % 오른쪽: sigmoid 값 (0~1)
    yyaxis right
    plot(p_all, prctile(sigmoid_log(X), p_all), '-', 'LineWidth', 0.5);
    % ax.YAxis(2).Scale = 'log';
    ylim([0 1])
    ylabel('Sigmoid value')
end

xlabel('Percentile (%)')
title('RAW vs sigmoid (all files)')
