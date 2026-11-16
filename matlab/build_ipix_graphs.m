function build_ipix_graphs(raw_dir, out_dir, ids, polarizations)
% Convert IPIX CDF recordings to MATLAB v7 graph files for train.py.
arguments
    raw_dir (1,1) string
    out_dir (1,1) string
    ids (1,:) double = [17 26 30 31 40 54 280 310 311 320]
    polarizations (1,:) string {mustBeMember(polarizations,["hh","hv","vh","vv"])} = ["hh","hv","vh","vv"]
end

for id = ids
    [filename, affected, primary] = file_chioce(id);
    source = fullfile(raw_dir, filename);
    assert(isfile(source), 'Missing IPIX recording: %s', source);
    adc = single(ncread(source, 'adc_data'));
    assert(isequal(size(adc), [4 14 2 131072]), 'Unexpected IPIX dimensions in %s', source);
    adc(adc < 0) = adc(adc < 0) + 256;
    prf = double(ncread(source, 'PRF'));

    window = 64;
    n_bins = size(adc, 2);
    n_windows = size(adc, 4) / window;
    n_nodes = n_bins * n_windows;
    data_e_2 = grid_edges(n_windows, n_bins);
    degree = single(accumarray(double(data_e_2(2, :))', 1, [n_nodes 1]));
    range_bin = uint8(repelem((1:n_bins)', n_windows));
    window_id = uint16(repmat((1:n_windows)', n_bins, 1));
    label = uint8(range_bin == primary);
    guard_bins = setdiff([affected{:}], primary);
    rng(42, 'twister');
    mask_train = false(n_nodes, 1);
    for bin = setdiff(1:n_bins, guard_bins)
        mask_train((bin-1)*n_windows + randperm(n_windows, round(0.7*n_windows))) = true;
    end
    mask_test = ~mask_train & ~ismember(range_bin, guard_bins);

    for pol = polarizations
        tx = 1 + double(startsWith(pol, 'v'));
        rx = 1 + 2*double(extractBetween(pol, 1, 1) ~= extractBetween(pol, 2, 2));
        i = double(squeeze(adc(rx, :, tx, :))');
        q = double(squeeze(adc(rx+1, :, tx, :))');
        i = (i - mean(i(:,1))) / std(i(:,1));
        q = (q - mean(q(:,1))) / std(q(:,1));
        imbalance = 2*mean(i(:,1).*q(:,1));
        assert(abs(imbalance) < 1, 'Invalid I/Q phase compensation in %s/%s', filename, pol);
        iq = complex((i - q*imbalance) / sqrt(1-imbalance^2), q);
        data_x = single(abs(fftshift(fft(reshape(iq, window, []), [], 1), 1)).');
        assert(all(isfinite(data_x), 'all'), 'Non-finite FFT features in %s/%s', filename, pol);

        % The original preprocessing notes use cosine similarity of FFT magnitudes.
        normalized = data_x ./ max(vecnorm(data_x, 2, 2), eps('single'));
        edge_weight = min(1, max(0, sum(normalized(data_e_2(1,:),:) .* normalized(data_e_2(2,:),:), 2)));
        folder = fullfile(out_dir, string(id), pol);
        if ~isfolder(folder), mkdir(folder); end
        save(fullfile(folder, 'graph.mat'), 'data_x', 'data_e_2', 'edge_weight', 'degree', ...
             'label', 'mask_train', 'mask_test', 'range_bin', 'window_id', 'primary', ...
             'guard_bins', 'prf', 'window', '-v7');
        fprintf('#%d/%s: %d nodes, %d directed edges, train=%d, test=%d\n', ...
            id, pol, n_nodes, size(data_e_2,2), nnz(mask_train), nnz(mask_test));
    end
end
end

function edges = grid_edges(n_windows, n_bins)
% Linear order is range bin first, with consecutive windows inside each bin.
ids = reshape(uint32(1:n_windows*n_bins), n_windows, n_bins);
offsets = [1 0; 0 1; 1 1; -1 1];
pairs = cell(1, size(offsets, 1));
for k = 1:size(offsets, 1)
    dt = offsets(k, 1);
    dr = offsets(k, 2);
    times = (1-min(0,dt)):(n_windows-max(0,dt));
    bins = 1:(n_bins-dr);
    src = ids(times,bins);
    dst = ids(times+dt,bins+dr);
    pairs{k} = [src(:)'; dst(:)'];
end
forward = [pairs{:}];
edges = [forward, flipud(forward)];
end
