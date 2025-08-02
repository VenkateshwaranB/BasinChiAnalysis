%% COMPLETE BASIN KSN AND CHI ANALYSIS WITH INTERACTIVE CHI PROFILER
% This script calculates KSN and Chi values for the entire basin and includes
% interactive chi profiler functionality for detailed stream analysis
% This code modified and edited by venkateshwaran
% This code adopted from Topographic-analysis-toolkit (TAK) [https://github.com/amforte/Topographic-Analysis-Kit/tree/master] and
% Chi-profiler (Author: Sean F. Gallen (sean.gallen[at]erdw.ethz.ch))

clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIGURATION PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demtxt = '../proje/mnjlr.tif'; % name of your DEM tiff file with extension
fileTag = 'mnjlr'; % tag used to identify specific files
crita = 1e6; % threshold drainage area for channel head initiation
mn = 0.45; % m/n or concavity index (theta)
Ao = 1; % reference drainage area for chi-analysis
smoWin = 250; % size of window (in map units) used to smooth elevation data
flowOption = 'fill'; % Option for flow routing. Either 'carve' or 'fill'

% Output folder - modify this path as needed
output_folder = 'E:\Mycomputer_onedrive\results\mnjlr';

% Add TopoToolbox to path
addpath(genpath('E:\Mycomputer_onedrive\topotoolbox-master'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1: CREATE STREAM NETWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Step 1: Creating stream network...');
try
    [DEM,FD,A,S] = MakeStreams(demtxt, crita, 'no_data_exp', 'DEM==0 | DEM>10000');
    disp('✓ Stream network created successfully');
catch ME
    fprintf('Error creating stream network: %s\n', ME.message);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 2: CALCULATE BASIN-WIDE KSN AND CHI VALUES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Step 2: Calculating basin-wide KSN and Chi values...');
try
    % Calculate KSN
    [KSNG, ksn_ms] = KsnChiBatch(DEM, FD, A, S, 'ksn', ...
        'output', true, ...
        'ref_concavity', mn, ...
        'smooth_distance', smoWin, ...
        'file_name_prefix', fileTag);
    
    % Calculate Chi
    [ChiMap, ChiGrid, chi_ms] = KsnChiBatch(DEM, FD, A, S, 'chi', ...
        'output', true, ...
        'ref_concavity', mn, ...
        'smooth_distance', smoWin, ...
        'file_name_prefix', fileTag);
    
    % Export to Excel
    export_to_excel(ksn_ms, fullfile(output_folder, [fileTag '_KSN_Analysis.xlsx']), 'KSN_Data');
    export_to_excel(chi_ms, fullfile(output_folder, [fileTag '_Chi_Analysis.xlsx']), 'Chi_Data');
    
    disp('✓ Basin-wide analysis completed and exported to Excel');
    
catch ME
    fprintf('Error in basin-wide analysis: %s\n', ME.message);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3: INTERACTIVE CHI PROFILER ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Step 3: Starting interactive chi profiler analysis...');
disp('=== CHI PROFILER ANALYSIS ===');

% Ask user if they want to do interactive analysis
txt = sprintf('\nWould you like to perform interactive chi profiler analysis?');
disp(txt);
txt = sprintf('This allows detailed regression analysis and knickpoint identification.\n');
disp(txt);
interactive_opt = input('\ntype "y" for yes or "n" for no:  ','s');

while ~strcmp(interactive_opt,'y') && ~strcmp(interactive_opt,'n')
    disp('Yes(y) or No(n) only!!!');
    interactive_opt = input('type "y" for yes or "n" for no:  ','s');
end

if strcmp(interactive_opt, 'y')
    try
        [chiFits, kp_data, regression_data] = run_interactive_chi_profiler(DEM, FD, A, S, fileTag, crita, mn, Ao, smoWin, output_folder);
        
        % Export interactive analysis results
        if ~isempty(chiFits)
            export_chi_fits(chiFits, output_folder, fileTag);
        end
        if ~isempty(kp_data)
            export_knickpoint_data(kp_data, output_folder, fileTag);
        end
        
        disp('✓ Interactive chi profiler analysis completed');
        
    catch ME
        fprintf('Error in interactive analysis: %s\n', ME.message);
    end
else
    chiFits = [];
    kp_data = [];
    regression_data = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 4: CREATE COMPREHENSIVE VISUALIZATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Step 4: Creating comprehensive visualizations...');
try
    create_all_plots(DEM, FD, S, A, ksn_ms, chi_ms, ChiMap, ChiGrid, KSNG, chiFits, kp_data, output_folder, fileTag, mn);
    
    % Create interactive analysis summary plot if data exists
    if ~isempty(chiFits) && ~isempty(kp_data)
        create_interactive_summary_plot(chiFits, kp_data, regression_data, output_folder, fileTag);
    end
    
    disp('✓ All visualizations created');
catch ME
    fprintf('Error creating plots: %s\n', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 5: CREATE SUMMARY ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Step 5: Creating summary analysis...');
try
    create_comprehensive_summary(ksn_ms, chi_ms, chiFits, kp_data, DEM, S, output_folder, fileTag);
    disp('✓ Summary analysis created');
catch ME
    fprintf('Error creating summary: %s\n', ME.message);
end

disp('=== ANALYSIS COMPLETE ===');
fprintf('Results saved to: %s\n', output_folder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN FUNCTION DEFINITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function streams_to_analyze = get_streams_to_analyze(num_streams)
    % Get user input for which streams to analyze
    
    fprintf('\nOptions for stream analysis:\n');
    fprintf('1. Analyze all streams (1-%d)\n', num_streams);
    fprintf('2. Analyze specific streams\n');
    fprintf('3. Analyze a range of streams\n');
    
    choice = input('Enter your choice (1, 2, or 3): ');
    
    switch choice
        case 1
            streams_to_analyze = 1:num_streams;
            fprintf('Selected: All %d streams\n', num_streams);
            
        case 2
            fprintf('Enter stream numbers separated by spaces (e.g., 1 3 5 7): ');
            stream_input = input('', 's');
            streams_to_analyze = str2num(stream_input); %#ok<ST2NM>
            streams_to_analyze = streams_to_analyze(streams_to_analyze >= 1 & streams_to_analyze <= num_streams);
            fprintf('Selected streams: %s\n', num2str(streams_to_analyze));
            
        case 3
            start_stream = input(sprintf('Enter start stream (1-%d): ', num_streams));
            end_stream = input(sprintf('Enter end stream (1-%d): ', num_streams));
            start_stream = max(1, min(start_stream, num_streams));
            end_stream = max(start_stream, min(end_stream, num_streams));
            streams_to_analyze = start_stream:end_stream;
            fprintf('Selected streams: %d to %d\n', start_stream, end_stream);
            
        otherwise
            fprintf('Invalid choice. Analyzing first 5 streams by default.\n');
            streams_to_analyze = 1:min(5, num_streams);
    end
end

function Schi = calculate_chi_for_streams(S, A, GridID, Ao, mn)
    % Calculate chi values for stream network
    
    Schi = zeros(size(S.distance));
    Six = S.ix;
    Sixc = S.ixc;
    Sd = S.distance;
    Sa = (Ao ./ A.Z(GridID)).^mn;
    
    h = waitbar(0, 'Calculating χ for stream network...');
    for lp = numel(Six):-1:1
        Schi(Six(lp)) = Schi(Sixc(lp)) + (Sa(Sixc(lp)) + (Sa(Six(lp)) - Sa(Sixc(lp)))/2) * abs(Sd(Sixc(lp)) - Sd(Six(lp)));
        f = (numel(Six) + 1 - lp) / numel(Six);
        waitbar(f, h);
    end
    close(h);
end

function dataMat = prepare_stream_data(strmInds, Sdfd, Sz, Sda, Sd, smoWin, cs, DEM, GridID, Sx, Sy, Schi)
    % Prepare data matrix for stream analysis
    
    dataMat = nan(length(strmInds), 11);
    dataMat(:,1) = Sdfd(strmInds);              % distance from divide
    dataMat(:,2) = Sz(strmInds);                % elevation
    dataMat(:,3) = Sda(strmInds);               % drainage area
    dataMat(:,4) = Sd(strmInds);                % distance from mouth
    dataMat(:,5) = smoothChannelZ(dataMat(:,2), smoWin, cs); % smoothed elevation
    
    [mrows, mcols] = ind2sub(DEM.size, GridID(strmInds));
    dataMat(:,6) = mrows;                       % row in matrix
    dataMat(:,7) = mcols;                       % column in matrix
    dataMat(:,8) = GridID(strmInds);            % grid index
    dataMat(:,9) = Sx(strmInds);                % x coordinate
    dataMat(:,10) = Sy(strmInds);               % y coordinate
    dataMat(:,11) = Schi(strmInds);             % chi values
end


function create_binned_ksn_plot(chi, z, step, Ao, mn, profile_fig, minchi, maxchi)
    % Create binned ksn plot
    
    incs = floor(length(chi)/step);
    if incs < 2, return; end
    
    spt = 1; ept = step;
    mp = nan(incs, 1);
    binKsn = nan(incs, 1);
    
    for i = 1:incs
        if ept > length(chi), ept = length(chi); end
        mp(i) = nanmean(chi(spt:ept));
        chi_segMat = [ones(size(chi(spt:ept))) chi(spt:ept)];
        try
            [b, ~, ~, ~, ~] = regress(z(spt:ept), chi_segMat, 0.05);
            binKsn(i) = b(2) * Ao^mn;
        catch
            binKsn(i) = NaN;
        end
        spt = spt + step;
        ept = ept + step;
    end
    
    figure(profile_fig);
    subplot(3,1,3);
    valid_idx = ~isnan(binKsn);
    if sum(valid_idx) >= 2
        plot(mp(valid_idx), binKsn(valid_idx), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
        xlabel('\chi (m)'); ylabel(['k_{sn} (m^{' num2str(2*mn) '})']);
        axis([minchi, maxchi + (maxchi - minchi) * 0.1, 0, nanmax(binKsn) * 1.1]);
    end
end

function chiFits = perform_chi_regressions(chi, z, dist, x_coord, y_coord, Ao, mn, strmID, profile_fig, map_fig)
    % Perform chi regressions with user interaction
    
    chiFits = [];
    seg = 0;
    continue_regression = true;
    
    while continue_regression
        fprintf('\n--- Regression Segment %d ---\n', seg + 1);
        fprintf('Click on MINIMUM then MAXIMUM chi bounds on the chi-elevation plot (middle plot)\n');
        fprintf('Include at least 3 data points\n');
        
        figure(profile_fig);
        subplot(3,1,2);
        [chiP, ~] = ginput(2);
        
        while chiP(1,1) >= chiP(2,1)
            fprintf('Follow the directions! Click MINIMUM chi (left) then MAXIMUM chi (right)\n');
            [chiP, ~] = ginput(2);
        end
        
        min_chi = chiP(1,1);
        max_chi = chiP(2,1);
        
        % Perform regression
        [chiKsn, ksnUC, R2, regBounds, reg_plots] = perform_single_regression(chi, z, dist, x_coord, y_coord, min_chi, max_chi, Ao, mn, seg, profile_fig, map_fig);
        
        if ~isempty(chiKsn)
            fprintf('\nRegression Results:\n');
            fprintf('K_sn = %.2f ± %.2f\n', chiKsn, ksnUC);
            fprintf('R² = %.3f\n', R2);
            
            % Ask if user wants to save this fit
            fit_opt = input('Do you want to remember this fit? (y/n): ', 's');
            
            if strcmp(fit_opt, 'y')
                newdata = [strmID, seg+1, chiKsn, ksnUC, chiKsn/Ao^mn, ksnUC/Ao^mn, R2, regBounds, x_coord(end), y_coord(end)];
                chiFits = [chiFits; newdata];
                seg = seg + 1;
            else
                % Delete the regression plots
                delete(reg_plots);
            end
        end
        
        % Ask about another segment
        fit_opt2 = input('Do you want to fit another channel segment? (y/n): ', 's');
        continue_regression = strcmp(fit_opt2, 'y');
    end
    
    if isempty(chiFits)
        noDatVect = -9999 * ones(1, 12);
        chiFits = [strmID, noDatVect];
    end
end

function [chiKsn, ksnUC, R2, regBounds, reg_plots] = perform_single_regression(chi, z, dist, x_coord, y_coord, min_chi, max_chi, Ao, mn, seg, profile_fig, map_fig)
    % Perform a single chi regression
    
    try
        % Select data range
        ind = find(chi >= min_chi & chi <= max_chi);
        if length(ind) < 3
            fprintf('Not enough data points for regression\n');
            chiKsn = []; ksnUC = []; R2 = []; regBounds = []; reg_plots = [];
            return;
        end
        
        chi_seg = chi(ind);
        z_seg = z(ind);
        dist_seg = dist(ind);
        x_seg = x_coord(ind);
        y_seg = y_coord(ind);
        
        regBounds = [min(chi_seg), max(chi_seg), min(z_seg), max(z_seg)];
        
        % Regression
        chi_segMat = [ones(size(z_seg)) chi_seg];
        [b, bint, ~, ~, stats] = regress(z_seg, chi_segMat, 0.05);
        
        % Calculate results
        chiSlope = b(2);
        chiKsn = chiSlope * Ao^mn;
        UnCert = (bint(4) - bint(2)) / 2;
        ksnUC = UnCert * Ao^mn;
        R2 = stats(1);
        
        % Create model line
        ymod = b(2) * chi_seg + b(1);
        
        % Plot regression
        figure(profile_fig);
        subplot(3,1,1);
        reg_plot1 = plot(dist_seg/1000, ymod, 'r--', 'LineWidth', 2);
        hold on;
        plot(dist_seg([1 end])/1000, ymod([1 end]), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
        
        subplot(3,1,2);
        reg_plot2 = plot(chi_seg, ymod, 'r--', 'LineWidth', 2);
        hold on;
        plot(chi_seg([1 end]), ymod([1 end]), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
        
        subplot(3,1,3);
        ck = chiKsn * ones(size(chi_seg));
        reg_plot3 = plot(chi_seg, ck, 'r--', 'LineWidth', 2);
        hold on;
        plot(chi_seg([1 end]), chiKsn * [1 1], 'rx', 'MarkerSize', 8, 'LineWidth', 2);
        
        figure(map_fig);
        reg_plot4 = plot(x_seg, y_seg, 'r--', 'LineWidth', 3);
        hold on;
        plot(x_seg([1 end]), y_seg([1 end]), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        
        reg_plots = [reg_plot1; reg_plot2; reg_plot3; reg_plot4];
        
        % Add text annotation
        figure(profile_fig);
        subplot(3,1,2);
        text(mean(chi_seg), mean(z_seg), ...
            {['Segment ' num2str(seg+1)], ['K_{sn} = ' num2str(chiKsn, '%.1f') ' ± ' num2str(ksnUC, '%.1f')], ['R² = ' num2str(R2, '%.3f')]}, ...
            'FontSize', 8, 'BackgroundColor', 'white', 'EdgeColor', 'black');
        
    catch ME
        fprintf('Regression failed: %s\n', ME.message);
        chiKsn = []; ksnUC = []; R2 = []; regBounds = []; reg_plots = [];
    end
end

function kp_data = identify_knickpoints(chi, zr, A, dist, data, x_coord, y_coord, strmID, profile_fig, map_fig)
    % Identify knickpoints interactively
    
    kp_data = [];
    kp = 0;
    continue_kp = true;
    
    dfd = data(:,1);
    sel = data(:,5);
    xmat = data(:,6);
    ymat = data(:,7);
    
    while continue_kp
        fprintf('\n--- Knickpoint %d ---\n', kp + 1);
        fprintf('Click on a point on the chi-elevation plot (middle plot)\n');
        
        figure(profile_fig);
        subplot(3,1,2);
        [chi_kp, ~] = ginput(1);
        
        kp_ind = find(chi <= chi_kp, 1, 'first');
        if isempty(kp_ind)
            fprintf('Point not within bounds. Try again.\n');
            continue;
        end
        
        % Plot knickpoint
        kp_p2 = plot(chi(kp_ind), sel(kp_ind), 'kv', 'MarkerFaceColor', [150/255 150/255 255/255], 'MarkerSize', 10);
        
        subplot(3,1,1);
        kp_p1 = plot(dist(kp_ind)/1000, sel(kp_ind), 'kv', 'MarkerFaceColor', [150/255 150/255 255/255], 'MarkerSize', 10);
        
        figure(map_fig);
        kp_p3 = plot(x_coord(kp_ind), y_coord(kp_ind), 'ko', 'MarkerFaceColor', [0.8 0.8 0.8], 'MarkerSize', 10);
        
        figure(profile_fig);
        
        % Ask if user wants to save this point
        fit_opt = input('Do you want to remember this point? (y/n): ', 's');
        
        if strcmp(fit_opt, 'y')
            fprintf('Classify the knickpoint type:\n');
            fprintf('1 = Major knickpoint\n');
            fprintf('2 = Minor knickpoint\n');
            fprintf('3 = Lithologic contact\n');
            fprintf('4 = Other\n');
            kp_class = input('Enter classification number: ');
            
            % Add data to table
            newdata = [strmID, kp+1, kp_class, chi(kp_ind), zr(kp_ind), A(kp_ind), ...
                      dist(kp_ind), dfd(kp_ind), sel(kp_ind), x_coord(kp_ind), ...
                      y_coord(kp_ind), xmat(kp_ind), ymat(kp_ind), ...
                      x_coord(end), y_coord(end)];
            kp_data = [kp_data; newdata];
            kp = kp + 1;
        else
            delete([kp_p1, kp_p2, kp_p3]);
        end
        
        % Ask about another point
        fit_opt2 = input('Do you want to select another point? (y/n): ', 's');
        continue_kp = strcmp(fit_opt2, 'y');
    end
end

function export_chi_fits(chiFits, output_folder, fileTag)
    % Export chi regression fits to Excel
    
    try
        if isempty(chiFits) || all(chiFits(:,2) == -9999)
            fprintf('No regression data to export\n');
            return;
        end
        
        % Remove no-data entries
        valid_rows = chiFits(:,2) ~= -9999;
        chiFits = chiFits(valid_rows, :);
        
        if isempty(chiFits)
            return;
        end
        
        % Create table
        stream_ID = chiFits(:,1);
        segment_num = chiFits(:,2);
        ksn = chiFits(:,3);
        ksn_95uc = chiFits(:,4);
        chi_slope = chiFits(:,5);
        slope_uc = chiFits(:,6);
        r_squared = chiFits(:,7);
        min_chi = chiFits(:,8);
        max_chi = chiFits(:,9);
        min_elev = chiFits(:,10);
        max_elev = chiFits(:,11);
        outlet_x = chiFits(:,12);
        outlet_y = chiFits(:,13);
        
        T = table(stream_ID, segment_num, ksn, ksn_95uc, chi_slope, slope_uc, ...
                 r_squared, min_chi, max_chi, min_elev, max_elev, outlet_x, outlet_y);
        
        filename = fullfile(output_folder, [fileTag, '_Chi_Regressions.xlsx']);
        writetable(T, filename);
        fprintf('✓ Chi regression data exported to: %s\n', filename);
        
    catch ME
        fprintf('Error exporting chi fits: %s\n', ME.message);
    end
end

function export_knickpoint_data(kp_data, output_folder, fileTag)
    % Export knickpoint data to Excel and shapefile
    
    try
        if isempty(kp_data)
            fprintf('No knickpoint data to export\n');
            return;
        end
        
        % Create Excel table
        X = kp_data(:,10);
        Y = kp_data(:,11);
        strm_num = kp_data(:,1);
        kp_num = kp_data(:,2);
        kp_type = kp_data(:,3);
        chi = kp_data(:,4);
        elev = kp_data(:,5);
        d_area = kp_data(:,6);
        dfm = kp_data(:,7);
        dfd = kp_data(:,8);
        smo_el = kp_data(:,9);
        GridX = kp_data(:,12);
        GridY = kp_data(:,13);
        outletX = kp_data(:,14);
        outletY = kp_data(:,15);
        
        T = table(X, Y, strm_num, kp_num, kp_type, chi, elev, smo_el, ...
                 d_area, dfm, dfd, GridX, GridY, outletX, outletY);
        
        filename = fullfile(output_folder, [fileTag, '_Knickpoints.xlsx']);
        writetable(T, filename);
        
        % Create shapefile
        MP = struct('Geometry', {'Point'}, ...
                   'X', num2cell(X), ...
                   'Y', num2cell(Y), ...
                   'strm_num', num2cell(strm_num), ...
                   'kp_num', num2cell(kp_num), ...
                   'kp_type', num2cell(kp_type), ...
                   'chi', num2cell(chi), ...
                   'elev', num2cell(elev), ...
                   'smo_el', num2cell(smo_el), ...
                   'd_area', num2cell(d_area), ...
                   'dfm', num2cell(dfm), ...
                   'dfd', num2cell(dfd));
        
        shp_filename = fullfile(output_folder, [fileTag, '_Knickpoints.shp']);
        shapewrite(MP, shp_filename);
        
        fprintf('✓ Knickpoint data exported to Excel and shapefile\n');
        
    catch ME
        fprintf('Error exporting knickpoint data: %s\n', ME.message);
    end
end

function create_interactive_summary_plot(chiFits, kp_data, regression_data, output_folder, fileTag)
    % Create summary plot of interactive analysis results
    
    try
        figure('Position', [100, 100, 1400, 1000]);
        
        % Load and plot all analyzed streams
        folder = regression_data.folder;
        analyzed_streams = regression_data.analyzed_streams;
        
        subplot(2,1,1);
        hold on;
        colors = lines(length(analyzed_streams));
        
        % Plot each analyzed stream
        for i = 1:length(analyzed_streams)
            stream_num = analyzed_streams(i);
            try
                dataFileName = [num2str(stream_num), '_', fileTag, '_chandata.mat'];
                load(fullfile(folder, dataFileName), 'dataMat');
                
                dist_data = dataMat(:,4) / 1000; % Convert to km
                elev_data = dataMat(:,5);
                
                plot(dist_data, elev_data, '-', 'Color', colors(i,:), 'LineWidth', 1.5, ...
                     'DisplayName', ['Stream ' num2str(stream_num)]);
            catch
                continue;
            end
        end
        
        % Plot knickpoints on distance plot
        if ~isempty(kp_data)
            scatter(kp_data(:,7)/1000, kp_data(:,9), 100, 'kv', 'filled', ...
                   'MarkerEdgeColor', 'k', 'DisplayName', 'Knickpoints');
        end
        
        xlabel('Distance from Mouth (km)');
        ylabel('Elevation (m)');
        title(['Distance-Elevation Profiles with Knickpoints - ' fileTag], 'FontSize', 14);
        legend('Location', 'best');
        grid on;
        
        % Chi-elevation plot
        subplot(2,1,2);
        hold on;
        
        % Plot each analyzed stream
        for i = 1:length(analyzed_streams)
            stream_num = analyzed_streams(i);
            try
                dataFileName = [num2str(stream_num), '_', fileTag, '_chandata.mat'];
                load(fullfile(folder, dataFileName), 'dataMat');
                
                chi_data = dataMat(:,11);
                elev_data = dataMat(:,5);
                
                plot(chi_data, elev_data, '-', 'Color', colors(i,:), 'LineWidth', 1.5, ...
                     'DisplayName', ['Stream ' num2str(stream_num)]);
            catch
                continue;
            end
        end
        
        % Plot knickpoints on chi plot
        if ~isempty(kp_data)
            scatter(kp_data(:,4), kp_data(:,9), 100, 'kv', 'filled', ...
                   'MarkerEdgeColor', 'k', 'DisplayName', 'Knickpoints');
        end
        
        % Plot regression lines
        if ~isempty(chiFits) && any(chiFits(:,2) ~= -9999)
            valid_fits = chiFits(chiFits(:,2) ~= -9999, :);
            for i = 1:size(valid_fits, 1)
                min_chi = valid_fits(i, 8);
                max_chi = valid_fits(i, 9);
                min_elev = valid_fits(i, 10);
                max_elev = valid_fits(i, 11);
                
                plot([min_chi, max_chi], [min_elev, max_elev], 'r--', 'LineWidth', 3, ...
                     'HandleVisibility', 'off');
            end
            plot(NaN, NaN, 'r--', 'LineWidth', 3, 'DisplayName', 'Chi Regressions');
        end
        
        xlabel('\chi (m)');
        ylabel('Elevation (m)');
        title(['Chi-Elevation Profiles with Regressions - ' fileTag], 'FontSize', 14);
        legend('Location', 'best');
        grid on;
        
        % Save the plot
        saveas(gcf, fullfile(output_folder, [fileTag '_Interactive_Analysis_Summary.png']), 'png');
        saveas(gcf, fullfile(output_folder, [fileTag '_Interactive_Analysis_Summary.fig']), 'fig');
        close(gcf);
        
    catch ME
        fprintf('Error creating interactive summary plot: %s\n', ME.message);
    end
end

function smoothed_z = smoothChannelZ(z, smoWin, cs)
    % Smooth channel elevation data
    
    try
        window_size = round(smoWin / cs);
        if window_size < 3
            window_size = 3;
        end
        if window_size > length(z)
            window_size = length(z);
        end
        
        % Make window size odd
        if mod(window_size, 2) == 0
            window_size = window_size + 1;
        end
        
        smoothed_z = movmean(z, window_size, 'omitnan');
    catch
        smoothed_z = z; % Return original if smoothing fails
    end
end

% Include all the other functions from the previous version
% (KsnChiBatch, calculate_ksn, calculate_chi, etc.)

function [varargout] = KsnChiBatch(DEM,FD,A,S,product,varargin)
    % Fixed version of KsnChiBatch that handles Chi calculation errors
    
    % Parse Inputs
    p = inputParser;         
    p.FunctionName = 'KsnChiBatch';
    addRequired(p,'DEM',@(x) isa(x,'GRIDobj'));
    addRequired(p,'FD', @(x) isa(x,'FLOWobj'));
    addRequired(p,'A', @(x) isa(x,'GRIDobj'));
    addRequired(p,'S',@(x) isa(x,'STREAMobj'));
    addRequired(p,'product',@(x) ischar(validatestring(x,{'ksn','chi','chimap','chigrid'})));

    addParameter(p,'file_name_prefix','analysis',@(x) ischar(x));
    addParameter(p,'smooth_distance',1000,@(x) isscalar(x) && isnumeric(x));
    addParameter(p,'ref_concavity',0.50,@(x) isscalar(x) && isnumeric(x));
    addParameter(p,'output',false,@(x) isscalar(x) && islogical(x));
    addParameter(p,'complete_networks_only',true,@(x) isscalar(x) && islogical(x));
    addParameter(p,'interp_value',0.1,@(x) isnumeric(x) && x>=0 && x<=1);

    parse(p,DEM,FD,A,S,product,varargin{:});
    
    % Extract parameters
    file_name_prefix = p.Results.file_name_prefix;
    segment_length = p.Results.smooth_distance;
    theta_ref = p.Results.ref_concavity;
    output = p.Results.output;
    cno = p.Results.complete_networks_only;
    iv = p.Results.interp_value;

    % Clean up stream network
    if cno
        S = removeedgeeffects(S, FD, DEM);
    end

    % Condition DEM
    disp('Conditioning DEM...');
    zc = mincosthydrocon(S, DEM, 'interp', iv);
    DEMc = GRIDobj(DEM);
    DEMc.Z(DEMc.Z==0) = NaN;
    DEMc.Z(S.IXgrid) = zc;

    switch product
        case 'ksn'
            [varargout{1:nargout}] = calculate_ksn(DEM, DEMc, A, S, theta_ref, segment_length, output);
            
        case 'chi'
            [varargout{1:nargout}] = calculate_chi(DEM, FD, A, S, DEMc, theta_ref, segment_length, output);
            
        case 'chimap'
            [varargout{1:nargout}] = calculate_chimap(DEM, FD, A, S, theta_ref, output);
            
        case 'chigrid'
            [varargout{1:nargout}] = calculate_chigrid(DEM, FD, theta_ref, output);
    end
end

function [varargout] = calculate_ksn(DEM, DEMc, A, S, theta_ref, segment_length, output)
    disp('Calculating KSN values...');
    
    % Calculate gradient
    g = gradient(S, DEMc);
    G = GRIDobj(DEM);
    G.Z(S.IXgrid) = g;

    % Calculate cut/fill
    Z_RES = DEMc - DEM;

    % Calculate KSN
    ksn = G ./ (A .* (A.cellsize^2)).^(-theta_ref);

    % Calculate stream distance
    SD = GRIDobj(DEM);
    SD.Z(S.IXgrid) = S.distance;
    
    % Create mapstructure
    try
        ksn_ms = STREAMobj2mapstruct(S, 'seglength', segment_length, 'attributes', ...
            {'ksn' ksn @mean 'uparea' (A.*(A.cellsize^2)) @mean 'gradient' G @mean 'cut_fill' Z_RES @mean...
            'min_dist' SD @min 'max_dist' SD @max});

        % Add segment distance
        seg_dist = [ksn_ms.max_dist] - [ksn_ms.min_dist];
        distcell = num2cell(seg_dist');
        [ksn_ms(1:end).seg_dist] = distcell{:};
        ksn_ms = rmfield(ksn_ms, {'min_dist', 'max_dist'});
        
        % Add additional useful fields
        for i = 1:length(ksn_ms)
            ksn_ms(i).stream_length = seg_dist(i);
            try
                ksn_ms(i).elevation_start = mean(DEM.Z(coord2ind(DEM, ksn_ms(i).X(1), ksn_ms(i).Y(1))));
                ksn_ms(i).elevation_end = mean(DEM.Z(coord2ind(DEM, ksn_ms(i).X(end), ksn_ms(i).Y(end))));
            catch
                ksn_ms(i).elevation_start = NaN;
                ksn_ms(i).elevation_end = NaN;
            end
        end
        
    catch ME
        fprintf('Error in KSN mapstructure creation: %s\n', ME.message);
        ksn_ms = [];
    end

    if output
        % Create grid
        KSNG = GRIDobj(DEM);
        KSNG.Z(:,:) = NaN;
        if ~isempty(ksn_ms)
            for ii = 1:numel(ksn_ms)
                try
                    ix = coord2ind(DEM, ksn_ms(ii).X, ksn_ms(ii).Y);
                    KSNG.Z(ix) = ksn_ms(ii).ksn;
                catch
                    continue;
                end
            end
        end
        varargout{1} = KSNG;
        varargout{2} = ksn_ms;
    else
        varargout{1} = ksn_ms;
    end
end

function [varargout] = calculate_chi(DEM, FD, A, S, DEMc, theta_ref, segment_length, output)
    disp('Calculating Chi values...');
    
    try
        % Calculate chi map
        C = chitransform(S, A, 'mn', theta_ref, 'a0', 1);
        ChiMap = GRIDobj(DEM);
        ChiMap.Z(ChiMap.Z==0) = NaN;
        ChiMap.Z(S.IXgrid) = C;
        
        % Calculate chi grid using simpler method
        disp('Calculating chi grid...');
        ChiGrid = calculate_chi_grid_simple(DEM, FD, A, theta_ref);
        
        % Create chi mapstructure
        chi_ms = create_chi_mapstructure(DEM, DEMc, A, S, theta_ref, segment_length);
        
    catch ME
        fprintf('Error in chi calculation: %s\n', ME.message);
        ChiMap = GRIDobj(DEM);
        ChiMap.Z(:,:) = NaN;
        ChiGrid = GRIDobj(DEM);
        ChiGrid.Z(:,:) = NaN;
        chi_ms = [];
    end

    if output
        varargout{1} = ChiMap;
        varargout{2} = ChiGrid;
        varargout{3} = chi_ms;
    else
        varargout{1} = chi_ms;
    end
end

function [ChiGrid] = calculate_chi_grid_simple(DEM, FD, A, theta_ref)
    % Simplified chi grid calculation
    
    try
        DA = A .* (DEM.cellsize^2);
        ChiGrid = GRIDobj(DEM);
        ChiGrid.Z(:,:) = 0;
        
        D = flowdistance(FD);
        chi_vals = D.Z .* (DA.Z).^(-theta_ref);
        chi_vals(isnan(DEM.Z)) = NaN;
        chi_vals(isinf(chi_vals)) = NaN;
        
        ChiGrid.Z = chi_vals;
        
    catch ME
        fprintf('Simple chi grid calculation failed: %s\n', ME.message);
        ChiGrid = GRIDobj(DEM);
        ChiGrid.Z(:,:) = NaN;
    end
end

function [chi_ms] = create_chi_mapstructure(DEM, DEMc, A, S, theta_ref, segment_length)
    % Create chi mapstructure with error handling
    
    try
        g = gradient(S, DEMc);
        G = GRIDobj(DEM);
        G.Z(S.IXgrid) = g;

        Z_RES = DEMc - DEM;
        chi = chitransform(S, A, 'mn', theta_ref, 'a0', 1);
        
        SD = GRIDobj(DEM);
        SD.Z(S.IXgrid) = S.distance;
        
        chi_ms = STREAMobj2mapstruct(S, 'seglength', segment_length, 'attributes', ...
            {'chi' chi @mean 'uparea' (A.*(A.cellsize^2)) @mean 'gradient' G @mean 'cut_fill' Z_RES @mean...
            'min_dist' SD @min 'max_dist' SD @max});

        seg_dist = [chi_ms.max_dist] - [chi_ms.min_dist];
        distcell = num2cell(seg_dist');
        [chi_ms(1:end).seg_dist] = distcell{:};
        chi_ms = rmfield(chi_ms, {'min_dist', 'max_dist'});
        
        for i = 1:length(chi_ms)
            chi_ms(i).stream_length = seg_dist(i);
            try
                chi_ms(i).elevation_start = mean(DEM.Z(coord2ind(DEM, chi_ms(i).X(1), chi_ms(i).Y(1))));
                chi_ms(i).elevation_end = mean(DEM.Z(coord2ind(DEM, chi_ms(i).X(end), chi_ms(i).Y(end))));
            catch
                chi_ms(i).elevation_start = NaN;
                chi_ms(i).elevation_end = NaN;
            end
        end
        
    catch ME
        fprintf('Error creating chi mapstructure: %s\n', ME.message);
        chi_ms = [];
    end
end

function export_to_excel(data_struct, filename, sheet_name)
    % Export structure array to Excel with error handling
    
    if isempty(data_struct)
        fprintf('No data to export for %s\n', filename);
        return;
    end
    
    try
        [filepath, ~, ~] = fileparts(filename);
        if ~exist(filepath, 'dir')
            mkdir(filepath);
        end
        
        data_table = struct2table(data_struct);
        
        if ismember('Geometry', data_table.Properties.VariableNames)
            data_table.Geometry = [];
        end
        if ismember('BoundingBox', data_table.Properties.VariableNames)
            data_table.BoundingBox = [];
        end
        if ismember('X', data_table.Properties.VariableNames)
            data_table.X = [];
        end  
        if ismember('Y', data_table.Properties.VariableNames)
            data_table.Y = [];
        end
        
        writetable(data_table, filename, 'Sheet', sheet_name);
        fprintf('✓ Data exported to: %s\n', filename);
        
    catch ME
        fprintf('Error exporting to Excel: %s\n', ME.message);
        try
            save([filename(1:end-5) '.mat'], 'data_struct');
            fprintf('✓ Data saved as MAT file instead\n');
        catch
            fprintf('Failed to save data in any format\n');
        end
    end
end

function create_comprehensive_summary(ksn_ms, chi_ms, chiFits, kp_data, DEM, S, output_folder, fileTag)
    % Create comprehensive summary including interactive results
    
    try
        summary = struct();
        summary.analysis_date = datestr(now);
        summary.dem_info.cellsize = DEM.cellsize;
        summary.dem_info.size = DEM.size;
        summary.dem_info.extent = DEM.extent;
        summary.stream_info.total_length = sum(S.distance);
        summary.stream_info.num_segments = numel(S.orderednanlist) - sum(isnan(S.orderednanlist));
        
        % Basin-wide statistics
        if ~isempty(ksn_ms)
            summary.basin_ksn_stats.mean = nanmean([ksn_ms.ksn]);
            summary.basin_ksn_stats.median = nanmedian([ksn_ms.ksn]);
            summary.basin_ksn_stats.std = nanstd([ksn_ms.ksn]);
            summary.basin_ksn_stats.min = nanmin([ksn_ms.ksn]);
            summary.basin_ksn_stats.max = nanmax([ksn_ms.ksn]);
            summary.basin_ksn_stats.num_segments = length(ksn_ms);
        end
        
        if ~isempty(chi_ms)
            summary.basin_chi_stats.mean = nanmean([chi_ms.chi]);
            summary.basin_chi_stats.median = nanmedian([chi_ms.chi]);
            summary.basin_chi_stats.std = nanstd([chi_ms.chi]);
            summary.basin_chi_stats.min = nanmin([chi_ms.chi]);
            summary.basin_chi_stats.max = nanmax([chi_ms.chi]);
            summary.basin_chi_stats.num_segments = length(chi_ms);
        end
        
        % Interactive analysis statistics
        if ~isempty(chiFits) && any(chiFits(:,2) ~= -9999)
            valid_fits = chiFits(chiFits(:,2) ~= -9999, :);
            summary.regression_stats.num_regressions = size(valid_fits, 1);
            summary.regression_stats.mean_ksn = nanmean(valid_fits(:,3));
            summary.regression_stats.mean_r2 = nanmean(valid_fits(:,7));
            summary.regression_stats.streams_analyzed = unique(valid_fits(:,1));
        else
            summary.regression_stats.num_regressions = 0;
        end
        
        if ~isempty(kp_data)
            summary.knickpoint_stats.total_knickpoints = size(kp_data, 1);
            summary.knickpoint_stats.streams_with_kp = length(unique(kp_data(:,1)));
            kp_types = unique(kp_data(:,3));
            for i = 1:length(kp_types)
                type_count = sum(kp_data(:,3) == kp_types(i));
                summary.knickpoint_stats.(['type_' num2str(kp_types(i))]) = type_count;
            end
        else
            summary.knickpoint_stats.total_knickpoints = 0;
        end
        
        % Save summary
        save(fullfile(output_folder, [fileTag '_Comprehensive_Summary.mat']), 'summary');
        
        % Create combined Excel summary
        if ~isempty(ksn_ms) && ~isempty(chi_ms)
            summary_table = table(...
                (1:length(ksn_ms))', ...
                [ksn_ms.ksn]', ...
                [chi_ms.chi]', ...
                [ksn_ms.uparea]', ...
                [ksn_ms.gradient]', ...
                'VariableNames', {'Segment_ID', 'KSN', 'Chi', 'Drainage_Area', 'Gradient'});
            
            writetable(summary_table, fullfile(output_folder, [fileTag '_Complete_Basin_Analysis.xlsx']), ...
                'Sheet', 'Basin_Wide_Data');
        end
        
        fprintf('✓ Comprehensive summary created\n');
        
    catch ME
        fprintf('Error creating comprehensive summary: %s\n', ME.message);
    end
end

function create_all_plots(DEM, FD, S, A, ksn_ms, chi_ms, ChiMap, ChiGrid, KSNG, chiFits, kp_data, output_folder, fileTag, theta_ref)
    % Create all visualization plots including interactive results
    
    try
        % Plot 1: KSN Map
        if ~isempty(ksn_ms) && exist('KSNG', 'var') && ~isempty(KSNG)
            PlotKsn(DEM, FD, ksn_ms);
            title(['KSN Map - ' fileTag], 'FontSize', 14, 'FontWeight', 'bold');
            saveas(gcf, fullfile(output_folder, [fileTag '_KSN_Map.png']), 'png');
            saveas(gcf, fullfile(output_folder, [fileTag '_KSN_Map.fig']), 'fig');
            close(gcf);
        end
        
        % Plot 2: Chi Map
        if exist('ChiMap', 'var') && ~isempty(ChiMap)
            PlotChi(DEM, S, ChiMap, 'chimap');
            title(['Chi Map - ' fileTag], 'FontSize', 14, 'FontWeight', 'bold');
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Map.png']), 'png');
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Map.fig']), 'fig');
            close(gcf);
        end
        
        % Plot 3: Chi Grid
        if exist('ChiGrid', 'var') && ~isempty(ChiGrid)
            PlotChi(DEM, S, ChiGrid, 'chigrid');
            title(['Chi Grid - ' fileTag], 'FontSize', 14, 'FontWeight', 'bold');
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Grid.png']), 'png');
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Grid.fig']), 'fig');
            close(gcf);
        end
        
        % Plot 4: Chi-Elevation Profile
        if ~isempty(chi_ms)
            create_chi_elevation_plot(DEM, S, A, chi_ms, output_folder, fileTag, theta_ref);
        end
        
        % Plot 5: Combined Analysis Dashboard
        create_analysis_dashboard(DEM, FD, S, ksn_ms, chi_ms, chiFits, kp_data, output_folder, fileTag);
        
        % Plot 6: Interactive Results Summary
        if ~isempty(chiFits) || ~isempty(kp_data)
            create_interactive_results_plot(chiFits, kp_data, output_folder, fileTag);
        end
        
    catch ME
        fprintf('Error in plot creation: %s\n', ME.message);
    end
end

function create_chi_elevation_plot(DEM, S, A, chi_ms, output_folder, fileTag, theta_ref)
    
    try
        figure('Position', [100, 100, 1200, 800]);
        
        % Extract chi and elevation data
        chi_vals = [];
        elev_vals = [];
        
        for i = 1:length(chi_ms)
            if ~isempty(chi_ms(i).X) && ~isempty(chi_ms(i).Y)
                try
                    seg_ix = coord2ind(DEM, chi_ms(i).X, chi_ms(i).Y);
                    seg_elev = DEM.Z(seg_ix);
                    seg_chi = ones(length(seg_ix), 1) * chi_ms(i).chi;
                    
                    valid_idx = ~isnan(seg_elev) & ~isnan(seg_chi);
                    chi_vals = [chi_vals; seg_chi(valid_idx)];
                    elev_vals = [elev_vals; seg_elev(valid_idx)];
                catch
                    continue;
                end
            end
        end
        
        if ~isempty(chi_vals) && ~isempty(elev_vals)
            % Plot tributaries (light blue)
            scatter(chi_vals, elev_vals, 10, [0.7, 0.9, 1], 'filled', 'MarkerEdgeColor', 'none');
            hold on;
            
            % Find and plot main stem
            try
                main_stem_data = extract_main_stem(S, A, DEM, chi_ms);
                if ~isempty(main_stem_data)
                    plot(main_stem_data.chi, main_stem_data.elevation, 'b-', 'LineWidth', 4);
                end
            catch
                [sorted_chi, sort_idx] = sort(chi_vals);
                sorted_elev = elev_vals(sort_idx);
                plot(sorted_chi, sorted_elev, 'b-', 'LineWidth', 2, 'Color', [0, 0.4, 0.8]);
            end
            
            % Add equilibrium line
            chi_range = [min(chi_vals), max(chi_vals)];
            p = polyfit(chi_vals, elev_vals, 1);
            eq_line_chi = linspace(chi_range(1), chi_range(2), 100);
            eq_line_elev = polyval(p, eq_line_chi);
            plot(eq_line_chi, eq_line_elev, 'r--', 'LineWidth', 2);
            
            % Formatting
            xlabel('\chi [m]', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Elevation [m]', 'FontSize', 12, 'FontWeight', 'bold');
            title([fileTag ' Basin - Chi-Elevation Analysis'], 'FontSize', 14, 'FontWeight', 'bold');
            
            legend({'Tributaries', 'Main River', 'Equilibrium Line'}, ...
                   'Location', 'northwest', 'FontSize', 10);
            
            grid on;
            grid minor;
            set(gca, 'FontSize', 10);
            
            % Add parameter text box
            text_str = {['Concavity (\theta) = ' num2str(theta_ref)], ...
                       ['Reference Area = 1 m²'], ...
                       ['Total Segments = ' num2str(length(chi_ms))]};
            annotation('textbox', [0.15, 0.75, 0.2, 0.15], 'String', text_str, ...
                      'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
            
            % Save plots
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Elevation_Profile.png']), 'png');
            saveas(gcf, fullfile(output_folder, [fileTag '_Chi_Elevation_Profile.fig']), 'fig');
            print(fullfile(output_folder, [fileTag '_Chi_Elevation_Profile.pdf']), '-dpdf', '-r300');
            
            close(gcf);
        end
        
    catch ME
        fprintf('Error creating chi-elevation plot: %s\n', ME.message);
    end
end

function main_stem_data = extract_main_stem(S, A, DEM, chi_ms)
    % Extract main stem data for plotting
    
    try
        main_stem_data = struct();
        main_stem_data.chi = [];
        main_stem_data.elevation = [];
        
        if ~isempty(chi_ms)
            areas = [chi_ms.uparea];
            [~, high_area_idx] = sort(areas, 'descend');
            
            n_main = max(1, round(0.2 * length(high_area_idx)));
            main_indices = high_area_idx(1:n_main);
            
            for i = main_indices
                if ~isempty(chi_ms(i).X) && ~isempty(chi_ms(i).Y)
                    try
                        seg_ix = coord2ind(DEM, chi_ms(i).X, chi_ms(i).Y);
                        seg_elev = DEM.Z(seg_ix);
                        seg_chi = ones(length(seg_ix), 1) * chi_ms(i).chi;
                        
                        valid_idx = ~isnan(seg_elev) & ~isnan(seg_chi);
                        main_stem_data.chi = [main_stem_data.chi; seg_chi(valid_idx)];
                        main_stem_data.elevation = [main_stem_data.elevation; seg_elev(valid_idx)];
                    catch
                        continue;
                    end
                end
            end
            
            if ~isempty(main_stem_data.chi)
                [main_stem_data.chi, sort_idx] = sort(main_stem_data.chi);
                main_stem_data.elevation = main_stem_data.elevation(sort_idx);
            end
        else
            main_stem_data = [];
        end
        
    catch
        main_stem_data = [];
    end
end

function create_analysis_dashboard(DEM, FD, S, ksn_ms, chi_ms, chiFits, kp_data, output_folder, fileTag)
    % Create comprehensive analysis dashboard
    
    try
        figure('Position', [50, 50, 1600, 1200]);
        
        % Subplot 1: DEM with streams and knickpoints
        subplot(2, 3, 1);
        imageschs(DEM, DEM, 'colormap', 'gray');
        hold on;
        plot(S, 'b-', 'LineWidth', 1);
        if ~isempty(kp_data)
            scatter(kp_data(:,10), kp_data(:,11), 50, 'rv', 'filled');
        end
        title('DEM with Streams & Knickpoints', 'FontSize', 12, 'FontWeight', 'bold');
        
        % Subplot 2: KSN Distribution
        subplot(2, 3, 2);
        if ~isempty(ksn_ms)
            ksn_values = [ksn_ms.ksn];
            histogram(ksn_values, 20, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'black');
            xlabel('KSN Values', 'FontSize', 10);
            ylabel('Frequency', 'FontSize', 10);
            title('Basin-wide KSN Distribution', 'FontSize', 12, 'FontWeight', 'bold');
            
            mean_ksn = mean(ksn_values, 'omitnan');
            std_ksn = std(ksn_values, 'omitnan');
            text(0.6, 0.8, sprintf('Mean: %.2f\nStd: %.2f', mean_ksn, std_ksn), ...
                 'Units', 'normalized', 'FontSize', 9, 'BackgroundColor', 'white');
        else
            text(0.5, 0.5, 'No KSN Data', 'HorizontalAlignment', 'center');
        end
        
        % Subplot 3: Chi Distribution  
        subplot(2, 3, 3);
        if ~isempty(chi_ms)
            chi_values = [chi_ms.chi];
            histogram(chi_values, 20, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'black');
            xlabel('Chi Values [m]', 'FontSize', 10);
            ylabel('Frequency', 'FontSize', 10);
            title('Basin-wide Chi Distribution', 'FontSize', 12, 'FontWeight', 'bold');
            
            mean_chi = mean(chi_values, 'omitnan');
            std_chi = std(chi_values, 'omitnan');
            text(0.6, 0.8, sprintf('Mean: %.2f\nStd: %.2f', mean_chi, std_chi), ...
                 'Units', 'normalized', 'FontSize', 9, 'BackgroundColor', 'white');
        else
            text(0.5, 0.5, 'No Chi Data', 'HorizontalAlignment', 'center');
        end
        
        % Subplot 4: Interactive Analysis Results
        subplot(2, 3, 4);
        if ~isempty(chiFits) && any(chiFits(:,2) ~= -9999)
            valid_fits = chiFits(chiFits(:,2) ~= -9999, :);
            regression_ksn = valid_fits(:,3);
            r2_values = valid_fits(:,7);
            
            scatter(r2_values, regression_ksn, 60, 'go', 'filled');
            xlabel('R² Values', 'FontSize', 10);
            ylabel('Regression K_{sn}', 'FontSize', 10);
            title('Chi Regression Results', 'FontSize', 12, 'FontWeight', 'bold');
            grid on;
            
            text(0.05, 0.95, sprintf('Regressions: %d', length(regression_ksn)), ...
                 'Units', 'normalized', 'FontSize', 9, 'BackgroundColor', 'white');
        else
            text(0.5, 0.5, 'No Regression Data', 'HorizontalAlignment', 'center');
        end
        
        % Subplot 5: Knickpoint Analysis
        subplot(2, 3, 5);
        if ~isempty(kp_data)
            kp_types = kp_data(:,3);
            kp_elevations = kp_data(:,5);
            
            unique_types = unique(kp_types);
            colors = lines(length(unique_types));
            
            for i = 1:length(unique_types)
                type_idx = kp_types == unique_types(i);
                scatter(kp_data(type_idx,4), kp_elevations(type_idx), 60, colors(i,:), 'filled', ...
                       'DisplayName', ['Type ' num2str(unique_types(i))]);
                hold on;
            end
            
            xlabel('Chi [m]', 'FontSize', 10);
            ylabel('Elevation [m]', 'FontSize', 10);
            title('Knickpoint Distribution', 'FontSize', 12, 'FontWeight', 'bold');
            legend('Location', 'best', 'FontSize', 8);
            grid on;
        else
            text(0.5, 0.5, 'No Knickpoint Data', 'HorizontalAlignment', 'center');
        end
        
        % Subplot 6: Summary Statistics
        subplot(2, 3, 6);
        axis off;
        
        summary_text = {[fileTag ' Basin Analysis Summary'], '', ...
                       ['DEM Resolution: ' num2str(DEM.cellsize) ' m'], ...
                       ['DEM Size: ' num2str(DEM.size(1)) ' × ' num2str(DEM.size(2))], ...
                       ['Total Stream Length: ' sprintf('%.2f km', sum(S.distance)/1000)], ''};
        
        if ~isempty(ksn_ms)
            summary_text = [summary_text, ...
                           ['Basin KSN Segments: ' num2str(length(ksn_ms))], ...
                           ['Mean Basin KSN: ' sprintf('%.2f', mean([ksn_ms.ksn], 'omitnan'))], ''];
        end
        
        if ~isempty(chi_ms)
            summary_text = [summary_text, ...
                           ['Chi Segments: ' num2str(length(chi_ms))], ...
                           ['Mean Chi: ' sprintf('%.2f m', mean([chi_ms.chi], 'omitnan'))], ''];
        end
        
        if ~isempty(chiFits) && any(chiFits(:,2) ~= -9999)
            valid_fits = chiFits(chiFits(:,2) ~= -9999, :);
            summary_text = [summary_text, ...
                           ['Interactive Regressions: ' num2str(size(valid_fits,1))], ...
                           ['Mean Regression K_{sn}: ' sprintf('%.2f', mean(valid_fits(:,3)))], ''];
        end
        
        if ~isempty(kp_data)
            summary_text = [summary_text, ...
                           ['Total Knickpoints: ' num2str(size(kp_data,1))], ...
                           ['Streams with KPs: ' num2str(length(unique(kp_data(:,1))))]];
        end
        
        text(0.1, 0.9, summary_text, 'Units', 'normalized', 'FontSize', 10, ...
             'VerticalAlignment', 'top', 'FontWeight', 'normal');
        
        sgtitle([fileTag ' - Comprehensive Analysis Dashboard'], 'FontSize', 16, 'FontWeight', 'bold');
        
        % Save dashboard
        saveas(gcf, fullfile(output_folder, [fileTag '_Complete_Analysis_Dashboard.png']), 'png');
        saveas(gcf, fullfile(output_folder, [fileTag '_Complete_Analysis_Dashboard.fig']), 'fig');
        close(gcf);
        
    catch ME
        fprintf('Error creating analysis dashboard: %s\n', ME.message);
    end
end

function create_interactive_results_plot(chiFits, kp_data, output_folder, fileTag)
    % Create detailed plot of interactive analysis results
    
    try
        figure('Position', [200, 200, 1400, 600]);
        
        % Plot regression results
        subplot(1, 2, 1);
        if ~isempty(chiFits) && any(chiFits(:,2) ~= -9999)
            valid_fits = chiFits(chiFits(:,2) ~= -9999, :);
            
            streams = valid_fits(:,1);
            ksn_vals = valid_fits(:,3);
            r2_vals = valid_fits(:,7);
            
            % Create color map for different streams
            unique_streams = unique(streams);
            colors = lines(length(unique_streams));
            
            for i = 1:length(unique_streams)
                stream_idx = streams == unique_streams(i);
                scatter(ksn_vals(stream_idx), r2_vals(stream_idx), 100, colors(i,:), 'filled', ...
                       'DisplayName', ['Stream ' num2str(unique_streams(i))]);
                hold on;
            end
            
            xlabel('K_{sn} Values', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('R² Values', 'FontSize', 12, 'FontWeight', 'bold');
            title('Chi Regression Quality vs K_{sn}', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            
            % Add quality thresholds
            line([min(ksn_vals), max(ksn_vals)], [0.8, 0.8], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
            text(mean(ksn_vals), 0.82, 'Good Fit Threshold (R² = 0.8)', 'HorizontalAlignment', 'center', ...
                 'Color', 'r', 'FontWeight', 'bold');
        else
            text(0.5, 0.5, 'No Regression Data Available', 'HorizontalAlignment', 'center', ...
                 'FontSize', 14, 'Units', 'normalized');
        end
        
        % Plot knickpoint results
        subplot(1, 2, 2);
        if ~isempty(kp_data)
            streams = kp_data(:,1);
            kp_types = kp_data(:,3);
            chi_vals = kp_data(:,4);
            elevations = kp_data(:,5);
            
            unique_streams = unique(streams);
            colors = lines(length(unique_streams));
            
            % Define knickpoint type markers
            type_markers = {'o', 's', '^', 'v', 'd', 'p', 'h'};
            
            for i = 1:length(unique_streams)
                stream_idx = streams == unique_streams(i);
                stream_kp_types = kp_types(stream_idx);
                stream_chi = chi_vals(stream_idx);
                stream_elev = elevations(stream_idx);
                
                for j = 1:length(stream_kp_types)
                    marker_idx = min(stream_kp_types(j), length(type_markers));
                    scatter(stream_chi(j), stream_elev(j), 100, colors(i,:), type_markers{marker_idx}, ...
                           'filled', 'LineWidth', 1.5);
                    hold on;
                end
            end
            
            xlabel('\chi [m]', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Elevation [m]', 'FontSize', 12, 'FontWeight', 'bold');
            title('Knickpoint Distribution in Chi-Space', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            
            % Create legend
            legend_entries = {};
            for i = 1:length(unique_streams)
                legend_entries{end+1} = ['Stream ' num2str(unique_streams(i))];
            end
            legend(legend_entries, 'Location', 'best');
            
        else
            text(0.5, 0.5, 'No Knickpoint Data Available', 'HorizontalAlignment', 'center', ...
                 'FontSize', 14, 'Units', 'normalized');
        end
        
        sgtitle([fileTag ' - Interactive Analysis Results'], 'FontSize', 16, 'FontWeight', 'bold');
        
        % Save plot
        saveas(gcf, fullfile(output_folder, [fileTag '_Interactive_Results.png']), 'png');
        saveas(gcf, fullfile(output_folder, [fileTag '_Interactive_Results.fig']), 'fig');
        close(gcf);
        
    catch ME
        fprintf('Error creating interactive results plot: %s\n', ME.message);
    end
end

% Include plotting functions (PlotKsn and PlotChi) from previous version
function PlotKsn(DEM,FD,ksn,varargin)
    % Enhanced KSN plotting function
    
    if ischar(ksn)
        [~,~,ext]=fileparts(ksn);
    else
        ext=' ';
    end

    % Parse Inputs
    p = inputParser;
    p.FunctionName = 'PlotKsn';
    addRequired(p,'DEM',@(x) isa(x,'GRIDobj'));
    addRequired(p,'FD',@(x) isa(x,'FLOWobj'));
    addRequired(p,'ksn',@(x) isstruct(x) || strcmpi(ext,'.shp') || strcmpi(ext,'.txt') || isa(x,'GRIDobj'));
    addParameter(p,'knicks',[],@(x) isnumeric(x) || regexp(x,regexptranslate('wildcard','*.shp')) || istable(x));
    addParameter(p,'ksn_lim',[],@(x) isnumeric(x) && numel(x)==2);

    parse(p,DEM,FD,ksn,varargin{:});
    DEM=p.Results.DEM;
    FD=p.Results.FD;
    ksn=p.Results.ksn;
    knks=p.Results.knicks;
    ksn_lim=p.Results.ksn_lim;

    % Handle different input types
    if ischar(ksn) && logical(regexp(ksn,regexptranslate('wildcard','*.shp')))
        ksn=shaperead(ksn);
        grid_flag=false;
    elseif ischar(ksn) && logical(regexp(ksn,regexptranslate('wildcard','*.txt')))
        ksn=GRIDobj(ksn);
        if ~validatealignment(DEM,ksn)
            ksn=resample(ksn,DEM);
        end
        grid_flag=true;
    elseif isstruct(ksn)
        grid_flag=false;
    elseif isa(ksn,'GRIDobj')
        if ~validatealignment(DEM,ksn)
            ksn=resample(ksn,DEM);
        end
        grid_flag=true;
    else
        error('Input to "ksn" not recognized');
    end
    
    if grid_flag
        f1=figure();
        set(f1,'Units','normalized','Position',[0.05 0.1 0.8 0.8],'renderer','painters');
        hold on
        if isempty(ksn_lim)
            imageschs(DEM,ksn,'colormap','parula');
        else
            imageschs(DEM,ksn,'colormap','parula','caxis',ksn_lim);
        end
        colorbar;
        if ~verLessThan('matlab','9.5')
            disableDefaultInteractivity(gca);
        end 
        hold off
    else    
        num_seg=numel(ksn);
        sx=cell(num_seg,1);
        sy=cell(num_seg,1);
        sk=cell(num_seg,1);
        
        for ii=1:num_seg
            sx{ii,1}=ksn(ii,1).X(:);
            sy{ii,1}=ksn(ii,1).Y(:);
            if isfield(ksn,'ksn')
                sk{ii,1}=ones(numel(sx{ii,1}),1)*ksn(ii,1).ksn;
            elseif isfield(ksn,'fit_ksn')
                sk{ii,1}=ones(numel(sx{ii,1}),1)*ksn(ii,1).fit_ksn;
            else
                error('No valid KSN field found');
            end
        end

        sx=vertcat(sx{:});
        sy=vertcat(sy{:});
        sk=vertcat(sk{:});

        ix=coord2ind(DEM,sx,sy);
        idx=isnan(ix);
        ix(idx)=[];
        sk(idx)=[];

        W=GRIDobj(DEM,'logical');
        W.Z(ix)=true;
        S=STREAMobj(FD,W);

        [~,loc,~]=unique(ix);
        sk=sk(loc);

        f1=figure();
        set(f1,'Visible','off');

        [RGB]=imageschs(DEM,DEM,'colormap','gray');
        [~,R]=GRIDobj2im(DEM);

        imshow(flipud(RGB),R);
        axis xy
        hold on
        colormap(parula(20));
        plotc(S,sk);
        if isempty(ksn_lim)
            caxis([0 max(sk)]);
        else
            caxis([min(ksn_lim) max(ksn_lim)]);
        end
        c1=colorbar;
        ylabel(c1,'Normalized Channel Steepness (KSN)');
        
        if ~isempty(knks)
            if ischar(knks) && logical(regexp(knks,regexptranslate('wildcard','*.shp')))
                knk=shaperead(knks);
                knkx=[knk.X];
                knky=[knk.Y];
                scatter(knkx,knky,100,'w','p','filled','MarkerEdgeColor','k');
            elseif istable(knks)
                knkx=knks.x_coord;
                knky=knks.y_coord;
                scatter(knkx,knky,100,'w','p','filled','MarkerEdgeColor','k');
            else
                scatter(knks(:,1),knks(:,2),100,'w','p','filled','MarkerEdgeColor','k');
            end
        end
        if ~verLessThan('matlab','9.5')
            disableDefaultInteractivity(gca);
        end 
        hold off
        set(f1,'Visible','on','Units','normalized','Position',[0.05 0.1 0.8 0.8],'renderer','painters');
    end
end

function PlotChi(DEM,S,chi,chi_type,varargin)
    % Enhanced Chi plotting function
    
    % Parse Inputs
    p = inputParser;
    p.FunctionName = 'PlotChi';
    addRequired(p,'DEM',@(x) isa(x,'GRIDobj'));
    addRequired(p,'S',@(x) isa(x,'STREAMobj'));
    addRequired(p,'chi',@(x) isa(x,'GRIDobj') || regexp(x,regexptranslate('wildcard','*.txt')));
    addRequired(p,'chi_type',@(x) ischar(validatestring(x,{'chimap','chigrid'})));
    addParameter(p,'chi_lim',[],@(x) isnumeric(x) && numel(x)==2);
    addParameter(p,'override_resample',false,@(x) isscalar(x) && islogical(x));
    
    parse(p,DEM,S,chi,chi_type,varargin{:});
    DEM=p.Results.DEM;
    S=p.Results.S;
    chi=p.Results.chi;
    chi_type=p.Results.chi_type;
    chi_lim=p.Results.chi_lim;
    os=p.Results.override_resample;
    
    if ischar(chi) && logical(regexp(chi,regexptranslate('wildcard','*.txt')))
        chi=GRIDobj(chi);
        if ~validatealignment(DEM,chi) && ~os
            chi=resample(chi,DEM);
        elseif ~validatealignment(DEM,chi) && os
            chi.refmat=DEM.refmat;
            chi.georef=DEM.georef;
        end
    elseif isa(chi,'GRIDobj')
        if ~validatealignment(DEM,chi)
            chi=resample(chi,DEM);
        end
    else
        error('Input to "chi" not recognized');
    end
    
    switch chi_type
        case 'chigrid'
            f1=figure();
            set(f1,'Units','normalized','Position',[0.05 0.1 0.8 0.8],'renderer','painters');
            hold on
            if isempty(chi_lim)
                imageschs(DEM,chi,'colormap','jet');
            else
                imageschs(DEM,chi,'colormap','jet','caxis',chi_lim);
            end
            c1=colorbar;
            ylabel(c1,'\chi [m]');
            if ~verLessThan('matlab','9.5')
                disableDefaultInteractivity(gca);
            end 
            hold off
            
        case 'chimap'
            nal=getnal(S,chi);
            f1=figure();
            set(f1,'Visible','off');
            [RGB]=imageschs(DEM,DEM,'colormap','gray');
            [~,R]=GRIDobj2im(DEM);
            imshow(flipud(RGB),R);
            axis xy
            hold on
            colormap(jet);
            plotc(S,nal);
            if isempty(chi_lim)
                caxis([0 max(nal)]);
            else
                caxis([min(chi_lim) max(chi_lim)])
            end
            c1=colorbar;
            ylabel(c1,'\chi [m]');
            if ~verLessThan('matlab','9.5')
                disableDefaultInteractivity(gca);
            end 
            hold off
            set(f1,'Visible','on','Units','normalized','Position',[0.05 0.1 0.8 0.8],'renderer','painters');
    end
end
function [map_fig, profile_fig] = create_overview_plots_enhanced(DEM, S, strmBreaks, ordList, Sd, Schi, SmoZ, smoWin, cs, mindfm, maxdfm, minel, maxel, minchi, maxchi)
    % Enhanced overview plots showing ALL streams in gray with better styling
    
    % Map figure
    map_fig = figure('Position', [100, 400, 800, 600]);
    imageschs(DEM); 
    hold on;
    plot(S, '-', 'LineWidth', 1, 'Color', [0.6 0.6 0.6]); % Gray streams on map
    title('Stream Network Overview', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Profile figure with enhanced styling
    profile_fig = figure('Position', [950, 100, 800, 900]);
    set(profile_fig, 'Color', 'white');
    
    % Plot all streams as gray lines
    fprintf('Creating overview plots with %d streams...\n', length(strmBreaks));
    h = waitbar(0, 'Creating overview plots...');
    id1 = 0;
    
    % Store all stream data for equilibrium line calculation
    all_chi_data = [];
    all_elev_data = [];
    all_dist_data = [];
    
    for i = 1:length(strmBreaks)
        strmInds = ordList(id1+1:strmBreaks(i)-1);
        SmoZ(strmInds) = smoothChannelZ(SmoZ(strmInds), smoWin, cs);
        
        % Distance-Elevation plot (top)
        subplot(3,1,1);
        plot(Sd(strmInds)/1000, SmoZ(strmInds), '-', 'LineWidth', 0.5, 'Color', [0.7, 0.7, 0.7]); 
        hold on;
        
        % Chi-Elevation plot (middle) 
        subplot(3,1,2);
        plot(Schi(strmInds), SmoZ(strmInds), '-', 'LineWidth', 0.5, 'Color', [0.7, 0.7, 0.7]); 
        hold on;
        
        % Store data for equilibrium line calculation
        valid_idx = ~isnan(Schi(strmInds)) & ~isnan(SmoZ(strmInds));
        if sum(valid_idx) > 0
            all_chi_data = [all_chi_data; Schi(strmInds(valid_idx))];
            all_elev_data = [all_elev_data; SmoZ(strmInds(valid_idx))];
            all_dist_data = [all_dist_data; Sd(strmInds(valid_idx))/1000];
        end
        
        id1 = strmBreaks(i);
        f = i / length(strmBreaks);
        waitbar(f, h);
    end
    close(h);
    
    % Calculate and plot equilibrium lines
    if ~isempty(all_chi_data) && ~isempty(all_elev_data)
        % Chi-elevation equilibrium line
        p_chi = polyfit(all_chi_data, all_elev_data, 1);
        chi_eq_line = linspace(minchi, maxchi, 100);
        elev_eq_chi = polyval(p_chi, chi_eq_line);
        
        % Distance-elevation equilibrium line  
        p_dist = polyfit(all_dist_data, all_elev_data, 1);
        dist_eq_line = linspace(mindfm, maxdfm, 100);
        elev_eq_dist = polyval(p_dist, dist_eq_line);
        
        % Plot equilibrium lines
        subplot(3,1,1);
        plot(dist_eq_line, elev_eq_dist, 'r--', 'LineWidth', 2, 'DisplayName', 'Equilibrium Line');
        
        subplot(3,1,2);
        plot(chi_eq_line, elev_eq_chi, 'r--', 'LineWidth', 2, 'DisplayName', 'Equilibrium Line');
    end
    
    % Format subplots
    subplot(3,1,1);
    axis([mindfm, maxdfm + (maxdfm - mindfm) * 0.1, minel, maxel + (maxel - minel) * 0.1]);
    xlabel('Distance (km)', 'FontSize', 12, 'FontWeight', 'bold'); 
    ylabel('Elevation (m)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Stream %d of %d', 0, length(strmBreaks)), 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    set(gca, 'FontSize', 10);
    
    subplot(3,1,2);
    axis([minchi, maxchi + (maxchi - minchi) * 0.1, minel, maxel + (maxel - minel) * 0.1]);
    xlabel('\chi (m)', 'FontSize', 12, 'FontWeight', 'bold'); 
    ylabel('Elevation (m)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Chi-Elevation Profiles', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    set(gca, 'FontSize', 10);
    
    % Add third subplot placeholder with better formatting
    subplot(3,1,3);
    xlabel('\chi (m)', 'FontSize', 12, 'FontWeight', 'bold'); 
    ylabel('k_{sn}', 'FontSize', 12, 'FontWeight', 'bold');
    title('Binned K_{sn} Values', 'FontSize', 14, 'FontWeight', 'bold');
    grid on; grid minor;
    set(gca, 'FontSize', 10);
    hold on;
    
    % Add legend to chi-elevation plot
    subplot(3,1,2);
    if ~isempty(all_chi_data)
        legend('All Streams', 'Equilibrium Line', 'Location', 'northwest', 'FontSize', 10);
    end
end

function [s1, s2, s3] = highlight_current_stream_enhanced(profile_fig, map_fig, dataMat, stream_num, total_streams)
    % Enhanced highlighting with cyan/blue color like your reference image
    
    figure(profile_fig);
    
    % Distance-elevation plot with cyan highlighting
    subplot(3,1,1);
    s1 = plot(dataMat(:,4)/1000, dataMat(:,5), '-', 'LineWidth', 3, 'Color', [0, 0.8, 0.8]);
    title(sprintf('Stream %d of %d', stream_num, total_streams), 'FontSize', 14, 'FontWeight', 'bold');
    
    % Chi-elevation plot with cyan highlighting
    subplot(3,1,2);
    s2 = plot(dataMat(:,11), dataMat(:,5), '-', 'LineWidth', 3, 'Color', [0, 0.8, 0.8]);
    
    % Map highlighting with yellow color for visibility
    figure(map_fig);
    s3 = plot(dataMat(:,9), dataMat(:,10), '-', 'LineWidth', 3, 'Color', [1, 1, 0]);
    title(sprintf('Stream %d of %d', stream_num, total_streams), 'FontSize', 14, 'FontWeight', 'bold');
end

function [chiFits, kp_data] = profile_chi_enhanced_with_equilibrium(data, strmID, profile_fig, map_fig, Ao, mn, step, chanDir, minchi, maxchi, all_chi_data, all_elev_data)
    % Enhanced profile chi analysis with equilibrium line integration
    
    % Extract data
    dist = data(:,4);
    zr = data(:,2);
    A = data(:,3);
    sel = data(:,5);
    x_coord = data(:,9);
    y_coord = data(:,10);
    chi = data(:,11);
    z = sel;
    
    % Create binned ksn plot with equilibrium reference
    create_binned_ksn_plot_enhanced(chi, z, step, Ao, mn, profile_fig, minchi, maxchi, all_chi_data, all_elev_data);
    
    % Add equilibrium line analysis for current stream
    if exist('all_chi_data', 'var') && ~isempty(all_chi_data)
        add_stream_equilibrium_analysis(chi, z, all_chi_data, all_elev_data, profile_fig);
    end
    
    % Ask user about regressions
    fprintf('\n=== STREAM %d ANALYSIS ===\n', strmID);
    fprintf('Do you want to make regressions through river channel segments?\n');
    regress_opt = input('type "y" for yes or "n" for no: ', 's');
    
    while ~strcmp(regress_opt,'y') && ~strcmp(regress_opt,'n')
        regress_opt = input('YES (y) or NO (n) only!!!: ', 's');
    end
    
    chiFits = [];
    if strcmp(regress_opt, 'y')
        chiFits = perform_chi_regressions_enhanced(chi, z, dist, x_coord, y_coord, Ao, mn, strmID, profile_fig, map_fig);
    else
        % Create empty entry
        noDatVect = -9999 * ones(1, 12);
        chiFits = [strmID, noDatVect];
    end
    
    % Ask user about knickpoints
    fprintf('\nDo you want to mark knickpoints on the chi-elevation plot?\n');
    kp_opt = input('type "y" for yes or "n" for no: ', 's');
    
    while ~strcmp(kp_opt,'y') && ~strcmp(kp_opt,'n')
        kp_opt = input('Yes(y) or No(n)!!!: ', 's');
    end
    
    kp_data = [];
    if strcmp(kp_opt, 'y')
        kp_data = identify_knickpoints_enhanced(chi, zr, A, dist, data, x_coord, y_coord, strmID, profile_fig, map_fig);
    end
    
    % Ask about saving plot
    fprintf('\nDo you want to save this plot?\n');
    save_opt = input('type "y" for yes or "n" for no: ', 's');
    
    if strcmp(save_opt, 'y')
        figure(profile_fig);
        plotname = fullfile(chanDir, [num2str(strmID), '_chi-plot_with_equilibrium']);
        print('-dpng', [plotname, '.png'], '-r300');
        print('-depsc', [plotname, '.eps']);
        fprintf('✓ Plot saved: %s\n', plotname);
    end
    
    % Clean up subplot
    figure(profile_fig);
    subplot(3,1,3);
    cla;
end

function create_binned_ksn_plot_enhanced(chi, z, step, Ao, mn, profile_fig, minchi, maxchi, all_chi_data, all_elev_data)
    
    incs = floor(length(chi)/step);
    if incs < 2, return; end
    
    spt = 1; ept = step;
    mp = nan(incs, 1);
    binKsn = nan(incs, 1);
    
    for i = 1:incs
        if ept > length(chi), ept = length(chi); end
        mp(i) = nanmean(chi(spt:ept));
        chi_segMat = [ones(size(chi(spt:ept))) chi(spt:ept)];
        try
            [b, ~, ~, ~, ~] = regress(z(spt:ept), chi_segMat, 0.05);
            binKsn(i) = b(2) * Ao^mn;
        catch
            binKsn(i) = NaN;
        end
        spt = spt + step;
        ept = ept + step;
    end
    
    figure(profile_fig);
    subplot(3,1,3);
    valid_idx = ~isnan(binKsn);
    if sum(valid_idx) >= 2
        % Plot current stream's binned K_sn values
        plot(mp(valid_idx), binKsn(valid_idx), 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0, 0.8, 0.8], ...
             'MarkerEdgeColor', [0, 0.6, 0.6], 'LineWidth', 1.5, 'DisplayName', 'Stream K_{sn}');
        hold on;
        
        % Add basin-wide equilibrium reference lines
        if exist('all_chi_data', 'var') && ~isempty(all_chi_data) && length(all_chi_data) > 10
            try
                % Calculate basin equilibrium slope
                valid_basin = ~isnan(all_chi_data) & ~isnan(all_elev_data);
                p_eq = polyfit(all_chi_data(valid_basin), all_elev_data(valid_basin), 1);
                reference_ksn = p_eq(1) * Ao^mn;
                
                % Plot basin equilibrium K_sn as horizontal line
                plot([minchi, maxchi], [reference_ksn, reference_ksn], 'r--', 'LineWidth', 2, ...
                     'DisplayName', ['Basin Eq. K_{sn} = ' num2str(reference_ksn, '%.1f')]);
                
                % Calculate theoretical range (±1 standard deviation)
                residuals = all_elev_data(valid_basin) - polyval(p_eq, all_chi_data(valid_basin));
                std_residual = std(residuals, 'omitnan');
                std_ksn = std_residual * Ao^mn;
                
                % Plot uncertainty envelope
                upper_ksn = reference_ksn + std_ksn;
                lower_ksn = max(0, reference_ksn - std_ksn); % Don't go below zero
                
                fill([minchi, maxchi, maxchi, minchi], ...
                     [lower_ksn, lower_ksn, upper_ksn, upper_ksn], ...
                     [1, 0.8, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
                     'DisplayName', 'Equilibrium Range');
                
                % Add annotations for stream behavior relative to equilibrium
                mean_stream_ksn = nanmean(binKsn(valid_idx));
                if mean_stream_ksn > reference_ksn + std_ksn
                    behavior_text = 'Above Equilibrium (High K_{sn})';
                    behavior_color = [1, 0.4, 0];
                elseif mean_stream_ksn < reference_ksn - std_ksn
                    behavior_text = 'Below Equilibrium (Low K_{sn})';
                    behavior_color = [0, 0.6, 0];
                else
                    behavior_text = 'Near Equilibrium';
                    behavior_color = [0, 0, 0.8];
                end
                
                fprintf('K_sn Analysis: Stream mean = %.1f, Basin equilibrium = %.1f (±%.1f)\n', ...
                        mean_stream_ksn, reference_ksn, std_ksn);
                fprintf('Stream K_sn Classification: %s\n', behavior_text);
                
            catch ME
                fprintf('Warning: Could not calculate equilibrium K_sn: %s\n', ME.message);
                % Fallback to simple reference line
                median_ksn = nanmedian(binKsn(valid_idx));
                plot([minchi, maxchi], [median_ksn, median_ksn], 'g--', 'LineWidth', 2, ...
                     'DisplayName', ['Stream Median K_{sn} = ' num2str(median_ksn, '%.1f')]);
            end
        end
        
        xlabel('\chi (m)', 'FontSize', 12, 'FontWeight', 'bold'); 
        ylabel(['k_{sn} (m^{' num2str(2*mn) '})'], 'FontSize', 12, 'FontWeight', 'bold');
        
        % Set axis limits with proper scaling
        y_max = max([nanmax(binKsn) * 1.2, reference_ksn * 2]);
        axis([minchi, maxchi + (maxchi - minchi) * 0.1, 0, y_max]);
        
        legend('Location', 'best', 'FontSize', 9);
        grid on; grid minor;
    end
end

function add_stream_equilibrium_analysis(chi, z, all_chi_data, all_elev_data, profile_fig)
    % Add equilibrium analysis for current stream following research methodology
    
    try
        % Calculate basin-wide equilibrium line (linear regression through ALL data)
        % This represents the theoretical steady-state profile for the entire basin
        valid_basin_data = ~isnan(all_chi_data) & ~isnan(all_elev_data);
        if sum(valid_basin_data) < 10
            fprintf('Warning: Insufficient basin data for equilibrium calculation\n');
            return;
        end
        
        % Fit linear regression: Z = slope * chi + intercept
        p_basin = polyfit(all_chi_data(valid_basin_data), all_elev_data(valid_basin_data), 1);
        
        % Calculate R-squared for equilibrium line quality
        y_pred_basin = polyval(p_basin, all_chi_data(valid_basin_data));
        ss_res = sum((all_elev_data(valid_basin_data) - y_pred_basin).^2);
        ss_tot = sum((all_elev_data(valid_basin_data) - mean(all_elev_data(valid_basin_data))).^2);
        r2_equilibrium = 1 - (ss_res / ss_tot);
        
        % Calculate expected elevations for current stream
        expected_z = polyval(p_basin, chi);
        
        % Calculate deviations (positive = above equilibrium, negative = below)
        deviations = z - expected_z;
        mean_dev = mean(deviations);
        rms_dev = sqrt(mean(deviations.^2));
        max_dev = max(abs(deviations));
        
        % Classify stream segments based on deviation significance
        deviation_threshold = rms_dev * 0.3; % Threshold for significant deviation
        above_eq = deviations > deviation_threshold;
        below_eq = deviations < -deviation_threshold;
        near_eq = abs(deviations) <= deviation_threshold;
        
        % Add visual markers on chi-elevation plot
        figure(profile_fig);
        subplot(3,1,2);
        
        % Mark significantly above equilibrium (convex/uplift zones)
        if any(above_eq)
            plot(chi(above_eq), z(above_eq), '^', 'Color', [1, 0.4, 0], 'MarkerSize', 8, ...
                 'MarkerFaceColor', [1, 0.6, 0], 'LineWidth', 1.5, 'DisplayName', 'Above Equilibrium');
        end
        
        % Mark significantly below equilibrium (concave/incision zones)
        if any(below_eq)
            plot(chi(below_eq), z(below_eq), 'v', 'Color', [0, 0.6, 0], 'MarkerSize', 8, ...
                 'MarkerFaceColor', [0.4, 1, 0.4], 'LineWidth', 1.5, 'DisplayName', 'Below Equilibrium');
        end
        
        % Mark near-equilibrium zones
        if any(near_eq)
            plot(chi(near_eq), z(near_eq), 'o', 'Color', [0.3, 0.3, 0.7], 'MarkerSize', 4, ...
                 'MarkerFaceColor', [0.6, 0.6, 1], 'DisplayName', 'Near Equilibrium');
        end
        
        % Classify overall stream behavior
        if mean_dev > rms_dev * 0.3
            stream_behavior = 'Above Equilibrium (Convex)';
            behavior_color = [1, 0.6, 0];
        elseif mean_dev < -rms_dev * 0.3
            stream_behavior = 'Below Equilibrium (Concave)';
            behavior_color = [0.4, 1, 0.4];
        else
            stream_behavior = 'Near Equilibrium';
            behavior_color = [0.6, 0.6, 1];
        end
        
        % Enhanced statistics text box
        analysis_text = {
            ['Stream: ' stream_behavior],
            sprintf('Mean Dev: %+.1f m', mean_dev),
            sprintf('RMS Dev: %.1f m', rms_dev),
            sprintf('Max Dev: %.1f m', max_dev),
            sprintf('Basin Eq. R²: %.3f', r2_equilibrium),
            sprintf('Eq. Slope: %.4f', p_basin(1))
        };
        
        text(0.02, 0.98, analysis_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
             'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', behavior_color, ...
             'HorizontalAlignment', 'left', 'LineWidth', 2);
        
        % Add percentage statistics
        pct_above = sum(above_eq) / length(deviations) * 100;
        pct_below = sum(below_eq) / length(deviations) * 100;
        pct_near = sum(near_eq) / length(deviations) * 100;
        
        fprintf('\n=== EQUILIBRIUM ANALYSIS RESULTS ===\n');
        fprintf('Stream Segments: %.1f%% above, %.1f%% below, %.1f%% near equilibrium\n', ...
                pct_above, pct_below, pct_near);
        fprintf('Overall Classification: %s\n', stream_behavior);
        fprintf('Mean Deviation: %+.1f m (RMS: %.1f m)\n', mean_dev, rms_dev);
        fprintf('Basin Equilibrium R²: %.3f\n', r2_equilibrium);
        
        % Geological interpretation
        if mean_dev > rms_dev * 0.5
            fprintf('Interpretation: Stream likely experiencing recent uplift or resistant lithology\n');
        elseif mean_dev < -rms_dev * 0.5
            fprintf('Interpretation: Stream dominated by incision, possibly due to base level fall\n');
        else
            fprintf('Interpretation: Stream approaching steady-state conditions\n');
        end
        
    catch ME
        fprintf('Warning: Could not add equilibrium analysis: %s\n', ME.message);
        fprintf('Error details: %s\n', ME.getReport);
    end
end

function [chiFits, kp_data, regression_data, all_chi_data, all_elev_data] = run_interactive_chi_profiler_enhanced(DEM, FD, A, S, fileTag, crita, mn, Ao, smoWin, output_folder)
    % Enhanced interactive chi profiler with all-streams overview and equilibrium analysis
    
    % Create data folder
    folder = fullfile(output_folder, [fileTag, '_stream_data']);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
    % Set nan values and get cellsize
    DEM.Z(DEM.Z <= -9999) = NaN;
    cs = DEM.cellsize;
    
    % Calculate flow accumulation and distance
    A_scaled = A .* (cs^2);
    DFD = flowdistance(FD, 'downstream');
    
    % Save STREAMobj
    fileName = [fileTag, '_pickedstreams.mat'];
    save(fullfile(folder, fileName), 'S');
    
    % Prepare stream variables
    ordList = S.orderednanlist;
    strmBreaks = find(isnan(ordList));
    GridID = S.IXgrid;
    
    % Calculate chi for entire network
    disp('Calculating chi for stream network...');
    Schi = calculate_chi_for_streams(S, A_scaled, GridID, Ao, mn);
    
    % Prepare other variables
    Sz = double(DEM.Z(GridID));
    SmoZ = Sz;
    Sx = S.x;
    Sy = S.y;
    Sdfd = double(DFD.Z(GridID));
    Sda = double(A_scaled.Z(GridID));
    Sd = S.distance;
    
    % Get plotting limits
    mindfm = nanmin(Sd)/1000; maxdfm = nanmax(Sd)/1000;
    minel = nanmin(Sz); maxel = nanmax(Sz);
    minchi = nanmin(Schi); maxchi = nanmax(Schi);
    
    % Create enhanced overview plots with all streams
    [map_fig, profile_fig] = create_overview_plots_enhanced(DEM, S, strmBreaks, ordList, Sd, Schi, SmoZ, smoWin, cs, mindfm, maxdfm, minel, maxel, minchi, maxchi);
    
    % Collect all chi and elevation data for equilibrium analysis
    fprintf('Collecting data from all %d streams for equilibrium analysis...\n', length(strmBreaks));
    all_chi_data = [];
    all_elev_data = [];
    
    id1 = 0;
    for i = 1:length(strmBreaks)
        strmInds = ordList(id1+1:strmBreaks(i)-1);
        valid_idx = ~isnan(Schi(strmInds)) & ~isnan(SmoZ(strmInds));
        if sum(valid_idx) > 0
            all_chi_data = [all_chi_data; Schi(strmInds(valid_idx))];
            all_elev_data = [all_elev_data; SmoZ(strmInds(valid_idx))];
        end
        id1 = strmBreaks(i);
    end
    
    % Show stream count and ask user which streams to analyze
    num_streams = length(strmBreaks);
    fprintf('\nYou have %d stream channels to analyze.\n', num_streams);
    fprintf('All streams are plotted in gray. Selected streams will be highlighted in cyan.\n');
    
    % Ask user which streams to analyze
    streams_to_analyze = get_streams_to_analyze(num_streams);
    
    % Initialize output variables
    chiFits = [];
    kp_data = [];
    regression_data = struct();
    
    % Process each selected stream
    step = round(smoWin/cs);
    id1 = 0;
    
    for i = 1:length(strmBreaks)
        strmInds = ordList(id1+1:strmBreaks(i)-1);
        
        if ismember(i, streams_to_analyze)
            fprintf('\n=== Analyzing Stream %d of %d ===\n', i, num_streams);
            
            % Prepare data matrix
            dataMat = prepare_stream_data(strmInds, Sdfd, Sz, Sda, Sd, smoWin, cs, DEM, GridID, Sx, Sy, Schi);
            
            % Save stream data
            dataFileName = [num2str(i), '_', fileTag, '_chandata.mat'];
            save(fullfile(folder, dataFileName), 'dataMat');
            
            % Highlight current stream in cyan
            [s1, s2, s3] = highlight_current_stream_enhanced(profile_fig, map_fig, dataMat, i, num_streams);
            
            % Run enhanced profile analysis for this stream
            [newCF, newKP] = profile_chi_enhanced_with_equilibrium(dataMat, i, profile_fig, map_fig, Ao, mn, step, folder, minchi, maxchi, all_chi_data, all_elev_data);
            
            % Store results
            if ~isempty(newCF)
                chiFits = [chiFits; newCF];
            end
            if ~isempty(newKP)
                kp_data = [kp_data; newKP];
            end
            
            % Clean up highlights
            delete([s1, s2, s3]);
            
            % Save intermediate results
            if ~isempty(chiFits)
                chiFileName = [fileTag, '_chiFits.mat'];
                save(fullfile(folder, chiFileName), 'chiFits');
            end
            if ~isempty(kp_data)
                kpFileName = [fileTag, '_kpData.mat'];
                save(fullfile(folder, kpFileName), 'kp_data');
            end
        end
        
        id1 = strmBreaks(i);
    end
    
    % Close overview figures
    close([profile_fig, map_fig]);
    
    % Store regression data for summary
    regression_data.folder = folder;
    regression_data.fileTag = fileTag;
    regression_data.num_streams = num_streams;
    regression_data.analyzed_streams = streams_to_analyze;
    regression_data.all_chi_data = all_chi_data;
    regression_data.all_elev_data = all_elev_data;
    
    % Calculate and save basin-wide equilibrium statistics
    if ~isempty(all_chi_data)
        p_basin = polyfit(all_chi_data, all_elev_data, 1);
        basin_stats = struct();
        basin_stats.equilibrium_slope = p_basin(1);
        basin_stats.equilibrium_intercept = p_basin(2);
        basin_stats.total_data_points = length(all_chi_data);
        
        % Calculate R-squared for basin equilibrium
        y_pred = polyval(p_basin, all_chi_data);
        ss_res = sum((all_elev_data - y_pred).^2);
        ss_tot = sum((all_elev_data - mean(all_elev_data)).^2);
        basin_stats.equilibrium_r_squared = 1 - (ss_res / ss_tot);
        
        save(fullfile(folder, [fileTag, '_basin_equilibrium_stats.mat']), 'basin_stats');
        
        fprintf('\n=== BASIN EQUILIBRIUM STATISTICS ===\n');
        fprintf('Equilibrium slope: %.4f m/m\n', basin_stats.equilibrium_slope);
        fprintf('R-squared: %.4f\n', basin_stats.equilibrium_r_squared);
        fprintf('Total data points: %d\n', basin_stats.total_data_points);
    end
end

% Add these enhanced versions of the existing functions with better regression handling
function [chiFits] = perform_chi_regressions_enhanced(chi, z, dist, x_coord, y_coord, Ao, mn, strmID, profile_fig, map_fig)
    % Enhanced chi regressions with better visualization
    
    chiFits = [];
    seg = 0;
    continue_regression = true;
    
    while continue_regression
        fprintf('\n--- Regression Segment %d ---\n', seg + 1);
        fprintf('Click on MINIMUM then MAXIMUM chi bounds on the chi-elevation plot (middle plot)\n');
        fprintf('Include at least 3 data points\n');
        
        figure(profile_fig);
        subplot(3,1,2);
        [chiP, ~] = ginput(2);
        
        while chiP(1,1) >= chiP(2,1)
            fprintf('Follow the directions! Click MINIMUM chi (left) then MAXIMUM chi (right)\n');
            [chiP, ~] = ginput(2);
        end
        
        min_chi = chiP(1,1);
        max_chi = chiP(2,1);
        
        % Perform regression
        [chiKsn, ksnUC, R2, regBounds, reg_plots] = perform_single_regression_enhanced(chi, z, dist, x_coord, y_coord, min_chi, max_chi, Ao, mn, seg, profile_fig, map_fig);
        
        if ~isempty(chiKsn)
            fprintf('\nRegression Results:\n');
            fprintf('K_sn = %.2f ± %.2f\n', chiKsn, ksnUC);
            fprintf('R² = %.3f\n', R2);
            
            % Ask if user wants to save this fit
            fit_opt = input('Do you want to remember this fit? (y/n): ', 's');
            
            if strcmp(fit_opt, 'y')
                newdata = [strmID, seg+1, chiKsn, ksnUC, chiKsn/Ao^mn, ksnUC/Ao^mn, R2, regBounds, x_coord(end), y_coord(end)];
                chiFits = [chiFits; newdata];
                seg = seg + 1;
            else
                % Delete the regression plots
                delete(reg_plots);
            end
        end
        
        % Ask about another segment
        fit_opt2 = input('Do you want to fit another channel segment? (y/n): ', 's');
        continue_regression = strcmp(fit_opt2, 'y');
    end
    
    if isempty(chiFits)
        noDatVect = -9999 * ones(1, 12);
        chiFits = [strmID, noDatVect];
    end
end

function [chiKsn, ksnUC, R2, regBounds, reg_plots] = perform_single_regression_enhanced(chi, z, dist, x_coord, y_coord, min_chi, max_chi, Ao, mn, seg, profile_fig, map_fig)
    % Enhanced single regression with better visualization
    
    try
        % Select data range
        ind = find(chi >= min_chi & chi <= max_chi);
        if length(ind) < 3
            fprintf('Not enough data points for regression\n');
            chiKsn = []; ksnUC = []; R2 = []; regBounds = []; reg_plots = [];
            return;
        end
        
        chi_seg = chi(ind);
        z_seg = z(ind);
        dist_seg = dist(ind);
        x_seg = x_coord(ind);
        y_seg = y_coord(ind);
        
        regBounds = [min(chi_seg), max(chi_seg), min(z_seg), max(z_seg)];
        
        % Regression
        chi_segMat = [ones(size(z_seg)) chi_seg];
        [b, bint, ~, ~, stats] = regress(z_seg, chi_segMat, 0.05);
        
        % Calculate results
        chiSlope = b(2);
        chiKsn = chiSlope * Ao^mn;
        UnCert = (bint(4) - bint(2)) / 2;
        ksnUC = UnCert * Ao^mn;
        R2 = stats(1);
        
        % Create model line
        ymod = b(2) * chi_seg + b(1);
        
        % Plot regression with enhanced styling
        figure(profile_fig);
        subplot(3,1,1);
        reg_plot1 = plot(dist_seg/1000, ymod, '--', 'LineWidth', 3, 'Color', [1, 0.2, 0.2]);
        hold on;
        plot(dist_seg([1 end])/1000, ymod([1 end]), 's', 'Color', [1, 0.2, 0.2], ...
             'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', [1, 0.5, 0.5]);
        
        subplot(3,1,2);
        reg_plot2 = plot(chi_seg, ymod, '--', 'LineWidth', 3, 'Color', [1, 0.2, 0.2]);
        hold on;
        plot(chi_seg([1 end]), ymod([1 end]), 's', 'Color', [1, 0.2, 0.2], ...
             'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', [1, 0.5, 0.5]);
        
        subplot(3,1,3);
        ck = chiKsn * ones(size(chi_seg));
        reg_plot3 = plot(chi_seg, ck, '--', 'LineWidth', 3, 'Color', [1, 0.2, 0.2]);
        hold on;
        plot(chi_seg([1 end]), chiKsn * [1 1], 's', 'Color', [1, 0.2, 0.2], ...
             'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', [1, 0.5, 0.5]);
        
        figure(map_fig);
        reg_plot4 = plot(x_seg, y_seg, '--', 'LineWidth', 4, 'Color', [1, 0.2, 0.2]);
        hold on;
        plot(x_seg([1 end]), y_seg([1 end]), 's', 'Color', [1, 0.2, 0.2], ...
             'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', [1, 0.5, 0.5]);
        
        reg_plots = [reg_plot1; reg_plot2; reg_plot3; reg_plot4];
        
        % Add enhanced text annotation
        figure(profile_fig);
        subplot(3,1,2);
        text(mean(chi_seg), mean(z_seg), ...
            {['Segment ' num2str(seg+1)], ...
             ['K_{sn} = ' num2str(chiKsn, '%.1f') ' ± ' num2str(ksnUC, '%.1f')], ...
             ['R² = ' num2str(R2, '%.3f')]}, ...
            'FontSize', 10, 'BackgroundColor', 'yellow', 'EdgeColor', 'red', ...
            'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        
    catch ME
        fprintf('Regression failed: %s\n', ME.message);
        chiKsn = []; ksnUC = []; R2 = []; regBounds = []; reg_plots = [];
    end
end

function [kp_data] = identify_knickpoints_enhanced(chi, zr, A, dist, data, x_coord, y_coord, strmID, profile_fig, map_fig)
    % Enhanced knickpoint identification with better visualization
    
    kp_data = [];
    kp = 0;
    continue_kp = true;
    
    dfd = data(:,1);
    sel = data(:,5);
    xmat = data(:,6);
    ymat = data(:,7);
    
    while continue_kp
        fprintf('\n--- Knickpoint %d ---\n', kp + 1);
        fprintf('Click on a point on the chi-elevation plot (middle plot)\n');
        
        figure(profile_fig);
        subplot(3,1,2);
        [chi_kp, ~] = ginput(1);
        
        kp_ind = find(chi <= chi_kp, 1, 'first');
        if isempty(kp_ind)
            fprintf('Point not within bounds. Try again.\n');
            continue;
        end
        
        % Plot knickpoint with enhanced styling
        kp_p2 = plot(chi(kp_ind), sel(kp_ind), 'p', 'MarkerFaceColor', [1, 0, 1], ...
                     'MarkerEdgeColor', 'k', 'MarkerSize', 12, 'LineWidth', 2);
        
        subplot(3,1,1);
        kp_p1 = plot(dist(kp_ind)/1000, sel(kp_ind), 'p', 'MarkerFaceColor', [1, 0, 1], ...
                     'MarkerEdgeColor', 'k', 'MarkerSize', 12, 'LineWidth', 2);
        
        figure(map_fig);
        kp_p3 = plot(x_coord(kp_ind), y_coord(kp_ind), 'p', 'MarkerFaceColor', [1, 0, 1], ...
                     'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'LineWidth', 2);
        
        figure(profile_fig);
        
        % Ask if user wants to save this point
        fit_opt = input('Do you want to remember this point? (y/n): ', 's');
        
        if strcmp(fit_opt, 'y')
            fprintf('Classify the knickpoint type:\n');
            fprintf('1 = Major knickpoint\n');
            fprintf('2 = Minor knickpoint\n');
            fprintf('3 = Lithologic contact\n');
            fprintf('4 = Other\n');
            kp_class = input('Enter classification number: ');
            
            % Add data to table
            newdata = [strmID, kp+1, kp_class, chi(kp_ind), zr(kp_ind), A(kp_ind), ...
                      dist(kp_ind), dfd(kp_ind), sel(kp_ind), x_coord(kp_ind), ...
                      y_coord(kp_ind), xmat(kp_ind), ymat(kp_ind), ...
                      x_coord(end), y_coord(end)];
            kp_data = [kp_data; newdata];
            kp = kp + 1;
        else
            delete([kp_p1, kp_p2, kp_p3]);
        end
        
        % Ask about another point
        fit_opt2 = input('Do you want to select another point? (y/n): ', 's');
        continue_kp = strcmp(fit_opt2, 'y');
    end
end

% Integration function to replace the existing one in main script
function [chiFits, kp_data, regression_data] = run_interactive_chi_profiler(DEM, FD, A, S, fileTag, crita, mn, Ao, smoWin, output_folder)
    % Main function that replaces the existing one - just calls the enhanced version
    [chiFits, kp_data, regression_data, ~, ~] = run_interactive_chi_profiler_enhanced(DEM, FD, A, S, fileTag, crita, mn, Ao, smoWin, output_folder);
end
