function PerformanceInfo = manor_1_performance(L_param, X_TT, Y_TT_clean, noiseZ_abs, estOpts, evalOpts)
% MANOR_1_PERFORMANCE
%
% Current MANOR performance wrapper with the same overall role as the old
% NVMperformance_kcluster_V2 workflow:
%   1) split full data into noisy TRAIN / clean TEST,
%   2) fit MANOR on the training split,
%   3) evaluate on the clean TEST split,
%   4) forward iterative TEST diagnostics,
%   5) forward TRAIN-curve diagnostics when curve truth is supplied.
%
% This wrapper is intentionally compatibility-oriented:
%   - it uses the current MANOR interfaces
%       manor_1_gpu(..., evalOpts, evalTruth)
%       manor_1_eval_gpu(..., evalOpts, evalTruth)
%   - it returns legacy fields expected by older driver scripts whenever the
%     current MANOR estimator provides the corresponding information.
%
% Inputs
%   L_param      : old slice-count knob. Used as fallback for estOpts.N_slices.
%   X_TT         : [D x N] data.
%   Y_TT_clean   : clean scalar labels, length N.
%   noiseZ_abs   : Gaussian noise std added to TRAIN labels only.
%   estOpts      : options for manor_1_gpu.
%   evalOpts     : options for manor_1_eval_gpu.
%
% Required curve truth in estOpts
%   estOpts.curveXData   : [D x N] true curve centers.
%   estOpts.curveTanData : [D x N] true tangents (up to sign).
%
% Silent-mode note
%   - estOpts.silent controls manor_1_gpu and the phase-1/phase-2 CUDA MEX.
%   - evalOpts.silent controls manor_1_eval_gpu and eval MEX.
%   - if only one of them is provided, this wrapper mirrors it to the other
%     so one silent flag is enough in practice.

    if nargin < 4 || isempty(noiseZ_abs)
        noiseZ_abs = 0;
    end
    if nargin < 5 || isempty(estOpts)
        estOpts = struct();
    end
    if nargin < 6 || isempty(evalOpts)
        evalOpts = struct();
    end

    validateattributes(L_param, {'numeric'}, {'real','scalar','positive','integer'}, ...
        mfilename, 'L_param', 1);
    validateattributes(X_TT, {'numeric'}, {'real','2d','nonempty'}, ...
        mfilename, 'X_TT', 2);
    validateattributes(Y_TT_clean, {'numeric'}, {'real','vector','nonempty'}, ...
        mfilename, 'Y_TT_clean', 3);
    validateattributes(noiseZ_abs, {'numeric'}, {'real','scalar','nonnegative'}, ...
        mfilename, 'noiseZ_abs', 4);

    [D, N] = size(X_TT);
    Y_TT_clean = reshape(Y_TT_clean, 1, []);
    if numel(Y_TT_clean) ~= N
        error('manor_1_performance:shape', ...
              'numel(Y_TT_clean) must equal size(X_TT,2).');
    end

    if ~isfield(estOpts, 'curveXData') || isempty(estOpts.curveXData)
        error('manor_1_performance:missingCurveX', ...
              'estOpts.curveXData is required.');
    end
    if ~isfield(estOpts, 'curveTanData') || isempty(estOpts.curveTanData)
        error('manor_1_performance:missingCurveTan', ...
              'estOpts.curveTanData is required.');
    end
    if ~isequal(size(estOpts.curveXData), [D, N])
        error('manor_1_performance:curveXShape', ...
              'estOpts.curveXData must have size [D, N].');
    end
    if ~isequal(size(estOpts.curveTanData), [D, N])
        error('manor_1_performance:curveTanShape', ...
              'estOpts.curveTanData must have size [D, N].');
    end

    % ---------------------------------------------------------------------
    % Defaults aligned with the old performance-wrapper role.
    % ---------------------------------------------------------------------
    if ~isfield(estOpts, 'Ns') || isempty(estOpts.Ns)
        estOpts.Ns = L_param; %#ok<STRNU>
    end
    if ~isfield(estOpts, 'N_slices') || isempty(estOpts.N_slices)
        estOpts.N_slices = L_param;
    end
    if ~isfield(estOpts, 'numRiter') || isempty(estOpts.numRiter)
        estOpts.numRiter = 100;
    end
    if ~isfield(estOpts, 'flag_itertesting') || isempty(estOpts.flag_itertesting)
        estOpts.flag_itertesting = false;
    end

    % Curve truth is present in this wrapper by design, so enable the curve
    % diagnostics expected by the old driver scripts.
    estOpts.flag_curveData = true;

    if ~isfield(evalOpts, 'Test_ratio') || isempty(evalOpts.Test_ratio)
        evalOpts.Test_ratio = 0.1;
    end
    validateattributes(evalOpts.Test_ratio, {'numeric'}, {'real','scalar','>',0,'<',1}, ...
        mfilename, 'evalOpts.Test_ratio');

    % Relative-MSE normalization for iterative test evaluation.
    if ~isfield(evalOpts, 'Y_All') || isempty(evalOpts.Y_All)
        evalOpts.Y_All = Y_TT_clean;
    end

    % Keep silent-mode behavior consistent across estimator build and eval.
    hasEstSilent = isfield(estOpts, 'silent') && ~isempty(estOpts.silent);
    hasEvalSilent = isfield(evalOpts, 'silent') && ~isempty(evalOpts.silent);
    if ~hasEstSilent && hasEvalSilent
        estOpts.silent = logical(evalOpts.silent);
    elseif hasEstSilent && ~hasEvalSilent
        evalOpts.silent = logical(estOpts.silent);
    elseif ~hasEstSilent && ~hasEvalSilent
        estOpts.silent = false;
        evalOpts.silent = false;
    else
        estOpts.silent = logical(estOpts.silent);
        evalOpts.silent = logical(evalOpts.silent);
    end

    % ---------------------------------------------------------------------
    % Holdout split: noisy TRAIN, clean TEST.
    % ---------------------------------------------------------------------
    TTpartition = cvpartition(N, 'Holdout', evalOpts.Test_ratio);
    trainMask = training(TTpartition);
    testMask  = test(TTpartition);
    idxTrain = find(trainMask);
    idxTest  = find(testMask);

    X_Train = X_TT(:, idxTrain);
    X_Test  = X_TT(:, idxTest);
    Y_Train = reshape(Y_TT_clean(idxTrain), 1, []) + noiseZ_abs * randn(1, numel(idxTrain));
    Y_Test  = reshape(Y_TT_clean(idxTest), 1, []);

    curveX_Train   = estOpts.curveXData(:, idxTrain);
    curveTan_Train = estOpts.curveTanData(:, idxTrain);
    curveX_Test    = estOpts.curveXData(:, idxTest);
    curveTan_Test  = estOpts.curveTanData(:, idxTest);

    % Current eval wrapper truth package.
    evalTruth = struct( ...
        'Y_True', Y_Test, ...
        'C_True', curveX_Test, ...
        'V_True', curveTan_Test);

    % Training-side estimator options.
    estOpts_train = estOpts;
    estOpts_train.curveXData   = curveX_Train;
    estOpts_train.curveTanData = curveTan_Train;

    % ---------------------------------------------------------------------
    % Fit estimator and run final clean test evaluation.
    % ---------------------------------------------------------------------
    Estimator = manor_1_gpu(X_Train, Y_Train, estOpts_train, X_Test, evalOpts, evalTruth);

    [MSE_Test, DevCen_Test, DevCen_tan_Test, DevVec_Test, EvalInfoFinal] = ...
        manor_1_eval_gpu(Estimator, X_Test, evalOpts, evalTruth);

    RelMSE_Test = double(MSE_Test) / max(std(double(Y_TT_clean))^2, 1e-12);

    % ---------------------------------------------------------------------
    % Core outputs.
    % ---------------------------------------------------------------------
    PerformanceInfo = struct();
    PerformanceInfo.D = D;
    PerformanceInfo.N = N;
    PerformanceInfo.N_train = numel(idxTrain);
    PerformanceInfo.N_test = numel(idxTest);
    PerformanceInfo.Test_ratio = evalOpts.Test_ratio;
    PerformanceInfo.idxTrain = idxTrain;
    PerformanceInfo.idxTest = idxTest;
    PerformanceInfo.noiseZ_abs = noiseZ_abs;

    PerformanceInfo.MSE_Test = double(MSE_Test);
    PerformanceInfo.RelMSE_Test = double(RelMSE_Test);
    PerformanceInfo.DevCen_Test = double(DevCen_Test);
    PerformanceInfo.DevCen_tan_Test = double(DevCen_tan_Test);
    PerformanceInfo.DevVec_Test = double(DevVec_Test);
    PerformanceInfo.EvalInfoFinal = EvalInfoFinal;

    % ---------------------------------------------------------------------
    % Legacy iterative TEST aliases expected by older scripts.
    % ---------------------------------------------------------------------
    if isfield(Estimator, 'EvalRes_MSE') && ~isempty(Estimator.EvalRes_MSE)
        PerformanceInfo.EvalRes_MSE = Estimator.EvalRes_MSE;
        PerformanceInfo.MSE_iter = local_collapse_if_single_row(Estimator.EvalRes_MSE);
    end
    if isfield(Estimator, 'EvalRes_DevCen') && ~isempty(Estimator.EvalRes_DevCen)
        PerformanceInfo.EvalRes_DevCen = Estimator.EvalRes_DevCen;
        PerformanceInfo.DevCen_Test_iter = local_collapse_if_single_row(Estimator.EvalRes_DevCen);
    end
    if isfield(Estimator, 'EvalRes_DevCen_tan') && ~isempty(Estimator.EvalRes_DevCen_tan)
        PerformanceInfo.EvalRes_DevCen_tan = Estimator.EvalRes_DevCen_tan;
        PerformanceInfo.DevCen_tan_Test_iter = local_collapse_if_single_row(Estimator.EvalRes_DevCen_tan);
    end
    if isfield(Estimator, 'EvalRes_DevVec') && ~isempty(Estimator.EvalRes_DevVec)
        PerformanceInfo.EvalRes_DevVec = Estimator.EvalRes_DevVec;
        PerformanceInfo.DevVec_Test_iter = local_collapse_if_single_row(Estimator.EvalRes_DevVec);
    end

    % ---------------------------------------------------------------------
    % TRAIN curve diagnostics. These should be produced by manor_1_gpu when
    % curve truth is supplied; do not silently fake them here.
    % ---------------------------------------------------------------------
    requiredCurveFields = {'DevVec_iter', 'DevCen_iter', 'DevCen_tan_iter'};
    for i = 1:numel(requiredCurveFields)
        f = requiredCurveFields{i};
        if ~isfield(Estimator, f) || isempty(Estimator.(f))
            error('manor_1_performance:missingCurveDiagnostic', ...
                ['manor_1_gpu did not return required curve-diagnostic field ''%s''. ', ...
                 'When estOpts.curveXData / curveTanData are supplied, manor_1_gpu must compute it.'], f);
        end
    end

    PerformanceInfo.DevVec_iter = reshape(double(Estimator.DevVec_iter), 1, []);
    PerformanceInfo.DevCen_iter = reshape(double(Estimator.DevCen_iter), 1, []);
    PerformanceInfo.DevCen_tan_iter = reshape(double(Estimator.DevCen_tan_iter), 1, []);

    if isfield(Estimator, 'ratio_heavy_iter') && ~isempty(Estimator.ratio_heavy_iter)
        PerformanceInfo.ratio_heavy_iter = reshape(double(Estimator.ratio_heavy_iter), 1, []);
    elseif isfield(Estimator, 'prop_heavy_iter') && ~isempty(Estimator.prop_heavy_iter)
        PerformanceInfo.ratio_heavy_iter = reshape(double(Estimator.prop_heavy_iter), 1, []);
    else
        error('manor_1_performance:missingRatioHeavyIter', ...
              'manor_1_gpu did not return ratio_heavy_iter or prop_heavy_iter.');
    end

    % Final train diagnostics: prefer direct estimator outputs if present;
    % otherwise recompute from the returned final estimator.
    if isfield(Estimator, 'DevVec_Train') && isfield(Estimator, 'DevCen_nor_Train') && isfield(Estimator, 'DevCen_tan_Train') && ...
            ~isempty(Estimator.DevVec_Train) && ~isempty(Estimator.DevCen_nor_Train) && ~isempty(Estimator.DevCen_tan_Train)
        PerformanceInfo.DevVec_Train = double(Estimator.DevVec_Train);
        PerformanceInfo.DevCen_nor_Train = double(Estimator.DevCen_nor_Train);
        PerformanceInfo.DevCen_tan_Train = double(Estimator.DevCen_tan_Train);
    else
        [okTrainDiag, DevCen_tan_Train, DevCen_nor_Train, DevVec_Train] = ...
            local_compute_final_train_curve_diag(Estimator, curveX_Train, curveTan_Train);
        if ~okTrainDiag
            error('manor_1_performance:missingFinalTrainDiagnostic', ...
                  'Unable to recover final train curve diagnostics from the returned estimator.');
        end
        PerformanceInfo.DevVec_Train = double(DevVec_Train);
        PerformanceInfo.DevCen_nor_Train = double(DevCen_nor_Train);
        PerformanceInfo.DevCen_tan_Train = double(DevCen_tan_Train);
    end

    % ---------------------------------------------------------------------
    % Forward additional current-MANOR metadata.
    % ---------------------------------------------------------------------
    copyFields = {'rhok','axial_normal','axial_tan','num_neigh_iter','rhok_iter','hk_iter', ...
                  'SlocMin','SlocMid','SlocMax','geomtype','prop_heavy_iter','ratio_heavy_iter', ...
                  'h0','rho0','kSlice','cuts','K','num_neigh_pp', ...
                  'phase1_time','phase2_total_time','phase2_per_iter_time','phase2_times', ...
                  'phase2_mode','cache_enabled','profile','memory','perm_sortedY', ...
                  'final_heavy_mask_sorted'};
    for t = 1:numel(copyFields)
        f = copyFields{t};
        if isfield(Estimator, f)
            PerformanceInfo.(f) = Estimator.(f);
        end
    end
end

function out = local_collapse_if_single_row(A)
    if size(A, 1) == 1
        out = reshape(A(1, :), 1, []);
    else
        out = A;
    end
end

function [ok, DevCen_tan_Train, DevCen_nor_Train, DevVec_Train] = ...
        local_compute_final_train_curve_diag(Estimator, curveX_Train, curveTan_Train)
    ok = false;
    DevCen_tan_Train = NaN;
    DevCen_nor_Train = NaN;
    DevVec_Train = NaN;

    if ~isfield(Estimator, 'Vec_pp') || ~isfield(Estimator, 'Cen_pp')
        return;
    end
    Vec_pp = Estimator.Vec_pp;
    Cen_pp = Estimator.Cen_pp;
    if isempty(Vec_pp) || isempty(Cen_pp)
        return;
    end

    nKeep = size(Vec_pp, 2);
    if size(Cen_pp, 2) ~= nKeep
        return;
    end

    if isfield(Estimator, 'final_heavy_mask_sorted') && ~isempty(Estimator.final_heavy_mask_sorted)
        keepMask = logical(Estimator.final_heavy_mask_sorted(:)).';
        if nnz(keepMask) ~= nKeep
            return;
        end
        curveX_keep = curveX_Train(:, keepMask);
        curveTan_keep = curveTan_Train(:, keepMask);
    elseif size(curveX_Train, 2) == nKeep && size(curveTan_Train, 2) == nKeep
        curveX_keep = curveX_Train;
        curveTan_keep = curveTan_Train;
    else
        return;
    end

    DevVec_Train = mean(min(vecnorm(curveTan_keep - Vec_pp, 2, 1), ...
                            vecnorm(curveTan_keep + Vec_pp, 2, 1)));
    DevCen = Cen_pp - curveX_keep;
    DevCen_tan_Train = mean(abs(sum(DevCen .* curveTan_keep, 1)));
    DevCen_nor_Train = mean(vecnorm(DevCen - sum(DevCen .* curveTan_keep, 1) .* curveTan_keep, 2, 1));
    ok = true;
end
