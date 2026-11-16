function Estimator = manor_1_gpu(X, Y, estOpts, X_Test, evalOpts, evalOpts_True)
% MANOR_1_GPU
% CUDA-oriented MATLAB wrapper for MANOR Algorithm 1.
%
% Main design:
%   1) MATLAB handles policy / scheduling / packaging.
%   2) CUDA MEX handles all heavy compute and all exact memory-based chunking.
%   3) Phase 2 defaults to one multi-iteration CUDA call.
%   4) Scalar per-iteration MATLAB orchestration is used when iterative
%      evaluation is requested, or when explicitly selected.

    if nargin < 3 || isempty(estOpts),       estOpts = struct(); end
    if nargin < 4,                           X_Test = []; end
    if nargin < 5 || isempty(evalOpts),      evalOpts = struct(); end
    if nargin < 6 || isempty(evalOpts_True), evalOpts_True = struct(); end

    estOpts = local_fill_defaults(estOpts);
    is_silent = logical(estOpts.silent);

    assert(exist('manor_1_phase1_cuda_mex', 'file') == 3 || exist('manor_1_phase1_cuda_mex', 'file') == 2, ...
        'Cannot find manor_1_phase1_cuda_mex.');
    assert(exist('manor_1_phase2_cuda_mex', 'file') == 3 || exist('manor_1_phase2_cuda_mex', 'file') == 2, ...
        'Cannot find manor_1_phase2_cuda_mex.');

    if estOpts.enable_cuda_forward_compat
        try
            parallel.gpu.enableCUDAForwardCompatibility(true);
        catch
        end
    end

    validateattributes(X, {'numeric'}, {'real','2d','nonempty'}, mfilename, 'X', 1);
    validateattributes(Y, {'numeric'}, {'real','vector','nonempty'}, mfilename, 'Y', 2);

    t_total = tic;

    X = single(X);
    Y = single(reshape(Y, 1, []));
    [D, N] = size(X);

    if numel(Y) ~= N
        error('manor_1_gpu:shape', 'Y must have exactly N entries, where size(X,2)=N.');
    end
    if D > 32
        error('manor_1_gpu:maxd', 'D=%d exceeds current CUDA limit MAX_D_MANOR=32.', D);
    end

    % ---------------------------------------------------------------------
    % sort by Y once
    % ---------------------------------------------------------------------
    t_sort = tic;
    [Y_sorted, perm] = sort(Y, 'ascend');
    X_sorted = X(:, perm);
    sort_time = toc(t_sort);

    [has_curve_data, curveX_sorted, curveTan_sorted] = local_prepare_curve_truth(estOpts, perm, D, N);
    phase2_mode = local_normalize_phase2_mode(estOpts, X_Test, has_curve_data);

    % ---------------------------------------------------------------------
    % phase-1 neighborhood schedule
    % ---------------------------------------------------------------------
    if isempty(estOpts.kSlice)
        kSlice = max(6*D + 3, ceil(N / estOpts.N_slices));
        kHalf  = floor(kSlice / 2);
        kSlice = 2*kHalf + 1;
    else
        kSlice = int32(round(double(estOpts.kSlice)));
        if mod(kSlice, 2) == 0
            kSlice = kSlice + 1;
        end
        kSlice = max(kSlice, int32(6*D + 3));
    end

    if isempty(estOpts.cuts)
        cuts = ceil(linspace(2*D + 1, ceil(double(kSlice)/3), estOpts.numRiter + 1));
        cuts(end) = ceil(double(kSlice)/3);
        cuts = max(2*D + 1, min(ceil(double(kSlice)/3), cuts));
        cuts = unique(int32(cuts), 'stable');
    else
        cuts = int32(estOpts.cuts(:).');
        cuts = max(int32(2*D + 1), min(int32(ceil(double(kSlice)/3)), cuts));
        cuts = unique(cuts, 'stable');
    end

    if isempty(cuts)
        error('manor_1_gpu:cuts', 'cuts is empty after sanitization.');
    end

    % ---------------------------------------------------------------------
    % phase 1
    % ---------------------------------------------------------------------
    phase1_opts = local_build_phase1_opts(estOpts);

    if estOpts.verbose && ~is_silent
        fprintf('[MANOR][Phase1] N=%d D=%d kSlice=%d numCuts=%d\n', N, D, int32(kSlice), numel(cuts));
    end

    t_phase1 = tic;
    p1 = manor_1_phase1_cuda_mex(X_sorted, Y_sorted, int32(kSlice), cuts, phase1_opts);
    phase1_time = toc(t_phase1);

    local_require_finite_array(p1.Vec0,       'p1.Vec0');
    local_require_finite_array(p1.Cen0,       'p1.Cen0');
    local_require_finite_array(p1.Rad0,       'p1.Rad0');
    local_require_finite_array(p1.axial_tan0, 'p1.axial_tan0');

    Vec0 = single(p1.Vec0);
    Cen0 = single(p1.Cen0);
    Rad0 = single(p1.Rad0);
    axial_tan0 = single(p1.axial_tan0);

    h0   = median(double(Rad0));
    rho0 = median(double(axial_tan0)) / max(h0, 1e-8);

    local_require_finite_scalar(h0, 'h0');
    local_require_finite_scalar(rho0, 'rho0');

    if isempty(estOpts.K)
        K = max(0, estOpts.hyper_param1 + ceil(estOpts.factorK * log(double(N) / 500)));
    else
        K = max(0, round(double(estOpts.K)));
    end
    rhok_iter = single(rho0 * exp(-1/(2 + 4*estOpts.fHolder)).^(0:K-1));
    hk_iter   = single(min(h0 * exp(1/(2*max(4,D))).^(10:K+9), 3*sqrt(double(D))));

    local_require_finite_array(hk_iter,   'hk_iter');
    local_require_finite_array(rhok_iter, 'rhok_iter');

    DevVec0 = NaN;
    DevCen0 = NaN;
    DevCen_tan0 = NaN;
    DevVec_iter = [];
    DevCen_iter = [];
    DevCen_tan_iter = [];
    if has_curve_data
        heavy_mask0 = logical(local_get_struct_field(p1, 'heavy_mask0', true(1, N)));
        [DevVec0, DevCen0, DevCen_tan0] = ...
            local_compute_curve_diagnostics(curveX_sorted, curveTan_sorted, Vec0, Cen0, heavy_mask0);
        DevVec_iter = NaN(1, K);
        DevCen_iter = NaN(1, K);
        DevCen_tan_iter = NaN(1, K);
    end

    % ---------------------------------------------------------------------
    % optional iterative eval containers
    % ---------------------------------------------------------------------
    do_iter_eval = estOpts.flag_itertesting && ~isempty(X_Test);
    EvalRes_MSE = [];
    EvalRes_DevCen = [];
    EvalRes_DevCen_tan = [];
    EvalRes_DevVec = [];
    EvalInfo = repmat(local_empty_eval_info(), 1, K+1);
    iter_eval_times = NaN(1, K+1);
    iter_eval_total_time = 0;
    iter_eval_subset_setup_time = 0;
    iter_eval_subset_clear_time = 0;

    cst_eval = local_eval_norm_var(evalOpts, evalOpts_True, Y_sorted);
    evalOpts_cached = evalOpts;
    evalOpts_iter = evalOpts;
    iter_eval_mask = false(1, K+1);
    iter_eval_subset_info = struct('used_subset', false, 'count', 0, 'full_count', 0, ...
                                   'fraction', 1.0, 'indices', []);

    if do_iter_eval
        if isfield(evalOpts,'hyper_param2') && ~isempty(evalOpts.hyper_param2)
            m_eval = numel(evalOpts.hyper_param2);
        else
            m_eval = 1;
        end
        EvalRes_MSE        = NaN(m_eval, K+1);
        EvalRes_DevCen     = NaN(m_eval, K+1);
        EvalRes_DevCen_tan = NaN(m_eval, K+1);
        EvalRes_DevVec     = NaN(m_eval, K+1);

        iter_eval_mask = local_build_iter_eval_mask(K, estOpts);
        [X_Test_iter, evalOpts_True_iter, iter_eval_subset_info] = local_prepare_iter_eval_subset(X_Test, evalOpts_True, estOpts);

        t_cache_setup = tic;
        manor_1_eval_gpu('cache_test', X_Test_iter, evalOpts_True_iter);
        iter_eval_subset_setup_time = toc(t_cache_setup);
        evalCacheCleanup = onCleanup(@() local_clear_eval_test_cache()); %#ok<NASGU>
        evalOpts_cached.use_cached_test = true;
        evalOpts_cached.cache_test_data = false;
        evalOpts_cached.keep_test_cache = true;
        evalOpts_iter = local_apply_eval_defaults(evalOpts_cached);

        flag_heavy0 = local_build_phase1_tmp_keep_mask( ...
            D, double(p1.num_neigh0), local_get_struct_field(p1, 'heavy_mask0', []), ...
            estOpts.heavy_factor_tmp);

        Estimator0 = local_pack_partial_estimator( ...
            estOpts.hyper_param1, Vec0, Cen0, X_sorted, Y_sorted, ...
            h0, rho0, double(p1.num_neigh0), flag_heavy0);

        if iter_eval_mask(1)
            t_eval = tic;
            [MSE_Test, DevCen_Test, DevCen_tan_Test, DevVec_Test, EvalInfo(1)] = ...
                manor_1_eval_gpu(Estimator0, [], evalOpts_iter, struct());
            iter_eval_times(1) = toc(t_eval);
            EvalInfo(1).skipped = false;
            EvalInfo(1).eval_scope = local_iter_eval_scope_label(iter_eval_subset_info, false);
            EvalInfo(1).test_count = double(iter_eval_subset_info.count);

            EvalRes_MSE(:,1)        = MSE_Test / max(cst_eval, 1e-12);
            EvalRes_DevCen(:,1)     = DevCen_Test;
            EvalRes_DevCen_tan(:,1) = DevCen_tan_Test;
            EvalRes_DevVec(:,1)     = DevVec_Test;
        else
            EvalInfo(1).skipped = true;
            EvalInfo(1).eval_scope = 'skipped';
            EvalInfo(1).test_count = 0;
        end
    end


    % ---------------------------------------------------------------------
    % phase 2
    % ---------------------------------------------------------------------
    phase2_opts = local_build_phase2_opts(estOpts, phase2_mode);

    p2 = struct();
    phase2_total_time = NaN;
    phase2_times = [];
    num_neigh_iter_phase2 = [];
    prop_heavy_iter_phase2 = [];
    phase2_iter_profiles = {};
    center_guard_accept_frac_iter = [];
    vector_guard_accept_frac_iter = [];
    num_neigh_hist_max_phase2 = [];
    center_guard_hist_max_phase2 = [];
    center_guard_metric_phase2 = [];
    vector_guard_hist_max_phase2 = [];
    vector_guard_metric_phase2 = [];
    Vec_pp = Vec0;
    Cen_pp = Cen0;

    if K == 0
        if estOpts.verbose && ~is_silent
            fprintf('[MANOR][Phase2] skipped because K=0; using phase-1 outputs as final estimator state.\n');
        end

        p2 = local_build_zero_phase2_output(p1, Vec0, Cen0);
        phase2_total_time = 0;
        phase2_times = zeros(1, 0);
        num_neigh_iter_phase2 = zeros(1, 0);
        prop_heavy_iter_phase2 = zeros(1, 0);
        center_guard_accept_frac_iter = zeros(1, 0);
        vector_guard_accept_frac_iter = zeros(1, 0);
        num_neigh_hist_max_phase2 = double(reshape(p2.num_neigh_hist_max, 1, []));
        center_guard_hist_max_phase2 = double(reshape(p2.center_guard_hist_max, 1, []));
        center_guard_metric_phase2 = double(reshape(p2.center_guard_metric, 1, []));
        vector_guard_hist_max_phase2 = double(reshape(p2.vector_guard_hist_max, 1, []));
        vector_guard_metric_phase2 = double(reshape(p2.vector_guard_metric, 1, []));
    elseif strcmp(phase2_mode, 'single_call')
        if estOpts.verbose && ~is_silent
            fprintf('[MANOR][Phase2] single CUDA call with K=%d\n', K);
        end

        t_phase2 = tic;
        p2 = manor_1_phase2_cuda_mex( ...
            X_sorted, Y_sorted, Vec0, Cen0, ...
            single(hk_iter), single(rhok_iter), int32(kSlice), phase2_opts);
        phase2_total_time = toc(t_phase2);

        local_require_finite_array(p2.Vec_next,  'p2.Vec_next');
        local_require_finite_array(p2.Cen_next,  'p2.Cen_next');
        local_require_finite_array(p2.num_neigh, 'p2.num_neigh');
        local_require_finite_array(p2.heavy_mask,'p2.heavy_mask');

        if isfield(p2, 'grad') && ~isempty(p2.grad)
            local_require_finite_array(p2.grad, 'p2.grad');
        end
        if isfield(p2, 'weight_sum') && ~isempty(p2.weight_sum)
            local_require_finite_array(p2.weight_sum, 'p2.weight_sum');
        end

        Vec_pp = single(p2.Vec_next);
        Cen_pp = single(p2.Cen_next);

        if isfield(p2, 'num_neigh_iter') && ~isempty(p2.num_neigh_iter)
            num_neigh_iter_phase2 = double(reshape(p2.num_neigh_iter, 1, []));
        else
            num_neigh_iter_phase2 = repmat(mean(double(p2.num_neigh)), 1, K);
        end

        if isfield(p2, 'heavy_frac_iter') && ~isempty(p2.heavy_frac_iter)
            prop_heavy_iter_phase2 = double(reshape(p2.heavy_frac_iter, 1, []));
        else
            prop_heavy_iter_phase2 = repmat(mean(double(p2.heavy_mask)), 1, K);
        end

        if isfield(p2, 'profile')
            phase2_iter_profiles = {p2.profile};
        end

        if isfield(p2, 'num_neigh_hist_max') && ~isempty(p2.num_neigh_hist_max)
            num_neigh_hist_max_phase2 = double(reshape(p2.num_neigh_hist_max, 1, []));
        else
            num_neigh_hist_max_phase2 = double(reshape(p2.num_neigh, 1, []));
        end

        if isfield(p2, 'center_guard_hist_max') && ~isempty(p2.center_guard_hist_max)
            center_guard_hist_max_phase2 = double(reshape(p2.center_guard_hist_max, 1, []));
        else
            center_guard_hist_max_phase2 = local_center_guard_metric_from_phase2_output(p2, estOpts);
        end

        if isfield(p2, 'center_guard_metric') && ~isempty(p2.center_guard_metric)
            center_guard_metric_phase2 = double(reshape(p2.center_guard_metric, 1, []));
        else
            center_guard_metric_phase2 = local_center_guard_metric_from_phase2_output(p2, estOpts);
        end

        if isfield(p2, 'vector_guard_hist_max') && ~isempty(p2.vector_guard_hist_max)
            vector_guard_hist_max_phase2 = double(reshape(p2.vector_guard_hist_max, 1, []));
        else
            vector_guard_hist_max_phase2 = local_vector_guard_metric_from_phase2_output(p2, estOpts);
        end

        if isfield(p2, 'vector_guard_metric') && ~isempty(p2.vector_guard_metric)
            vector_guard_metric_phase2 = double(reshape(p2.vector_guard_metric, 1, []));
        else
            vector_guard_metric_phase2 = local_vector_guard_metric_from_phase2_output(p2, estOpts);
        end

        if isfield(p2, 'center_accept_frac_iter') && ~isempty(p2.center_accept_frac_iter)
            center_guard_accept_frac_iter = double(reshape(p2.center_accept_frac_iter, 1, []));
        else
            center_guard_accept_frac_iter = ones(1, K);
        end

        if isfield(p2, 'vector_accept_frac_iter') && ~isempty(p2.vector_accept_frac_iter)
            vector_guard_accept_frac_iter = double(reshape(p2.vector_accept_frac_iter, 1, []));
        else
            vector_guard_accept_frac_iter = ones(1, K);
        end
    else
        if estOpts.verbose && ~is_silent
            fprintf('[MANOR][Phase2] scalar-loop compatibility mode with K=%d\n', K);
        end

        phase2_times = NaN(1, K);
        num_neigh_iter_phase2 = NaN(1, K);
        prop_heavy_iter_phase2 = NaN(1, K);
        center_guard_accept_frac_iter = NaN(1, K);
        vector_guard_accept_frac_iter = NaN(1, K);
        num_neigh_hist_max_phase2 = zeros(1, N);
        center_guard_hist_max_phase2 = zeros(1, N);
        center_guard_metric_phase2 = zeros(1, N);
        vector_guard_hist_max_phase2 = zeros(1, N);
        vector_guard_metric_phase2 = zeros(1, N);

        for k = 1:K
            t_phase2 = tic;
            p2 = manor_1_phase2_cuda_mex( ...
                X_sorted, Y_sorted, Vec_pp, Cen_pp, ...
                single(hk_iter(k)), single(rhok_iter(k)), int32(kSlice), phase2_opts);
            phase2_times(k) = toc(t_phase2);
            if isfield(p2, 'profile')
                phase2_iter_profiles{k} = p2.profile;
            else
                phase2_iter_profiles{k} = struct();
            end

            local_require_finite_array(p2.Vec_next,  sprintf('p2.Vec_next(k=%d)', k));
            local_require_finite_array(p2.Cen_next,  sprintf('p2.Cen_next(k=%d)', k));
            local_require_finite_array(p2.num_neigh, sprintf('p2.num_neigh(k=%d)', k));
            local_require_finite_array(p2.heavy_mask,sprintf('p2.heavy_mask(k=%d)', k));

            Vec_pp = single(p2.Vec_next);

            num_neigh = double(p2.num_neigh);
            heavy_mask = logical(p2.heavy_mask);
            center_guard_metric_phase2 = local_center_guard_metric_from_phase2_output(p2, estOpts);
            vector_guard_metric_phase2 = local_vector_guard_metric_from_phase2_output(p2, estOpts);
            num_neigh_hist_max_phase2 = max(num_neigh_hist_max_phase2, num_neigh);
            center_guard_hist_max_phase2 = max(center_guard_hist_max_phase2, center_guard_metric_phase2);
            vector_guard_hist_max_phase2 = max(vector_guard_hist_max_phase2, vector_guard_metric_phase2);

            if estOpts.center_update_guard_enable
                accept_center = center_guard_metric_phase2 >= double(estOpts.center_update_guard_ratio) .* center_guard_hist_max_phase2;
                Cen_next_candidate = single(p2.Cen_next);
                Cen_pp(:, accept_center) = Cen_next_candidate(:, accept_center);
                center_guard_accept_frac_iter(k) = mean(double(accept_center));
            else
                Cen_pp = single(p2.Cen_next);
                center_guard_accept_frac_iter(k) = 1.0;
            end

            if isfield(p2, 'vector_accept_frac_iter') && ~isempty(p2.vector_accept_frac_iter)
                tmp_vector_accept = double(reshape(p2.vector_accept_frac_iter, 1, []));
                vector_guard_accept_frac_iter(k) = tmp_vector_accept(end);
            elseif estOpts.vector_update_guard_enable
                accept_vector = vector_guard_metric_phase2 >= double(estOpts.vector_update_guard_ratio) .* vector_guard_hist_max_phase2;
                vector_guard_accept_frac_iter(k) = mean(double(accept_vector));
            else
                vector_guard_accept_frac_iter(k) = 1.0;
            end

            num_neigh_iter_phase2(k) = mean(num_neigh);
            prop_heavy_iter_phase2(k) = mean(double(heavy_mask));

            if has_curve_data
                [DevVec_iter(k), DevCen_iter(k), DevCen_tan_iter(k)] = ...
                    local_compute_curve_diagnostics(curveX_sorted, curveTan_sorted, Vec_pp, Cen_pp, heavy_mask);
            end

            if do_iter_eval
                flag_heavy_tmp = local_build_phase2_tmp_keep_mask( ...
                    D, num_neigh, heavy_mask, estOpts.heavy_factor_tmp, false);

                Estimatork = local_pack_partial_estimator( ...
                    estOpts.hyper_param1, Vec_pp, Cen_pp, X_sorted, Y_sorted, ...
                    hk_iter(k), rhok_iter(k), num_neigh, flag_heavy_tmp);
                Estimatork.num_neigh_iter = [mean(double(p1.num_neigh0)), num_neigh_iter_phase2];
                Estimatork.hk_iter = double(hk_iter);
                Estimatork.rhok_iter = double(rhok_iter);

                if iter_eval_mask(k+1)
                    use_full_final_eval = (k == K) && logical(estOpts.iter_eval_final_full) && iter_eval_subset_info.used_subset;
                    t_eval = tic;
                    if use_full_final_eval
                        evalOpts_full = evalOpts;
                        evalOpts_full.use_cached_test = false;
                        evalOpts_full.cache_test_data = true;
                        evalOpts_full.keep_test_cache = false;
                        evalOpts_full.enable_host_double_buffer = 0;
                        evalOpts_full = local_apply_eval_defaults(evalOpts_full);
                        [MSE_Test, DevCen_Test, DevCen_tan_Test, DevVec_Test, EvalInfo(k+1)] = ...
                            manor_1_eval_gpu(Estimatork, X_Test, evalOpts_full, evalOpts_True);
                    else
                        [MSE_Test, DevCen_Test, DevCen_tan_Test, DevVec_Test, EvalInfo(k+1)] = ...
                            manor_1_eval_gpu(Estimatork, [], evalOpts_iter, struct());
                    end
                    iter_eval_times(k+1) = toc(t_eval);
                    EvalInfo(k+1).skipped = false;
                    EvalInfo(k+1).eval_scope = local_iter_eval_scope_label(iter_eval_subset_info, use_full_final_eval);
                    EvalInfo(k+1).test_count = double(local_iter_eval_test_count(iter_eval_subset_info, use_full_final_eval));

                    EvalRes_MSE(:,k+1)        = MSE_Test / max(cst_eval, 1e-12);
                    EvalRes_DevCen(:,k+1)     = DevCen_Test;
                    EvalRes_DevCen_tan(:,k+1) = DevCen_tan_Test;
                    EvalRes_DevVec(:,k+1)     = DevVec_Test;
                else
                    EvalInfo(k+1).skipped = true;
                    EvalInfo(k+1).eval_scope = 'skipped';
                    EvalInfo(k+1).test_count = 0;
                end
            end
        end

        phase2_total_time = sum(phase2_times);
    end

    if do_iter_eval
        iter_eval_total_time = nansum(iter_eval_times);
        t_cache_clear = tic;
        local_clear_eval_test_cache();
        iter_eval_subset_clear_time = toc(t_cache_clear);
    end

    if isempty(num_neigh_hist_max_phase2)
        num_neigh_hist_max_phase2 = double(reshape(p2.num_neigh, 1, []));
    end
    if isempty(center_guard_hist_max_phase2)
        center_guard_hist_max_phase2 = local_center_guard_metric_from_phase2_output(p2, estOpts);
    end
    if isempty(center_guard_metric_phase2)
        center_guard_metric_phase2 = local_center_guard_metric_from_phase2_output(p2, estOpts);
    end
    if isempty(center_guard_accept_frac_iter)
        center_guard_accept_frac_iter = ones(1, K);
    end

    if estOpts.center_reduce_tangent_final
        Cen_pp = local_project_center_off_tangent(Cen_pp, Vec_pp, X_sorted);
    end

    num_neigh_final = double(p2.num_neigh);
    heavy_mask_final = logical(p2.heavy_mask);

    % ---------------------------------------------------------------------
    % final heavy filtering
    % ---------------------------------------------------------------------
    final_heavy = local_build_phase2_tmp_keep_mask( ...
        D, num_neigh_final, heavy_mask_final, estOpts.heavy_factor_tmp, ~do_iter_eval);

    Vec_keep = Vec_pp(:, final_heavy);
    Cen_keep = Cen_pp(:, final_heavy);
    X_keep   = X_sorted(:, final_heavy);
    Y_keep   = Y_sorted(final_heavy);

    % ---------------------------------------------------------------------
    % package estimator
    % ---------------------------------------------------------------------
    Estimator = struct();
    Estimator.hyper_param1   = estOpts.hyper_param1;
    Estimator.Vec_pp         = Vec_keep;
    Estimator.Cen_pp         = Cen_keep;
    Estimator.X_Train        = X_keep;
    Estimator.Y_pp           = Y_keep;

    if isfield(evalOpts, 'hyper_param2') && ~isempty(evalOpts.hyper_param2)
        Estimator.hyper_param2 = single(evalOpts.hyper_param2(:));
    end

    if K > 0
        Estimator.rhok           = double(rhok_iter(end));
        Estimator.axial_normal   = double(hk_iter(end));
        Estimator.axial_tan      = double(hk_iter(end)) * double(rhok_iter(end));
    else
        Estimator.rhok           = double(rho0);
        Estimator.axial_normal   = double(h0);
        Estimator.axial_tan      = double(h0) * double(rho0);
    end

    Estimator.h0             = h0;
    Estimator.rho0           = rho0;
    Estimator.kSlice         = double(kSlice);
    Estimator.cuts           = double(cuts);
    Estimator.K              = K;

    Estimator.num_neigh_pp   = num_neigh_final(final_heavy);
    Estimator.num_neigh_hist_max_pp = double(num_neigh_hist_max_phase2(final_heavy));
    Estimator.center_guard_hist_max_pp = double(center_guard_hist_max_phase2(final_heavy));
    Estimator.center_guard_metric_pp = double(center_guard_metric_phase2(final_heavy));
    Estimator.vector_guard_hist_max_pp = double(vector_guard_hist_max_phase2(final_heavy));
    Estimator.vector_guard_metric_pp = double(vector_guard_metric_phase2(final_heavy));
    Estimator.num_neigh_iter = [mean(double(p1.num_neigh0)), num_neigh_iter_phase2];
    Estimator.rhok_iter      = double(rhok_iter);
    Estimator.hk_iter        = double(hk_iter);
    Estimator.center_guard_accept_frac_iter = double(center_guard_accept_frac_iter);
    Estimator.vector_guard_accept_frac_iter = double(vector_guard_accept_frac_iter);
    Estimator.center_update_guard_enable = logical(estOpts.center_update_guard_enable);
    Estimator.center_update_guard_ratio = double(estOpts.center_update_guard_ratio);
    Estimator.center_update_guard_use_ess = false;
    Estimator.center_guard_metric_name = 'count';
    Estimator.center_reduce_tangent_final = logical(estOpts.center_reduce_tangent_final);
    Estimator.center_kernel_weighted = false;
    Estimator.center_weight_source = 'X';
    Estimator.vector_update_guard_enable = logical(estOpts.vector_update_guard_enable);
    Estimator.vector_update_guard_ratio = double(estOpts.vector_update_guard_ratio);
    Estimator.vector_update_guard_use_ess = false;
    Estimator.vector_guard_metric_name = 'count';
    Estimator.vector_kernel_weighted = false;

    Estimator.prop_heavy_iter      = [mean(double(p1.heavy_mask0)), prop_heavy_iter_phase2];
    Estimator.ratio_heavy_iter     = Estimator.prop_heavy_iter;
    Estimator.phase1_time          = phase1_time;
    Estimator.phase2_total_time    = phase2_total_time;
    Estimator.phase2_per_iter_time = phase2_total_time / max(K, 1);
    Estimator.phase2_times         = phase2_times;
    Estimator.phase2_mode          = phase2_mode;
    Estimator.sort_time            = sort_time;
    Estimator.perm_sortedY         = perm;
    Estimator.final_heavy_mask_sorted = final_heavy;

    if isfield(p1, 'geomtype0'), Estimator.geomtype = mean(double(p1.geomtype0)); end
    if isfield(p1, 'SlocMin0'),  Estimator.SlocMin  = mean(double(p1.SlocMin0));  end
    if isfield(p1, 'SlocMid0'),  Estimator.SlocMid  = mean(double(p1.SlocMid0));  end
    if isfield(p1, 'SlocMax0'),  Estimator.SlocMax  = mean(double(p1.SlocMax0));  end

    if isfield(p2, 'cache_enabled')
        Estimator.cache_enabled = double(p2.cache_enabled);
    else
        Estimator.cache_enabled = 0;
    end

    if estOpts.phase2_return_debug
        if isfield(p2, 'grad'), Estimator.grad = p2.grad; end
    end
    if isfield(p2, 'weight_sum'), Estimator.weight_sum = p2.weight_sum; end
    if isfield(p2, 'weight_sum_sq'), Estimator.weight_sum_sq = p2.weight_sum_sq; end
    if isfield(p2, 'vector_guard_hist_max'), Estimator.vector_guard_hist_max = p2.vector_guard_hist_max; end
    if isfield(p2, 'vector_guard_metric'), Estimator.vector_guard_metric = p2.vector_guard_metric; end

    if estOpts.keep_raw_outputs
        Estimator.phase1_raw = p1;
        Estimator.phase2_raw = p2;
    end

    profile = struct();
    profile.total_time = toc(t_total);
    profile.sort_time = sort_time;
    profile.phase1_time = phase1_time;
    profile.phase2_total_time = phase2_total_time;
    profile.iter_eval_total_time = iter_eval_total_time;
    profile.iter_eval_times = iter_eval_times;
    profile.iter_eval_subset_setup_time = iter_eval_subset_setup_time;
    profile.iter_eval_subset_clear_time = iter_eval_subset_clear_time;
    profile.iter_eval_mask = iter_eval_mask;
    profile.iter_eval_indices = find(iter_eval_mask) - 1;
    profile.iter_eval_schedule = estOpts.iter_eval_schedule;
    profile.iter_eval_used_subset = iter_eval_subset_info.used_subset;
    profile.iter_eval_subset_count = iter_eval_subset_info.count;
    profile.iter_eval_subset_fraction = iter_eval_subset_info.fraction;
    profile.iter_eval_final_full = logical(estOpts.iter_eval_final_full);
    profile.iter_eval_full_count = iter_eval_subset_info.full_count;
    profile.iter_eval_scopes = local_collect_evalinfo_field(EvalInfo, 'eval_scope');
    profile.iter_eval_test_counts = local_collect_evalinfo_numeric_field(EvalInfo, 'test_count');
    profile.iter_eval_wrapper_total_times = local_collect_evalinfo_numeric_field(EvalInfo, 'wrapper_total_time');
    profile.iter_eval_wrapper_init_times = local_collect_evalinfo_numeric_field(EvalInfo, 'wrapper_init_time');
    profile.iter_eval_wrapper_mex_times = local_collect_evalinfo_numeric_field(EvalInfo, 'wrapper_mex_eval_time');
    profile.normalized_mse_variance = cst_eval;
    profile.phase1_mex_profile = local_get_struct_field(p1, 'profile', struct());
    profile.phase1_memory = local_get_struct_field(p1, 'memory', struct());
    if isfield(p2, 'profile'), profile.phase2_mex_profile = p2.profile; end
    profile.phase2_iter_profiles = phase2_iter_profiles;
    profile.phase2_cache_build_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'cache_build_sec');
    profile.phase2_copyout_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'copyout_sec');
    profile.phase2_iter_total_times_mex = local_collect_phase2_profile_field(phase2_iter_profiles, 'iter_total_sec');
    profile.phase2_regression_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'regression_sec');
    profile.phase2_project_pre_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'project_pre_sec');
    profile.phase2_heavy_reduce_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'heavy_reduce_sec');
    profile.phase2_copy_pre_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'copy_pre_sec');
    profile.phase2_aggregation_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'aggregation_sec');
    profile.phase2_project_final_times = local_collect_phase2_profile_field(phase2_iter_profiles, 'project_final_sec');
    profile.center_guard_accept_frac_iter = double(center_guard_accept_frac_iter);
    profile.center_guard_hist_max = double(center_guard_hist_max_phase2);
    profile.center_guard_metric = double(center_guard_metric_phase2);
    profile.center_update_guard_enable = logical(estOpts.center_update_guard_enable);
    profile.center_update_guard_ratio = double(estOpts.center_update_guard_ratio);
    profile.center_update_guard_use_ess = false;
    profile.center_guard_metric_name = 'count';
    profile.center_reduce_tangent_final = logical(estOpts.center_reduce_tangent_final);
    profile.center_kernel_weighted = false;
    profile.center_weight_source = 'X';
    profile.vector_guard_accept_frac_iter = double(vector_guard_accept_frac_iter);
    profile.vector_guard_hist_max = double(vector_guard_hist_max_phase2);
    profile.vector_guard_metric = double(vector_guard_metric_phase2);
    profile.vector_update_guard_enable = logical(estOpts.vector_update_guard_enable);
    profile.vector_update_guard_ratio = double(estOpts.vector_update_guard_ratio);
    profile.vector_update_guard_use_ess = false;
    profile.vector_guard_metric_name = 'count';
    profile.vector_kernel_weighted = false;
    Estimator.profile = profile;

    mem = struct();
    mem.phase2_saved_cen_next_bytes = 4 * double(D) * double(N);
    mem.phase2_saved_weight_sum_bytes = 4 * double(N);
    mem.phase2_saved_weight_sum_sq_bytes = 4 * double(N);
    mem.phase2_saved_vector_guard_hist_max_bytes = 4 * double(N);
    Estimator.memory = mem;

    if has_curve_data
        [Estimator.DevVec_Train, Estimator.DevCen_nor_Train, Estimator.DevCen_tan_Train] = ...
            local_compute_curve_diagnostics(curveX_sorted, curveTan_sorted, Vec_pp, Cen_pp, heavy_mask_final);
        Estimator.DevVec_iter = [DevVec0, DevVec_iter];
        Estimator.DevCen_iter = [DevCen0, DevCen_iter];
        Estimator.DevCen_tan_iter = [DevCen_tan0, DevCen_tan_iter];
    end

    if do_iter_eval
        Estimator.EvalRes_MSE        = EvalRes_MSE;
        Estimator.EvalRes_DevCen     = EvalRes_DevCen;
        Estimator.EvalRes_DevCen_tan = EvalRes_DevCen_tan;
        Estimator.EvalRes_DevVec     = EvalRes_DevVec;
        Estimator.EvalInfo           = EvalInfo;
    end

    if estOpts.verbose && ~is_silent
        fprintf('[MANOR] done: sort=%.4fs, phase1=%.4fs, phase2_total=%.4fs, iter_eval=%.4fs, K=%d, cache=%d, keep=%d/%d\n', ...
            sort_time, phase1_time, phase2_total_time, iter_eval_total_time, K, Estimator.cache_enabled ~= 0, nnz(final_heavy), N);
    end
end

function estOpts = local_fill_defaults(estOpts)
    estOpts = local_set_default(estOpts, 'N_slices', 64);
    estOpts = local_set_default(estOpts, 'numRiter', 64);
    estOpts = local_set_default(estOpts, 'kSlice', []);
    estOpts = local_set_default(estOpts, 'cuts', []);
    estOpts = local_set_default(estOpts, 'K', []);
    estOpts = local_set_default(estOpts, 'fHolder', 2);
    estOpts = local_set_default(estOpts, 'factorK', 2);
    estOpts = local_set_default(estOpts, 'hyper_param1', 0);
    estOpts = local_set_default(estOpts, 'flag_curveData', false);

    estOpts = local_set_default(estOpts, 'heavy_factor', 0.1);
    estOpts = local_set_default(estOpts, 'heavy_factor_tmp', 0.20);

    estOpts = local_set_default(estOpts, 'gpu_mem_fraction', 0.98);
    estOpts = local_set_default(estOpts, 'reserve_fraction', 0.02);
    estOpts = local_set_default(estOpts, 'reserve_bytes', 1 * 1024^3);

    estOpts = local_set_default(estOpts, 'phase1_return_debug', false);
    estOpts = local_set_default(estOpts, 'phase2_return_debug', false);
    estOpts = local_set_default(estOpts, 'keep_raw_outputs', false);
    estOpts = local_set_default(estOpts, 'flag_itertesting', true);
    estOpts = local_set_default(estOpts, 'enable_cuda_forward_compat', true);
    estOpts = local_set_default(estOpts, 'verbose', false);
    estOpts = local_set_default(estOpts, 'silent', false);
    estOpts = local_set_default(estOpts, 'profile_timers', false);
    estOpts = local_set_default(estOpts, 'iter_eval_schedule', 'all');
    estOpts = local_set_default(estOpts, 'iter_eval_warmup', 4);
    estOpts = local_set_default(estOpts, 'iter_eval_mid_stride', 2);
    estOpts = local_set_default(estOpts, 'iter_eval_late_start', []);
    estOpts = local_set_default(estOpts, 'iter_eval_late_stride', 4);
    estOpts = local_set_default(estOpts, 'iter_eval_indices', []);
    estOpts = local_set_default(estOpts, 'iter_eval_subsample_fraction', 2000); % inactive when min=max=2000
    estOpts = local_set_default(estOpts, 'iter_eval_subsample_min', 2000);
    estOpts = local_set_default(estOpts, 'iter_eval_subsample_max', 2000);
    estOpts = local_set_default(estOpts, 'iter_eval_subsample_seed', 1);
    estOpts = local_set_default(estOpts, 'iter_eval_final_full', false);

    % Phase 1 CUDA policies
    estOpts = local_set_default(estOpts, 'phase1_jitter', 1e-6);
    estOpts = local_set_default(estOpts, 'phase1_power_iter_steps', 12);

    % Large-kSlice speed defaults
    estOpts = local_set_default(estOpts, 'phase1_min_query_block', 1024);
    estOpts = local_set_default(estOpts, 'phase1_max_query_block', 65536);
    estOpts = local_set_default(estOpts, 'phase1_gpu_pass2', true);
    estOpts = local_set_default(estOpts, 'phase1_fullsort_fallback', true);

    % Shared exact top-k path
    estOpts = local_set_default(estOpts, 'phase1_shared_topk_max_kslice', 8192);
    estOpts = local_set_default(estOpts, 'phase1_topk_block_threads', 256);

    % New exact tile-sort + merge top-k path
    estOpts = local_set_default(estOpts, 'phase1_tile_topk_min_kslice', 2048);
    estOpts = local_set_default(estOpts, 'phase1_tile_sort_block_threads', 128); % 128 or 256
    estOpts = local_set_default(estOpts, 'phase1_tile_items_per_thread', 2);     % fixed at 2 in CUDA

    % Phase 2 CUDA policies
    estOpts = local_set_default(estOpts, 'phase2_jitter', 1e-6);
    estOpts = local_set_default(estOpts, 'phase2_power_iter_steps', 10);
    estOpts = local_set_default(estOpts, 'phase2_min_query_block', 512);
    estOpts = local_set_default(estOpts, 'phase2_max_query_block', 131072);
    estOpts = local_set_default(estOpts, 'phase2_exact_slice_cache_max_bytes', 32 * 1024^3);
    estOpts = local_set_default(estOpts, 'phase2_use_cuda_graphs', true);
    estOpts = local_set_default(estOpts, 'phase2_shared_sort_mode', 0);           % 0 auto, 1 on, 2 off
    estOpts = local_set_default(estOpts, 'phase2_shared_sort_max_kslice', 16384);
    estOpts = local_set_default(estOpts, 'phase2_warps_per_block', 2);            % 2 or 4
    estOpts = local_set_default(estOpts, 'phase2_build_block_threads', 256);      % 128,256,512
    estOpts = local_set_default(estOpts, 'center_update_guard_enable', true);
    estOpts = local_set_default(estOpts, 'center_update_guard_ratio', 0.75);
    estOpts = local_set_default(estOpts, 'center_reduce_tangent_final', true);
    estOpts = local_set_default(estOpts, 'vector_update_guard_enable', false);
    estOpts = local_set_default(estOpts, 'vector_update_guard_ratio', 0.75);
end

function phase1_opts = local_build_phase1_opts(estOpts)
    phase1_opts = struct();
    phase1_opts.gpu_mem_fraction       = single(estOpts.gpu_mem_fraction);
    phase1_opts.reserve_fraction       = single(estOpts.reserve_fraction);
    phase1_opts.reserve_bytes          = double(estOpts.reserve_bytes);
    phase1_opts.heavy_factor           = single(estOpts.heavy_factor);
    phase1_opts.return_debug           = logical(estOpts.phase1_return_debug);
    phase1_opts.jitter                 = single(estOpts.phase1_jitter);
    phase1_opts.power_iter_steps       = int32(estOpts.phase1_power_iter_steps);
    phase1_opts.min_query_block        = int32(estOpts.phase1_min_query_block);
    phase1_opts.max_query_block        = int32(estOpts.phase1_max_query_block);
    phase1_opts.gpu_pass2              = logical(estOpts.phase1_gpu_pass2);
    phase1_opts.fullsort_fallback      = logical(estOpts.phase1_fullsort_fallback);
    phase1_opts.shared_topk_mode       = int32(2);
    phase1_opts.shared_topk_max_kslice = int32(estOpts.phase1_shared_topk_max_kslice);
    phase1_opts.topk_block_threads     = int32(estOpts.phase1_topk_block_threads);
    phase1_opts.tile_topk_mode         = int32(2);
    phase1_opts.tile_topk_min_kslice   = int32(estOpts.phase1_tile_topk_min_kslice);
    phase1_opts.tile_sort_block_threads= int32(estOpts.phase1_tile_sort_block_threads);
    phase1_opts.tile_items_per_thread  = int32(estOpts.phase1_tile_items_per_thread);
    phase1_opts.profile_timers         = logical(estOpts.profile_timers);
    phase1_opts.silent                 = logical(estOpts.silent);
end

function phase2_opts = local_build_phase2_opts(estOpts, phase2_mode)
    phase2_opts = struct();
    phase2_opts.gpu_mem_fraction            = single(estOpts.gpu_mem_fraction);
    phase2_opts.reserve_fraction            = single(estOpts.reserve_fraction);
    phase2_opts.reserve_bytes               = double(estOpts.reserve_bytes);
    phase2_opts.heavy_factor                = single(estOpts.heavy_factor);
    phase2_opts.return_debug                = logical(estOpts.phase2_return_debug);
    phase2_opts.jitter                      = single(estOpts.phase2_jitter);
    phase2_opts.power_iter_steps            = int32(estOpts.phase2_power_iter_steps);
    phase2_opts.min_query_block             = int32(estOpts.phase2_min_query_block);
    phase2_opts.max_query_block             = int32(estOpts.phase2_max_query_block);
    phase2_opts.exact_slice_cache_mode      = int32(2);
    phase2_opts.exact_slice_cache_max_bytes = double(estOpts.phase2_exact_slice_cache_max_bytes);
    phase2_opts.use_cuda_graphs             = logical(estOpts.phase2_use_cuda_graphs);
    phase2_opts.shared_sort_mode            = int32(estOpts.phase2_shared_sort_mode);
    phase2_opts.shared_sort_max_kslice      = int32(estOpts.phase2_shared_sort_max_kslice);
    phase2_opts.warps_per_block             = int32(estOpts.phase2_warps_per_block);
    phase2_opts.build_block_threads         = int32(estOpts.phase2_build_block_threads);
    phase2_opts.center_update_guard_enable  = logical(estOpts.center_update_guard_enable);
    phase2_opts.center_update_guard_ratio   = single(estOpts.center_update_guard_ratio);
    phase2_opts.center_update_guard_use_ess = false;   % hardcoded off
    phase2_opts.center_reduce_tangent_final = logical(estOpts.center_reduce_tangent_final);
    phase2_opts.center_kernel_weighted      = false;   % hardcoded off
    phase2_opts.center_weight_source_mode   = int32(1);% hardcoded raw X centers
    phase2_opts.vector_update_guard_enable  = logical(estOpts.vector_update_guard_enable);
    phase2_opts.vector_update_guard_ratio   = single(estOpts.vector_update_guard_ratio);
    phase2_opts.vector_update_guard_use_ess = false;   % hardcoded off
    phase2_opts.vector_kernel_weighted      = false;   % hardcoded off
    phase2_opts.profile_timers              = logical(estOpts.profile_timers);
    phase2_opts.silent                      = logical(estOpts.silent);

    % Force cache truly off when requested.
    if double(phase2_opts.exact_slice_cache_mode) == 2
        phase2_opts.exact_slice_cache_max_bytes = 0;
        phase2_opts.shared_sort_mode = int32(2);
    end

    if strcmp(phase2_mode, 'scalar_loop')
        phase2_opts.use_cuda_graphs = false;
    end
end

function tf = local_center_guard_uses_ess(~)
    tf = false;
end

function label = local_center_guard_metric_label(~)
    label = 'count';
end

function tf = local_vector_guard_uses_ess(~)
    tf = false;
end

function label = local_vector_guard_metric_label(~)
    label = 'count';
end

function metric = local_center_guard_metric_from_phase2_output(p2, ~)
    num_neigh = double(reshape(local_get_struct_field(p2, 'num_neigh', []), 1, []));
    if isempty(num_neigh)
        metric = [];
        return;
    end

    if isfield(p2, 'center_guard_metric') && ~isempty(p2.center_guard_metric)
        metric = double(reshape(p2.center_guard_metric, 1, []));
        return;
    end

    metric = num_neigh;
end

function metric = local_vector_guard_metric_from_phase2_output(p2, ~)
    num_neigh = double(reshape(local_get_struct_field(p2, 'num_neigh', []), 1, []));
    if isempty(num_neigh)
        metric = [];
        return;
    end

    if isfield(p2, 'vector_guard_metric') && ~isempty(p2.vector_guard_metric)
        metric = double(reshape(p2.vector_guard_metric, 1, []));
        return;
    end

    metric = num_neigh;
end


function mode = local_center_weight_source_mode(~)
    mode = int32(1);
end

function label = local_center_weight_source_label(~)
    label = 'X';
end

function mode = local_normalize_phase2_mode(estOpts, X_Test, has_curve_data)
    if local_get_struct_field(estOpts, 'flag_itertesting', false) && ~isempty(X_Test)
        mode = 'scalar_loop';
        return;
    end
    if has_curve_data
        mode = 'scalar_loop';
        return;
    end

    mode_in = local_get_struct_field(estOpts, 'phase2_mode', '');
    if isempty(mode_in)
        mode = 'single_call';
        return;
    end

    mode = lower(strtrim(char(string(mode_in))));
    if ~strcmp(mode, 'single_call') && ~strcmp(mode, 'scalar_loop')
        error('manor_1_gpu:phase2_mode', 'phase2_mode must be ''single_call'' or ''scalar_loop''.');
    end
end

function evalOpts = local_apply_eval_defaults(evalOpts)
    if ~isfield(evalOpts, 'metrics_mode') || isempty(evalOpts.metrics_mode)
        evalOpts.metrics_mode = 'full';
    end
    if ~isfield(evalOpts, 'exact_tangent_small_d') || isempty(evalOpts.exact_tangent_small_d)
        evalOpts.exact_tangent_small_d = true;
    end
end


function p2 = local_build_zero_phase2_output(p1, Vec0, Cen0)
    num_neigh0 = double(reshape(local_get_struct_field(p1, 'num_neigh0', []), 1, []));
    if isempty(num_neigh0)
        error('manor_1_gpu:K0', 'Phase-1 output must contain num_neigh0 when K=0.');
    end

    heavy_mask0 = logical(reshape(local_get_struct_field(p1, 'heavy_mask0', true(1, numel(num_neigh0))), 1, []));

    p2 = struct();
    p2.Vec_next = single(Vec0);
    p2.Cen_next = single(Cen0);
    p2.num_neigh = double(num_neigh0);
    p2.heavy_mask = logical(heavy_mask0);
    p2.num_neigh_hist_max = double(num_neigh0);
    p2.center_guard_hist_max = double(num_neigh0);
    p2.center_guard_metric = double(num_neigh0);
    p2.vector_guard_hist_max = double(num_neigh0);
    p2.vector_guard_metric = double(num_neigh0);
    p2.center_accept_frac_iter = zeros(1, 0, 'single');
    p2.vector_accept_frac_iter = zeros(1, 0, 'single');
    p2.cache_enabled = 0;
    p2.profile = struct('total_sec', 0);
end

function keep_mask = local_build_phase1_tmp_keep_mask(D, num_neigh, heavy_mask, heavy_factor_tmp)
    num_neigh = double(reshape(num_neigh, 1, []));
    if isempty(heavy_mask)
        heavy_mask = true(1, numel(num_neigh));
    else
        heavy_mask = logical(reshape(heavy_mask, 1, []));
    end

    avg_num_neigh = mean(num_neigh);
    keep_mask = heavy_mask & (num_neigh > max(2 * double(D), double(heavy_factor_tmp) * avg_num_neigh));

    if ~any(keep_mask)
        keep_mask = heavy_mask;
    end
    if ~any(keep_mask)
        keep_mask = true(1, numel(num_neigh));
    end
end

function keep_mask = local_build_phase2_tmp_keep_mask(D, num_neigh, heavy_mask, heavy_factor_tmp, use_twoD_floor)
    num_neigh = double(reshape(num_neigh, 1, []));
    if isempty(heavy_mask)
        heavy_mask = true(1, numel(num_neigh));
    else
        heavy_mask = logical(reshape(heavy_mask, 1, []));
    end

    avg_num_neigh = mean(num_neigh);
    thresh = double(heavy_factor_tmp) * avg_num_neigh;
    if use_twoD_floor
        thresh = max(thresh, 2 * double(D));
    end
    keep_mask = heavy_mask & (num_neigh > thresh);

    if ~any(keep_mask)
        keep_mask = heavy_mask;
    end
    if ~any(keep_mask)
        keep_mask = true(1, numel(num_neigh));
    end
end

function Est = local_pack_partial_estimator(hyper_param1, Vec_pp, Cen_pp, X_sorted, Y_sorted, hk, rhok, num_neigh, keep_mask)
    keep_mask = logical(reshape(keep_mask, 1, []));
    if numel(keep_mask) ~= size(Vec_pp, 2)
        error('manor_1_gpu:pack', 'keep_mask has wrong length.');
    end
    if ~any(keep_mask)
        keep_mask = true(1, size(Vec_pp, 2));
    end

    Est = struct();
    Est.hyper_param1 = hyper_param1;
    Est.Vec_pp       = Vec_pp(:, keep_mask);
    Est.Cen_pp       = Cen_pp(:, keep_mask);
    Est.X_Train      = X_sorted(:, keep_mask);
    Est.Y_pp         = Y_sorted(keep_mask);
    Est.rhok         = double(rhok);
    Est.axial_normal = double(hk);
    Est.axial_tan    = double(hk) * double(rhok);

    if ~isempty(num_neigh)
        num_neigh = double(reshape(num_neigh, 1, []));
        Est.num_neigh_pp = num_neigh(keep_mask);
    end
end

function [has_curve_data, curveX_sorted, curveTan_sorted] = local_prepare_curve_truth(estOpts, perm, D, N)
    curveX_sorted = [];
    curveTan_sorted = [];

    have_curveX = isfield(estOpts, 'curveXData') && ~isempty(estOpts.curveXData);
    have_curveTan = isfield(estOpts, 'curveTanData') && ~isempty(estOpts.curveTanData);
    has_curve_data = logical(local_get_struct_field(estOpts, 'flag_curveData', false)) || (have_curveX && have_curveTan);

    if ~has_curve_data
        return;
    end
    if ~(have_curveX && have_curveTan)
        error('manor_1_gpu:curveData', 'curveXData and curveTanData must both be provided when curve-data diagnostics are enabled.');
    end

    curveX = single(estOpts.curveXData);
    curveTan = single(estOpts.curveTanData);
    if ~isequal(size(curveX), [D, N])
        error('manor_1_gpu:curveXData', 'curveXData must have size D x N.');
    end
    if ~isequal(size(curveTan), [D, N])
        error('manor_1_gpu:curveTanData', 'curveTanData must have size D x N.');
    end

    curveX_sorted = curveX(:, perm);
    curveTan_sorted = curveTan(:, perm);
end

function [devVec, devCen, devCenTan] = local_compute_curve_diagnostics(curveX, curveTan, Vec_pp, Cen_pp, keep_mask)
    keep_mask = logical(reshape(keep_mask, 1, []));
    if isempty(keep_mask) || ~any(keep_mask)
        keep_mask = true(1, size(Vec_pp, 2));
    end

    devVec = local_compute_devvec(curveTan, Vec_pp, keep_mask);
    DevCenMat = curveX(:, keep_mask) - Cen_pp(:, keep_mask);
    devCenTan = mean(abs(sum(DevCenMat .* curveTan(:, keep_mask), 1)));
    devCen = mean(vecnorm(DevCenMat));
end

function devVec = local_compute_devvec(curveTan, Vec_pp, keep_mask)
    keep_mask = logical(reshape(keep_mask, 1, []));
    if isempty(keep_mask) || ~any(keep_mask)
        keep_mask = true(1, size(Vec_pp, 2));
    end
    dev_minus = vecnorm(curveTan(:, keep_mask) - Vec_pp(:, keep_mask));
    dev_plus = vecnorm(curveTan(:, keep_mask) + Vec_pp(:, keep_mask));
    devVec = mean(min(dev_minus, dev_plus));
end

function cst = local_eval_norm_var(evalOpts, evalOpts_True, Y_train_sorted)
    y_all = [];
    if isstruct(evalOpts) && isfield(evalOpts, 'Y_All') && ~isempty(evalOpts.Y_All)
        y_all = double(evalOpts.Y_All(:));
    elseif isstruct(evalOpts_True) && isfield(evalOpts_True, 'Y_All') && ~isempty(evalOpts_True.Y_All)
        y_all = double(evalOpts_True.Y_All(:));
    elseif isstruct(evalOpts) && isfield(evalOpts, 'Y_norm_all') && ~isempty(evalOpts.Y_norm_all)
        y_all = double(evalOpts.Y_norm_all(:));
    else
        y_all = double(Y_train_sorted(:));
        if isstruct(evalOpts_True) && isfield(evalOpts_True, 'Y_True') && ~isempty(evalOpts_True.Y_True)
            y_all = [y_all; double(evalOpts_True.Y_True(:))]; %#ok<AGROW>
        end
    end
    cst = max(var(y_all, 1), 1e-12);
end

function s = local_set_default(s, name, value)
    if ~isfield(s, name)
        s.(name) = value;
    end
end

function local_require_finite_array(x, name)
    if ~all(isfinite(double(x(:))))
        error('manor_1_gpu:nonfinite', 'Non-finite values detected in %s.', name);
    end
end

function local_require_finite_scalar(x, name)
    if ~(isscalar(x) && isfinite(double(x)))
        error('manor_1_gpu:nonfinite', 'Non-finite scalar detected in %s.', name);
    end
end

function out = local_get_struct_field(s, name, defaultValue)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        out = s.(name);
    else
        out = defaultValue;
    end
end

function mask = local_build_iter_eval_mask(K, estOpts)
    mask = false(1, K+1);
    custom_idx = local_get_struct_field(estOpts, 'iter_eval_indices', []);
    schedule = lower(string(local_get_struct_field(estOpts, 'iter_eval_schedule', 'adaptive')));

    if isnumeric(custom_idx) && ~isempty(custom_idx)
        idx = unique(round(double(custom_idx(:).')));
        idx = idx(isfinite(idx));
        idx = idx(idx >= 0 & idx <= K);
        mask(idx + 1) = true;
    elseif schedule == "all"
        mask(:) = true;
    else
        warmup = max(0, min(K, round(double(estOpts.iter_eval_warmup))));
        mid_stride = max(1, round(double(estOpts.iter_eval_mid_stride)));
        if isempty(estOpts.iter_eval_late_start)
            late_start = max(warmup + 1, ceil(K / 2));
        else
            late_start = max(warmup + 1, min(K, round(double(estOpts.iter_eval_late_start))));
        end
        late_stride = max(1, round(double(estOpts.iter_eval_late_stride)));

        mask(1) = true;
        if warmup > 0
            mask(2:(warmup+1)) = true;
        end
        for k = (warmup + 1):K
            if k >= late_start
                if mod(k - late_start, late_stride) == 0
                    mask(k+1) = true;
                end
            else
                if mod(k - (warmup + 1), mid_stride) == 0
                    mask(k+1) = true;
                end
            end
        end
    end

    mask(1) = true;
    mask(end) = true;
end

function [X_Test_iter, evalOpts_True_iter, info] = local_prepare_iter_eval_subset(X_Test, evalOpts_True, estOpts)
    N_Test = size(X_Test, 2);
    info = struct('used_subset', false, 'count', N_Test, 'full_count', N_Test, ...
                  'fraction', 1.0, 'indices', 1:N_Test);
    X_Test_iter = X_Test;
    evalOpts_True_iter = evalOpts_True;

    if N_Test <= 0
        return;
    end

    frac = double(local_get_struct_field(estOpts, 'iter_eval_subsample_fraction', 1.0));
    frac = max(0, min(1, frac));
    nmin = max(1, round(double(local_get_struct_field(estOpts, 'iter_eval_subsample_min', 1))));
    nmax = max(1, round(double(local_get_struct_field(estOpts, 'iter_eval_subsample_max', N_Test))));
    target = min(N_Test, max(nmin, min(nmax, round(frac * N_Test))));

    if target >= N_Test
        return;
    end

    seed = round(double(local_get_struct_field(estOpts, 'iter_eval_subsample_seed', 1)));
    rs = RandStream('mt19937ar', 'Seed', seed);
    idx = sort(randperm(rs, N_Test, target));

    X_Test_iter = X_Test(:, idx);
    evalOpts_True_iter = local_apply_truth_subset(evalOpts_True, idx);
    info.used_subset = true;
    info.count = target;
    info.full_count = N_Test;
    info.fraction = target / max(N_Test, 1);
    info.indices = idx;
end

function evalOpts_True_sub = local_apply_truth_subset(evalOpts_True, idx)
    evalOpts_True_sub = evalOpts_True;
    if isfield(evalOpts_True_sub, 'Y_True') && ~isempty(evalOpts_True_sub.Y_True)
        evalOpts_True_sub.Y_True = evalOpts_True_sub.Y_True(idx);
    end
    if isfield(evalOpts_True_sub, 'V_True') && ~isempty(evalOpts_True_sub.V_True)
        evalOpts_True_sub.V_True = evalOpts_True_sub.V_True(:, idx);
    end
    if isfield(evalOpts_True_sub, 'C_True') && ~isempty(evalOpts_True_sub.C_True)
        evalOpts_True_sub.C_True = evalOpts_True_sub.C_True(:, idx);
    end
    if isfield(evalOpts_True_sub, 'curveTanData_True') && ~isempty(evalOpts_True_sub.curveTanData_True)
        evalOpts_True_sub.curveTanData_True = evalOpts_True_sub.curveTanData_True(:, idx);
    end
    if isfield(evalOpts_True_sub, 'curveXData_True') && ~isempty(evalOpts_True_sub.curveXData_True)
        evalOpts_True_sub.curveXData_True = evalOpts_True_sub.curveXData_True(:, idx);
    end
end

function label = local_iter_eval_scope_label(info, use_full_final_eval)
    if use_full_final_eval
        label = 'full_final';
    elseif info.used_subset
        label = 'subset_cached';
    else
        label = 'full_cached';
    end
end

function n = local_iter_eval_test_count(info, use_full_final_eval)
    if use_full_final_eval
        n = info.full_count;
    else
        n = info.count;
    end
end


function vals = local_collect_evalinfo_numeric_field(EvalInfo, field_name)
    if isempty(EvalInfo)
        vals = [];
        return;
    end

    vals = NaN(1, numel(EvalInfo));
    for i = 1:numel(EvalInfo)
        e = EvalInfo(i);
        if isstruct(e) && isfield(e, field_name) && ~isempty(e.(field_name))
            v = double(e.(field_name));
            if isscalar(v) && isfinite(v)
                vals(i) = v;
            end
        end
    end
end

function vals = local_collect_evalinfo_field(EvalInfo, field_name)
    if isempty(EvalInfo)
        vals = cell(1,0);
        return;
    end

    vals = cell(1, numel(EvalInfo));
    for i = 1:numel(EvalInfo)
        e = EvalInfo(i);
        if isstruct(e) && isfield(e, field_name) && ~isempty(e.(field_name))
            vals{i} = char(string(e.(field_name)));
        else
            vals{i} = '';
        end
    end
end

function Cen_out = local_project_center_off_tangent(Cen_in, Vec_in, X_in)
    Cen_out = single(Cen_in);
    if isempty(Cen_out)
        return;
    end
    denom = sum(single(Vec_in).^2, 1);
    denom = max(denom, single(1e-12));
    alpha = sum(single(Vec_in) .* (single(Cen_out) - single(X_in)), 1) ./ denom;
    Cen_out = single(Cen_out - single(Vec_in) .* alpha);
end

function vals = local_collect_phase2_profile_field(profiles, field_name)
    if isempty(profiles)
        vals = [];
        return;
    end

    if isstruct(profiles)
        profiles = {profiles};
    end

    vals = NaN(1, numel(profiles));
    sum_field_name = [field_name, '_sum'];

    for i = 1:numel(profiles)
        p = profiles{i};
        if ~isstruct(p)
            continue;
        end

        used = false;
        if isfield(p, field_name) && ~isempty(p.(field_name))
            v = double(p.(field_name));
            if isscalar(v)
                if isfinite(v)
                    vals(i) = v;
                    used = true;
                end
            elseif isvector(v)
                good = v(isfinite(v));
                if ~isempty(good)
                    vals(i) = sum(good);
                    used = true;
                end
            end
        end

        if ~used && isfield(p, sum_field_name) && ~isempty(p.(sum_field_name))
            sv = double(p.(sum_field_name));
            if isscalar(sv) && isfinite(sv)
                vals(i) = sv;
            end
        end
    end
end



function info = local_empty_eval_info()
    info = struct();
    info.totalCount = NaN;
    info.wrapper_total_time = NaN;
    info.wrapper_init_time = NaN;
    info.wrapper_unpack_truth_time = NaN;
    info.wrapper_cache_test_time = NaN;
    info.wrapper_mex_eval_time = NaN;
    info.wrapper_normalize_time = NaN;
    info.used_cached_test = false;
    info.profile_timers = false;
    info.mex_profile = struct();
    info.eval_backend = 'unknown';
    info.input_test_count = 0;
    info.keep_test_cache = false;
    info.skipped = true;
    info.eval_scope = 'uninitialized';
    info.test_count = 0;
end

function local_clear_eval_test_cache()
    try
        manor_1_eval_gpu('clear_test');
    catch
    end
end
