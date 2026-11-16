function varargout = manor_1_eval_gpu(varargin)
% MANOR_1_EVAL_GPU
%
% Main usage:
%   [MSE_Test,DevCen_Test,DevCen_tan_Test,DevVec_Test] = ...
%       manor_1_eval_gpu(Estimator, X_Test, evalOpts, evalOpts_True)
%
% Optional 5th output:
%   [MSE_Test,DevCen_Test,DevCen_tan_Test,DevVec_Test,info] = ...
%       manor_1_eval_gpu(...)
%
% Optional command usage for repeated evaluation with fixed test set:
%   manor_1_eval_gpu('cache_test', X_Test, evalOpts_True)
%   manor_1_eval_gpu('clear_test')
%   manor_1_eval_gpu('get_last_profile')
%   manor_1_eval_gpu('free')
%
% Repeated-iteration pattern:
%   manor_1_eval_gpu('cache_test', X_Test, evalOpts_True);
%   evalOpts.use_cached_test = true;
%   [MSE,DC,DCtan,DV,info] = manor_1_eval_gpu(Estimator_k, [], evalOpts, struct());
%   manor_1_eval_gpu('clear_test');
%
% Notes:
%   - This wrapper keeps the original "one-shot" contract.
%   - Internally, the CUDA MEX performs chunking itself and chooses
%     chunk size using actual GPU free memory after estimator init.
%   - If cache_test / use_cached_test is used, the test set can stay on GPU
%     across many repeated evaluation calls.

    if nargin >= 1 && (ischar(varargin{1}) || isstring(varargin{1}))
        [varargout{1:nargout}] = local_command_dispatch(varargin{:});
        return;
    end

    if nargin < 3 || isempty(varargin{3})
        evalOpts = struct();
    else
        evalOpts = varargin{3};
    end

    if nargin < 4 || isempty(varargin{4})
        evalOpts_True = struct();
    else
        evalOpts_True = varargin{4};
    end

    Estimator = varargin{1};
    X_Test    = varargin{2};

    if ~exist('manor_1_eval_mex', 'file')
        error(['Missing MEX function manor_1_eval_mex. ', ...
               'Compile it first with build_manor_1_mex.']);
    end

    t_total = tic;

    use_cached_test  = local_get_opt(evalOpts, 'use_cached_test', false);
    cache_test_data  = local_get_opt(evalOpts, 'cache_test_data', false);
    keep_test_cache  = local_get_opt(evalOpts, 'keep_test_cache', false);

    hyper_param2 = local_pick_hyper_param2(Estimator, evalOpts);
    m = numel(hyper_param2);

    mexOpts = local_build_mex_opts(evalOpts, Estimator);

    t_init = tic;
    manor_1_eval_mex('init', Estimator, hyper_param2, mexOpts);
    init_time = toc(t_init);
    estCleanup = onCleanup(@() local_safe_cmd('clear_estimator')); %#ok<NASGU>

    unpack_time = 0;
    cache_time = 0;

    if use_cached_test
        t_eval = tic;
        [sMSE, sDevCen, sDevCenTan, sDevVec, totalCount] = ...
            manor_1_eval_mex('eval_cached');
        mex_eval_time = toc(t_eval);
    else
        t_unpack = tic;
        [Y_True, V_True, C_True] = local_unpack_truth(evalOpts_True, X_Test);
        unpack_time = toc(t_unpack);

        if cache_test_data
            t_cache = tic;
            manor_1_eval_mex('cache_test', X_Test, Y_True, V_True, C_True);
            cache_time = toc(t_cache);
            if ~keep_test_cache
                testCleanup = onCleanup(@() local_safe_cmd('clear_test')); %#ok<NASGU>
            end
            t_eval = tic;
            [sMSE, sDevCen, sDevCenTan, sDevVec, totalCount] = ...
                manor_1_eval_mex('eval_cached');
            mex_eval_time = toc(t_eval);
        else
            t_eval = tic;
            [sMSE, sDevCen, sDevCenTan, sDevVec, totalCount] = ...
                manor_1_eval_mex('eval_full', X_Test, Y_True, V_True, C_True);
            mex_eval_time = toc(t_eval);
        end
    end

    if totalCount <= 0
        error('No test samples were evaluated.');
    end

    t_norm = tic;
    MSE_Test        = single(double(sMSE(:))       / double(totalCount));
    DevCen_Test     = single(double(sDevCen(:))    / double(totalCount));
    DevCen_tan_Test = single(double(sDevCenTan(:)) / double(totalCount));
    DevVec_Test     = double(sDevVec(:))           / double(totalCount);
    normalize_time = toc(t_norm);

    if m == 1
        MSE_Test        = reshape(MSE_Test,        [1,1]);
        DevCen_Test     = reshape(DevCen_Test,     [1,1]);
        DevCen_tan_Test = reshape(DevCen_tan_Test, [1,1]);
        DevVec_Test     = reshape(DevVec_Test,     [1,1]);
    end

    info = struct();
    info.totalCount = double(totalCount);
    info.wrapper_total_time = toc(t_total);
    info.wrapper_init_time = init_time;
    info.wrapper_unpack_truth_time = unpack_time;
    info.wrapper_cache_test_time = cache_time;
    info.wrapper_mex_eval_time = mex_eval_time;
    info.wrapper_normalize_time = normalize_time;
    info.used_cached_test = logical(use_cached_test || cache_test_data);
    info.profile_timers = logical(local_get_opt(evalOpts, 'profile_timers', false));

    try
        info.mex_profile = manor_1_eval_mex('get_last_profile');
    catch
        info.mex_profile = struct();
    end

    % Compatibility with the curve-estimator EvalInfo layout.
    if use_cached_test
        info.eval_backend = 'subset_cached';
        info.input_test_count = 0;
    elseif cache_test_data
        info.eval_backend = 'subset_cached';
        info.input_test_count = size(X_Test, 2);
    else
        info.eval_backend = 'full';
        info.input_test_count = size(X_Test, 2);
    end
    info.keep_test_cache = logical(keep_test_cache);
    info.skipped = false;
    info.eval_scope = 'direct';
    info.test_count = double(totalCount);

    varargout = {MSE_Test, DevCen_Test, DevCen_tan_Test, DevVec_Test};
    if nargout >= 5
        varargout{5} = info;
    end
end


function varargout = local_command_dispatch(varargin)
    cmd = lower(string(varargin{1}));

    switch cmd
        case "cache_test"
            if nargin < 3
                error('Usage: manor_1_eval_gpu(''cache_test'', X_Test, evalOpts_True)');
            end
            X_Test = varargin{2};
            evalOpts_True = varargin{3};
            [Y_True, V_True, C_True] = local_unpack_truth(evalOpts_True, X_Test);
            manor_1_eval_mex('cache_test', X_Test, Y_True, V_True, C_True);
            varargout = {};
        case "clear_test"
            local_safe_cmd('clear_test');
            varargout = {};
        case "get_last_profile"
            if exist('manor_1_eval_mex', 'file')
                varargout = {manor_1_eval_mex('get_last_profile')};
            else
                varargout = {struct()};
            end
        case "free"
            local_safe_cmd('free');
            varargout = {};
        otherwise
            error('Unknown command "%s".', cmd);
    end
end


function mexOpts = local_build_mex_opts(evalOpts, Estimator)
    D = size(Estimator.X_Train, 1);

    mexOpts = struct();

    % Tangent computation:
    %   - exact_tangent_small_d = 1  => exact Jacobi eig on GPU for D <= 64
    %   - power_iter_steps       => only used for the large-D fallback path
    mexOpts.power_iter_steps = int32(local_get_opt(evalOpts, 'power_iter_steps', 12));
    mexOpts.sum_sparse_threshold = single(local_get_opt(evalOpts, 'sum_sparse_threshold', 1e1));
    mexOpts.count_sparse_threshold = int32(local_get_opt(evalOpts, 'count_sparse_threshold', 2 * D));
    mexOpts.fallback_exp_scale = single(local_get_opt(evalOpts, 'fallback_exp_scale', 2));
    mexOpts.eps_weight = single(local_get_opt(evalOpts, 'eps_weight', 1e-20));

    % Internal chunking / memory sizing
    if isfield(evalOpts,'manual_chunk') && ~isempty(evalOpts.manual_chunk)
        mexOpts.manual_chunk = int32(max(0, floor(double(evalOpts.manual_chunk))));
    else
        mexOpts.manual_chunk = int32(0);
    end
    mexOpts.auto_chunk_mem_fraction = single(local_get_opt(evalOpts, 'auto_chunk_mem_fraction', 0.80));
    mexOpts.auto_chunk_safety_bytes = double(local_get_opt(evalOpts, 'auto_chunk_safety_bytes', 0));
    mexOpts.auto_chunk_min = int32(local_get_opt(evalOpts, 'auto_chunk_min', 1024));
    mexOpts.auto_chunk_max = int32(local_get_opt(evalOpts, 'auto_chunk_max', 131072));
    mexOpts.auto_chunk_target_pairs = double(local_get_opt(evalOpts, 'auto_chunk_target_pairs', 8.0e9));

    % Optional behavior flags
    mexOpts.enable_host_double_buffer = int32(local_get_opt(evalOpts, 'enable_host_double_buffer', 1));
    mexOpts.profile_timers = int32(logical(local_get_opt(evalOpts, 'profile_timers', false)));
    mexOpts.exact_tangent_small_d = int32(logical(local_get_opt(evalOpts, 'exact_tangent_small_d', 1)));
    mexOpts.silent = int32(logical(local_get_opt(evalOpts, 'silent', false)));

end


function hyper_param2 = local_pick_hyper_param2(Estimator, evalOpts)
    if isfield(Estimator,'hyper_param2') && ~isempty(Estimator.hyper_param2)
        hyper_param2 = Estimator.hyper_param2;
    elseif isfield(evalOpts,'hyper_param2') && ~isempty(evalOpts.hyper_param2)
        hyper_param2 = evalOpts.hyper_param2;
    else
        hyper_param2 = 1;
        if ~logical(local_get_opt(evalOpts, 'silent', false))
            fprintf('[warn] No hyper_param2 found in Estimator or evalOpts. Defaulting to 1.\n');
        end
    end
    hyper_param2 = single(hyper_param2(:));
end

function [Y_True, V_True, C_True] = local_unpack_truth(evalOpts_True, X_Test)
    if isempty(X_Test)
        error(['X_Test is empty. If you want to use cached test data, set ', ...
               'evalOpts.use_cached_test = true and make sure cache_test was called.']);
    end

    required = {'Y_True', 'V_True', 'C_True'};
    for i = 1:numel(required)
        if ~isfield(evalOpts_True, required{i})
            error('evalOpts_True must contain %s.', required{i});
        end
    end

    [D, N_Test] = size(X_Test);

    Y_True = evalOpts_True.Y_True;
    V_True = evalOpts_True.V_True;
    C_True = evalOpts_True.C_True;

    if ~isnumeric(Y_True) || ~isreal(Y_True)
        error('evalOpts_True.Y_True must be real numeric.');
    end
    if ~isnumeric(V_True) || ~isreal(V_True)
        error('evalOpts_True.V_True must be real numeric.');
    end
    if ~isnumeric(C_True) || ~isreal(C_True)
        error('evalOpts_True.C_True must be real numeric.');
    end

    if numel(Y_True) ~= N_Test
        error('evalOpts_True.Y_True length does not match size(X_Test,2).');
    end
    if size(V_True,1) ~= D || size(V_True,2) ~= N_Test
        error('evalOpts_True.V_True must have size [D, N_Test].');
    end
    if size(C_True,1) ~= D || size(C_True,2) ~= N_Test
        error('evalOpts_True.C_True must have size [D, N_Test].');
    end
end


function v = local_get_opt(s, name, defaultValue)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultValue;
    end
end


function local_safe_cmd(cmd)
    try
        if exist('manor_1_eval_mex', 'file')
            manor_1_eval_mex(cmd);
        end
    catch
    end
end
