function build_manor_1_mex()
% BUILD_MANOR_1_MEX
% Build MANOR CUDA MEX files used by manor_1_gpu.m and manor_1_eval_gpu.m.

root = fileparts(mfilename('fullpath'));
if isempty(root)
    root = pwd;
end

phase1 = fullfile(root, 'manor_1_phase1_cuda_mex.cu');
phase2 = fullfile(root, 'manor_1_phase2_cuda_mex.cu');
evalcu = fullfile(root, 'manor_1_eval_mex.cu');

if ~isfile(phase1), error('Missing file: %s', phase1); end
if ~isfile(phase2), error('Missing file: %s', phase2); end
if ~isfile(evalcu), error('Missing file: %s', evalcu); end

MAX_D_MANOR_PHASE = 64;
MAX_R_MANOR_PHASE = 160;
MAX_D_MANOR_EVAL  = 64;
COV_D_THRESHOLD   = 64;

% Clean C++17 build with no duplicate std warnings.
% IMPORTANT:
%   - Keep CXXFLAGS fixed to c++17 explicitly.
%   - Do NOT use -warn-spills if you want a warning-free build log.

commonMexArgs = { ...
    '-O', ...
    '-largeArrayDims', ...
    ['CXXFLAGS=--compiler-options=-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread,-fwrapv ' ...
     '-std=c++17 ' ...
     '-DMX_COMPAT_64 ' ...
     '-DMATLAB_DEFAULT_RELEASE=R2017b ' ...
     '-DUSE_MEX_CMD ' ...
     '--compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE'], ...
    'NVCCFLAGS=$NVCCFLAGS --use_fast_math -lineinfo -Xptxas=-O3,-dlcm=ca' ...
};

phaseArgs = [commonMexArgs, { ...
    sprintf('-DMAX_D_MANOR=%d', MAX_D_MANOR_PHASE), ...
    sprintf('-DMAX_R_MANOR=%d', MAX_R_MANOR_PHASE) ...
}];

evalArgs = [commonMexArgs, { ...
    sprintf('-DMAX_D_MANOR_EVAL=%d', MAX_D_MANOR_EVAL), ...
    sprintf('-DCOV_D_THRESHOLD=%d', COV_D_THRESHOLD) ...
}];

fprintf('\n[build] Building phase 1 CUDA MEX...\n');
mexcuda(phaseArgs{:}, '-output', fullfile(root, 'manor_1_phase1_cuda_mex'), phase1);

fprintf('\n[build] Building phase 2 CUDA MEX...\n');
mexcuda(phaseArgs{:}, '-output', fullfile(root, 'manor_1_phase2_cuda_mex'), phase2);

fprintf('\n[build] Building eval CUDA MEX...\n');
mexcuda(evalArgs{:}, '-output', fullfile(root, 'manor_1_eval_mex'), evalcu);

fprintf('\n[build] Done.\n');
end
