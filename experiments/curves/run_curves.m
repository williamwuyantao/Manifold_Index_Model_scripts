function run_curves(model, D, N_values, repetitions, dataset_root, output_root)
%RUN_CURVES Fresh repetitions with the manuscript's one-dimensional settings.
% model: 'single_index' or 'meyer_helix'. Build the backend once beforehand.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'experiments','curves','backend'));
addpath(fullfile(root,'dataset_generator','curves'));
settings = jsondecode(fileread(fullfile(root,'config','curve_experiments.json')));
p = settings.(model).(sprintf('D%d',D));
if nargin < 3 || isempty(N_values), N_values = p.N_values; end
if nargin < 4 || isempty(repetitions), repetitions = 3; end
if nargin < 5 || isempty(dataset_root), dataset_root = fullfile(root,'dataset','curves'); end
if nargin < 6 || isempty(output_root), output_root = fullfile(root,'reproduced','curves',model,sprintf('D%d',D)); end
if ~isfolder(output_root), mkdir(output_root); end
f = @(t)64*t.^3-96*t.^2+36*t-2+exp(t)+0.8*cos(pi*t)+2*sin(5*pi*t+0.1);
if strcmp(model,'meyer_helix')
    pool = fullfile(dataset_root,'meyer_helix',sprintf('D%d.h5',D));
    poolX = h5read(pool,'/gamma/X_UniCstDevSig');
    poolQ = h5read(pool,'/gamma/Xclean');
    poolT = h5read(pool,'/gamma/Tan');
    poolU = reshape(h5read(pool,'/gamma/Xlatent'),1,[]);
    length_gamma = double(h5read(pool,'/gamma/Len'));
    if size(poolX,1) ~= D
        poolX = poolX'; poolQ = poolQ'; poolT = poolT';
    end
end
for N = reshape(N_values,1,[])
    for repetition = 1:repetitions
        target = fullfile(output_root,sprintf('N%d-r%02d.mat',N,repetition));
        if isfile(target), continue; end
        rng(1000000*D + N + repetition,'twister');
        rng_before_data = rng;
        if strcmp(model,'single_index')
            [X,Q,T,U,length_gamma] = generate_single_index(N,D,settings.normal_sigma);
            sample_indices = [];
        else
            sample_indices = randperm(size(poolX,2),N);
            X = poolX(:,sample_indices); Q = poolQ(:,sample_indices);
            T = poolT(:,sample_indices); U = poolU(sample_indices);
        end
        Yclean = f(U);
        if strcmp(model,'meyer_helix'), Yclean = length_gamma*Yclean; end
        Yclean = Yclean-mean(Yclean);
        K = max(0,ceil(2*log(N)+p.K_offset));
        est = struct('fHolder',2,'K',K,'hyper_param1',p.K_offset,'numRiter',64, ...
            'length',length_gamma,'flag_itertesting',true,'flag_curveData',true, ...
            'curveXData',Q,'curveTanData',T,'silent',true);
        eval_options = struct('hyper_param2',p.eval_rho_multipliers, ...
            'length',length_gamma,'fHolder',2,'Test_ratio',0.1,'silent',true);
        rng_before_fit = rng;
        performance = manor_1_performance(p.response_resolution_denominator, ...
            X,Yclean,settings.response_noise_relative*std(Yclean),est,eval_options);
        parameters = p;
        parameters.N = N; parameters.D = D; parameters.K = K;
        parameters.repetition = repetition; parameters.model = model;
        save(target,'parameters','performance','sample_indices','rng_before_data','rng_before_fit','-v7.3');
        if strcmp(model,'single_index')
            folder = fullfile(output_root,'datasets');
            if ~isfolder(folder), mkdir(folder); end
            save(fullfile(folder,sprintf('N%d-r%02d.mat',N,repetition)), ...
                'X','Q','T','U','Yclean','rng_before_data','-v7.3');
        end
    end
end
end
