function generate_meyer_pool(D,N,output_file,seed)
%GENERATE_MEYER_POOL Create a fresh predictor pool for run_curves.
% Requires the Statistics and Parallel Computing Toolboxes. The numerical
% reach calculation is the same one used by the archived curve generator.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fileparts(mfilename('fullpath')));
if nargin < 2 || isempty(N), N = 2000000; end
if nargin < 3 || isempty(output_file)
    output_file = fullfile(root,'dataset','curves','meyer_helix',sprintf('D%d.h5',D));
end
if nargin < 4 || isempty(seed), seed = 2000000+D; end
validateattributes(D,{'numeric'},{'integer','scalar','>=',3});
validateattributes(N,{'numeric'},{'integer','scalar','>=',100});
assert(~isfile(output_file),'Output already exists: %s',output_file);
folder = fileparts(output_file);
if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
rng(seed,'twister');
parameters = struct('Name','Meyer-Helix-L1-C1-variant1','N',N,'D',D, ...
    'normalSigma',0.5,'rescale','Reach','type_noise_off_curve','uniform_constant', ...
    'Info_DimReduction','None');
gamma = createCurve(parameters);
temporary = [output_file,'.partial'];
assert(~isfile(temporary),'Incomplete output already exists: %s',temporary);
fields = {'X_UniCstDevSig','Xclean','Tan','Xlatent','Len','Reach','normalSigma'};
for k = 1:numel(fields)
    name = fields{k}; value = gamma.(name);
    h5create(temporary,['/gamma/',name],size(value),'Datatype',class(value));
    h5write(temporary,['/gamma/',name],value);
end
h5writeatt(temporary,'/','manifold',parameters.Name);
h5writeatt(temporary,'/','ambient_dimension',D);
h5writeatt(temporary,'/','rows',size(gamma.Xclean,2));
h5writeatt(temporary,'/','generator_seed',seed);
h5writeatt(temporary,'/','predictor_array','gamma/X_UniCstDevSig');
h5writeatt(temporary,'/','state','complete');
movefile(temporary,output_file);
fprintf('%s\n',output_file);
end
