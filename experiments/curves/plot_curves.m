function plot_curves(model,D,result_root,output_file)
%PLOT_CURVES Plot means and sample standard deviations of fresh curve runs.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 3 || isempty(result_root)
    result_root = fullfile(root,'reproduced','curves',model,sprintf('D%d',D));
end
if nargin < 4 || isempty(output_file)
    output_file = fullfile(result_root,sprintf('%s_D%d.pdf',model,D));
end
files = dir(fullfile(result_root,'N*-r*.mat'));
observations = cell(1,numel(files));
for i = 1:numel(files)
    observations{i} = load(fullfile(files(i).folder,files(i).name),'parameters','performance');
end
assert(~isempty(observations),'No curve results found.');
sizes = unique(cellfun(@(r)r.parameters.N,observations));
fields = {'RelMSE_Test','DevVec_Test'};
labels = {'Relative mean squared error','Mean tangent-space error'};
colors = [0,.447,.741; .850,.325,.098; .929,.694,.125];
figure('Color','w','Renderer','painters','Position',[100,100,900,350]);
for panel = 1:2
    subplot(1,2,panel); hold on;
    means = zeros(numel(sizes),3); deviations = means;
    for i = 1:numel(sizes)
        selected = observations(cellfun(@(r)r.parameters.N==sizes(i),observations));
        assert(numel(selected)>=2,'At least two repetitions are needed for error bars.');
        values = cellfun(@(r)reshape(double(r.performance.(fields{panel})),1,[]), ...
            selected,'UniformOutput',false);
        values = vertcat(values{:});
        assert(size(values,2)==3 && all(isfinite(values(:))) && all(values(:)>0));
        means(i,:) = mean(values,1); deviations(i,:) = std(values,0,1);
    end
    for c = 1:3
        line = polyfit(log(sizes(:)),log(means(:,c)),1);
        errorbar(sizes,means(:,c),deviations(:,c),'o-', ...
            'Color',colors(c,:),'LineWidth',1.3,'MarkerSize',5, ...
            'DisplayName',sprintf('slope = %.2f',line(1)));
    end
    set(gca,'XScale','log','YScale','log','FontName','Times New Roman','FontSize',11);
    xlabel('n'); title(labels{panel}); legend('Location','northeast'); box on; axis tight;
end
exportgraphics(gcf,output_file,'ContentType','vector');
end
