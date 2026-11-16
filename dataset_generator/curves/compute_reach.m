function [loc_reach_arr, appendix] = compute_reach(Xclean_PCA, Tan_PCA, dts, Mnode, opt)
%
% function [loc_reach_arr] = compute_reach(Xclean_SVD, Tan_SVD, Mnode)
%
% IN:
%       Xclean_PCA : Dx(N+1) matrix of N samples in D dimension for the curve (already change basis via PCA)
%       Tan_PCA    : Dx(N+1) matrix of tangentent vectors for each point in Xclean_PCA, not necessarily unit vector
%       dts        : 1xN array of time between each point on the curve
%       Mnode      : number of points we choose on the curve (to simplify the computation cost when N is large)
%       opt        : struct that contains the following:
%           flag_DimReduction_PCA : logical value indicating whether we want to find length&reach for projected curve (dimension reduction via PCA)
%           flag_DimReduction_RND : logical value indicating whether we want to find length&reach for projected curve (dimension reduction via random projection)
%
% OUT:
%   loc_reach_arr  : 1xN array of estimated local reach for each point Xclean_SVD
%   appendix       : struct that contains the following:
%           PCAproj_gamma_Len_arr_d : 1xD array, the i-th entry is the length of the projected curve (onto the first i principal components)
%           PCAproj_gamma_Reach_arr_d : 1xD array, the i-th entry is the global reach of the projected curve (onto the first i principal components)
%           RNDproj_gamma_Len_arr_d : 1xD array, the i-th entry is the length of the projected curve (onto the first i subspace of random projection)
%           RNDproj_gamma_Reach_arr_d : 1xD array, the i-th entry is the global reach of the projected curve (onto the first i subspace of random projection)

% (c) Mauro Maggioni, Yantao Wu

[D, N]          = size(Xclean_PCA); N = N-1;
number_RND      = 10;

PCAproj_gamma_Len_arr_d         = NaN(1,D);
PCAproj_gamma_Reach_arr_d       = NaN(1,D);
RNDproj_gamma_Len_arr_d         = NaN(number_RND,D);
RNDproj_gamma_Reach_arr_d       = NaN(number_RND,D);

Xclean_RND = cell(1,number_RND);
Tan_RND    = cell(1,number_RND);
for i = 1:number_RND
    oA            = orth( randn( D, D ) )' ;
    Xclean_RND{i} = oA * Xclean_PCA;
    Tan_RND{i}    = oA * Tan_PCA;
end

for d = 2:D
    %simple test for d=D/3
    if d<D && ~ismember(d,floor([D/4, D/3, D/2, 2*D/3, 3*D/4])), continue; end
    if opt.flag_DimReduction_PCA || d==D
        C_Xclean_d              = parallel.pool.Constant(Xclean_PCA(1:d,1:N));
        arclength_d             = vecnorm(Tan_PCA(1:d, 1:N)) .* dts; %vecnorm(Xclean_d(1:d,2:N) - Xclean_d(1:d,1:N-1));
        PCAproj_gamma_Len_arr_d(d) = sum(arclength_d);
        Tan_PCA_unit            = normcols(Tan_PCA(1:d,1:N+1));
        Kapval_d                = vecnorm(Tan_PCA_unit(1:d,2:N+1)-Tan_PCA_unit(1:d,1:N)) ./ arclength_d ; %vecnorm(Tan_d(:,2:N) - Tan_d(:,1:N-1)) ./ arclength_d;
        loc_reach_arr_d         = 1./Kapval_d;

        idn_subset_arr          = sort([floor(linspace(1,N,Mnode)),find(arclength_d > min(maxk(arclength_d,Mnode))), find(Kapval_d > min(maxk(Kapval_d,Mnode))),N]);
        idn_subset_arr_next     = idn_subset_arr(2:end);

        loc_reach_subset_candiate       = loc_reach_arr_d(idn_subset_arr(1:end-1));
        C_arclength_d           = parallel.pool.Constant(arclength_d);
        C_Kapval_d              = parallel.pool.Constant(Kapval_d);
        parfor i = 1:size(loc_reach_subset_candiate,2)
            idn                 = idn_subset_arr(i);
            idn_next            = idn_subset_arr_next(i);
            currX               = C_Xclean_d.Value(:,idn);
            distonCurve_node_n_left     = cumsum(C_arclength_d.Value(1:idn),'reverse')-C_arclength_d.Value(idn);
            distonCurve_node_n_right    = cumsum(C_arclength_d.Value(idn+1:N-1));
            idm_left                    = find(distonCurve_node_n_left  >= 2/C_Kapval_d.Value(idn));
            idm_right                   = find(distonCurve_node_n_right >= 2/C_Kapval_d.Value(idn)) + idn;
            if ~isempty([idm_left,idm_right])
                candidate               = min(vecnorm(C_Xclean_d.Value(:,[idm_left,idm_right])-currX))/2;
                if candidate < min([ 1/min(C_Kapval_d.Value(idn:idn_next)), vecnorm(C_Xclean_d.Value(:,[max(idm_left),min(idm_right)])-currX)/2])   %candidate<mean(loc_reach_arr(last_n:n))*0.85
                    loc_reach_subset_candiate(i) = candidate;
                end
            end
        end
        PCAproj_gamma_Reach_arr_d(d) = min([loc_reach_subset_candiate, loc_reach_arr_d]);
    end

    if d == D
        RNDproj_gamma_Len_arr_d(:,D)  = PCAproj_gamma_Len_arr_d(D);
        RNDproj_gamma_Reach_arr_d(:,D) = PCAproj_gamma_Reach_arr_d(D);
        for i = 1:size(loc_reach_subset_candiate,2)
            idn                 = idn_subset_arr(i);
            idn_next            = idn_subset_arr(i+1);
            candidate           = loc_reach_subset_candiate(i);
            loc_reach_arr_d(idn:idn_next) =  arrayfun(@(x) min(x,candidate),loc_reach_arr_d(idn:idn_next));
        end
        loc_reach_arr = loc_reach_arr_d;
    end

    if opt.flag_DimReduction_RND && d<D
        for rndi=1:number_RND

        C_Xclean_d              = parallel.pool.Constant(Xclean_RND{rndi}(1:d,:));
        arclength_d             = vecnorm(Tan_RND{rndi}(1:d,1:N)) .* dts; %vecnorm(Xclean_d(1:d,2:N) - Xclean_d(1:d,1:N-1));
        RNDproj_gamma_Len_arr_d(rndi,d) = sum(arclength_d);
        Tan_RND_unit            = normcols(Tan_RND{rndi}(1:d,1:N+1));
        Kapval_d                = vecnorm(Tan_RND_unit(1:d,2:N+1)-Tan_RND_unit(1:d,1:N)) ./ arclength_d ; %vecnorm(Tan_d(:,2:N) - Tan_d(:,1:N-1)) ./ arclength_d;
        loc_reach_arr_d         = 1./Kapval_d;

        idn_subset_arr          = sort([floor(linspace(1,N,Mnode)),find(arclength_d > min(maxk(arclength_d,Mnode))), find(Kapval_d > min(maxk(Kapval_d,Mnode))),N]);
        idn_subset_arr_next     = idn_subset_arr(2:end);

        loc_reach_subset_candiate       = loc_reach_arr_d(idn_subset_arr(1:end-1));
        C_arclength_d           = parallel.pool.Constant(arclength_d);
        C_Kapval_d              = parallel.pool.Constant(Kapval_d);
        parfor i = 1:size(loc_reach_subset_candiate,2)
            idn                 = idn_subset_arr(i);
            idn_next            = idn_subset_arr_next(i);
            currX               = C_Xclean_d.Value(:,idn);
            distonCurve_node_n_left     = cumsum(C_arclength_d.Value(1:idn),'reverse')-C_arclength_d.Value(idn);
            distonCurve_node_n_right    = cumsum(C_arclength_d.Value(idn+1:N-1));
            idm_left                    = find(distonCurve_node_n_left  >= 2/C_Kapval_d.Value(idn));
            idm_right                   = find(distonCurve_node_n_right >= 2/C_Kapval_d.Value(idn)) + idn;
            if ~isempty([idm_left,idm_right])
                candidate               = min(vecnorm(C_Xclean_d.Value(:,[idm_left,idm_right])-currX))/2;
                if candidate < min([ 1/min(C_Kapval_d.Value(idn:idn_next)), vecnorm(C_Xclean_d.Value(:,[max(idm_left),min(idm_right)])-currX)/2])   %candidate<mean(loc_reach_arr(last_n:n))*0.85
                    loc_reach_subset_candiate(i) = candidate;
                end
            end
        end
        RNDproj_gamma_Reach_arr_d(rndi,d) = min([loc_reach_subset_candiate, loc_reach_arr_d]);
        end
    end

end

appendix = struct('PCAproj_gamma_Len_arr_d',PCAproj_gamma_Len_arr_d,'PCAproj_gamma_Reach_arr_d',PCAproj_gamma_Reach_arr_d,...
                  'RNDproj_gamma_Len_arr_d',RNDproj_gamma_Len_arr_d,'RNDproj_gamma_Reach_arr_d',RNDproj_gamma_Reach_arr_d);


return
