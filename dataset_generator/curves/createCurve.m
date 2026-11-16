 function [gamma] = createCurve( Pars )
% function [gamma] = createCurve( Pars )
% INPUT:
%   Pars : structure with parameters:
%       Name    : name for the type of curve to create, in the set {'Segment','Holder','PieceHolder','Parabola','SkewParabola','Helix','multiHelix','Meyer'}
%       D       : ambient dimension of data
%       N       : number of points to sample
%       normalSigma   : add Gaussian noise in the normal direction. Default: 0.5
%       rescale : true/false whether we will do rescaling, default: false
%       -- the following are optional, not required for some curves
%           M       : number of nodes to generate the curve (for curve such as Segment, Holder, PeiceHolder, the shape of underlying curve should be independent of N the number of samples we take)
%           KappaU  : upperbound for absolute value of curvature
%
% OUTPUT:
%   gamma   : structure with the following fields:
%       X_UniCstDevSig  : DxN matrix of N samples in D dimensions for the curve, with Gaussian deviation added, noise-level is a uniform constant.
%       Xclean          : DxN matrix of N samples in D dimensions on the curve, without Gaussian deviation added
%       Tan             : DxN matrix of unit tangent vectors to each of the N points on the curve
%       Xlatent         : 1xN vector of latent coordinates of the points on the curve, normalized inside the interval [0,1].
%       Len             : the length of the curve
%       Reach           : the scalar of the reach of the curve, the maximum distance from the curve without misclassification.
%       Kap             : 1x3 vector of min,mean,max curvature value
%       Kapval          : 1x(n-1) vector of curvature values on each point
%       Kapvec          : Dx(N-1) matrix of unit normal vectors to each of the N points on the curve
%       Diamater        : approximately diameter
%       arclength       : 1xN vector of length of each curve segments between points
%       SV              : 1xD vector of singular values of Xclean
%       -- may also contain X_RelCstDevSig : DxN matrix of N samples in D dimensions for the curve, with Gaussian deviation added, noise-level proportional to local reach.

gamma = struct('Name',Pars.Name);
if ~isfield(Pars,'rescale'),            Pars.rescale            = 'None';       end
if ~isfield(Pars,'normalSigma'),        Pars.normalSigma        = 0;            end
if ~isfield(Pars, 'Info_DimReduction'), Pars.Info_DimReduction  = 'None';       end

N                           = Pars.N;
D                           = Pars.D;
if isempty(gcp('nocreate')), parpool('Threads'); end
%% curve data generation: Xlatent(ordeded arrar 0,...,1), X, and Tan (not unit vector here)
switch( Pars.Name )
    case 'Line-segment' %length 1, Reach = infinity, rescale = 'None'
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1); X(1,:)=Xlatent;
        Tan             = zeros(D, N+1); Tan(1,:) = ones(1,N+1);
        gamma.Xlatent   = Xlatent;
        gamma.Xclean    = X;
        gamma.Tan       = Tan;

    case 'arc-circle' % length 1, Reach = 1/Pars.KappaU, rescale = 'None'
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        if Pars.KappaU == 0
            X(1,:)          = Xlatent-0.5;
            Tan(1,:)        = ones(1,N+1);
        else
            rad             = 1/ Pars.KappaU;
            theta           = Pars.KappaU;
            X(1,:)          = rad*sin(theta* (Xlatent-0.5));
            X(2,:)          = rad - rad*cos(theta* (Xlatent-0.5));
            Tan(1,:)        = cos(theta* (Xlatent-0.5));
            Tan(2,:)        = sin(theta* (Xlatent-0.5));
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean    = X;
        gamma.Tan       = Tan;

    case 'arc-circle-UBD'
        rad          = 1/ Pars.KappaU;
        D2           = D-2-2*floor(D/4);
        D3           = floor(D/4);
        D4           = floor(D/4);
        C2           = 1;
        C3           = 1.4;
        C4           = 0.7;
        Sigma_gamma  = 0.5;
        trun_normal1 = truncate(makedist('Normal','mu',0,'sigma',   Sigma_gamma),-0.9*rad,0.9*rad);
        %trun_normal2 = truncate(makedist('Normal','mu',0,'sigma',C2*Sigma_gamma),-2,2);
        %trun_normal3 = truncate(makedist('Normal','mu',0,'sigma',C3*Sigma_gamma),-2,2);
        %trun_normal4 = truncate(makedist('Normal','mu',0,'sigma',C4*Sigma_gamma),-2,2);

        Xlatent                 = sort(rand(1, N));
        Tan                     = zeros(D, N); % omit remaining D-2 coordinates with zero entries
        Xclean                  = zeros(D,N);  % omit remaining D-2 coordinates with zero entries
        X                       = zeros(D, N);
        X(3:D2+2,:)             = randn(D2, N)*(C2*Sigma_gamma);
        X(D2+3:D2+D3+2,:)       = randn(D3, N)*(C3*Sigma_gamma);
        X(D2+D3+3:D,:)          = randn(D4, N)*(C4*Sigma_gamma);

        Tan(1,:)             =       cos(Pars.KappaU* (Xlatent-0.5));
        Tan(2,:)             =       sin(Pars.KappaU* (Xlatent-0.5));
        Xclean(1,:)          =  rad*Tan(2,:); % rad*sin(KappaU* (Xlatent-0.5));
        Xclean(2,:)          = -rad*Tan(1,:); %-rad*cos(KappaU* (Xlatent-0.5));
        noise                =  Pars.KappaU*random(trun_normal1, 1, N) + 1;
        X(1,:)               = Xclean(1,:).*noise;
        X(2,:)               = Xclean(2,:).*noise;

        gamma                = struct('Len',1,'Reach',rad,'loc_reach_arr',rad+zeros(1,N),'Kapval',Pars.KappaU,'Kapvec',Xclean/rad,'Kap',[Pars.KappaU,Pars.KappaU,Pars.KappaU],...
                                      'X_RelCstDevSig',X,'Xlatent',Xlatent,'Xclean',Xclean,'Tan',Tan,'arclength',[0,diff(Xlatent)]);
        delete(gcp('nocreate'));
        return

    %for random segment curves, we use M nodes, and create curve by sorted latent
    case 'Segment'
        nodelatent          = sort(rand(1, Pars.M));
        dt                  = 1/(Pars.M-1);
        nodeKap             = randn(D, Pars.M);
        nodeKap             = normcols(nodeKap);
        nodeKap             = Pars.KappaU * nodeKap;
        nodeTan             = zeros(D,Pars.M);
        nodeTan(:,1)        = randn(D,1);
        nodeTan(:,1)        = normcols(nodeTan(:,1));
        nodeX               = zeros(D,Pars.M);
        nodeTan(:,2:Pars.M) = nodeTan(:, 1:Pars.M-1) + nodeKap(:, 1:Pars.M-1).*dt;
        nodeTan             = normcols(nodeTan);
        nodeX(:, 2:Pars.M)  = nodeX(:, 1:Pars.M-1) + nodeTan(:, 1:Pars.M-1).*dt;
        %use linear interpolation on above curve nodes to get Xlatent,X,Tan
        rndincrement        = rand(1,N)/(Pars.M-1);
        gamma.Xlatent       = nodelatent  + rndincrement;
        gamma.Tan           = nodeTan + nodeKap .* rndincrement;
        gamma.Xclean        = nodeX + nodeTan .* rndincrement;

    case 'Holder'  %curvacure vector is Holder continuous
        nodelatent          = sort(rand(1, Pars.M));
        dt                  = 1/(Pars.M-1);
        %fH:R^1->R^D is a randomly generated Holder continuous function, with Holder exponent=0.75, and Holder constant 10*Pars.KappaU, which we use to generate random curvature
        fH                  = generateDRandomHolderFunction([0,1], 0.75, 2^8, D);
        nodeKap             = fH(nodelatent)* D^(-0.5) *Pars.KappaU;
        nodeTan             = zeros(D,Pars.M);
        nodeTan(:,1)        = randn(D,1);
        nodeTan(:,1)        = normcols(nodeTan(:,1));
        nodeX               = zeros(D,Pars.M);
        nodeTan(:, 2:Pars.M)= nodeTan(:, 1:Pars.M-1) + nodeKap(:, 1:Pars.M-1).*dt;
        nodeTan             = normcols(nodeTan);
        nodeX(:, 2:Pars.M)  = nodeX(:, 1:Pars.M-1) + nodeTan(:, 1:Pars.M-1).*dt;
        %use quadratic interpolation on above curve nodes to get Xlatent,X,Tan
        rndincrement        = rand(1,N)*dt;
        gamma.Xlatent       = nodelatent + rndincrement;
        gamma.Tan           = nodeTan + nodeKap .* rndincrement;
        gamma.Xclean             = nodeX + nodeTan .* rndincrement + 0.5*nodeKap .* (rndincrement.^2);


    case 'PieceHolder' %the magnitude of curvature (not as a vector-valued function) is Holder continuous
        nodelatent          = sort(rand(1, Pars.M));
        dt                  = 1/(Pars.M-1);
        %fH are random Holder continuous function, with Holder exponent=0.7, Holder constant=1*Pars.KappaU
        %which we use to generate random curvature
        fH                  = generateRandomHolderFunction([0,1], 0.7, 2^8);
        nodeKapVal          = (rand(1) * Pars.KappaU*0.2) + fH(nodelatent)*1* Pars.KappaU;
        nodeKap             = randn(D, Pars.M);
        nodeKap             = normcols(nodeKap);
        nodeKap             = bsxfun(@times, nodeKap, nodeKapVal) ;
        nodeTan             = zeros(D,Pars.M);
        nodeTan(:,1)        = randn(D,1);
        nodeTan(:,1)        = normcols(nodeTan(:,1));
        nodeX               = zeros(D,Pars.M);
        nodeTan(:, 2:Pars.M)= nodeTan(:, 1:Pars.M-1) + nodeKap(:, 1:Pars.M-1).*dt;
        nodeTan             = normcols(nodeTan);
        nodeX(:, 2:Pars.M)  = nodeX(:, 1:Pars.M-1) + nodeTan(:, 1:Pars.M-1).*dt + 0.5*nodeKap(:, 1:Pars.M-1)*dt^2;
        %use quadratic interpolation on above curve nodes to get Xlatent,X,Tan
        rndincrement        = rand(1,N)*dt;
        gamma.Xlatent       = nodelatent + rndincrement;
        gamma.Tan           = nodeTan + nodeKap .* rndincrement;
        gamma.Xclean             = nodeX + nodeTan .* rndincrement + 0.5 * nodeKap .* (rndincrement.^2);

    %For analytic curves such as Parabola, SkewParabola, Helix, multiHelix,
    %Meyer, we do not use nodes, but we can calculate Xclean and Tan
    %directly.
    case 'Parabola'
        D          = max(D,2);
        gamma.Xlatent   = sort(rand(1, N));
        gamma.Xclean         = zeros(D,N);
        gamma.Xclean(1,:)    = gamma.Xlatent;
        gamma.Xclean(2,:)    = Pars.KappaU * gamma.Xclean(1,:).^2;
        gamma.Tan       = zeros(D,N);
        gamma.Tan(1,:)  = 1;
        gamma.Tan(2,:)  = Pars.KappaU * 2 * gamma.Xclean(1,:);

    case 'SkewParabola'
        D          = max(D,3);
        gamma.Xlatent   = sort(rand(1, N+1));
        dts             = gamma.Xlatent(2:N+1)-gamma.Xlatent(1:N);
        gamma.Xclean         = zeros(D,N+1);
        gamma.Xclean(1,:)    = gamma.Xlatent;
        gamma.Xclean(2,:)    = Pars.KappaU * gamma.Xclean(1,:).^2;
        gamma.Xclean(3,:)    = Pars.KappaU * gamma.Xclean(1,:).^3- 0.5 * Pars.KappaU * gamma.Xclean(1,:).^2;
        gamma.Tan       = zeros(D,N+1);
        gamma.Tan(1,:)  = 1;
        gamma.Tan(2,:)  = Pars.KappaU * 2*gamma.Xclean(1,:);
        gamma.Tan(3,:)  = Pars.KappaU * 3*gamma.Xclean(1,:).^2 - Pars.KappaU * gamma.Xclean(1,:);

    case 'Helix'
        D          = max(D,3);
        gamma.Xlatent   = sort(rand(1, N));
        gamma.Xclean         = zeros(D, N);
        gamma.Xclean(1,:)    = cos(20*pi*gamma.Xlatent); % log(1+gamma.Xlatent) .* cos(2*pi*gamma.Xlatent);
        gamma.Xclean(2,:)    = sin(20*pi*gamma.Xlatent); % log(1+gamma.Xlatent) .* sin(2*pi*gamma.Xlatent);
        gamma.Xclean(3,:)    = gamma.Xlatent; % log(1+gamma.Xlatent);
        gamma.Tan       = zeros(D, N);
        gamma.Tan(1,:)  = -20*pi*gamma.Xclean(2,:); % (1+gamma.Xlatent).^-1 .* cos(gamma.Xlatent) - 2*pi*gamma.Xclean(2,:);
        gamma.Tan(2,:)  = 20*pi*gamma.Xclean(1,:); % (1+gamma.Xlatent).^-1 .* sin(gamma.Xlatent) + 2*pi*gamma.Xclean(1,:);
        gamma.Tan(3,:)  = ones(1,N); % (1+gamma.Xlatent).^-1;

    case 'multiHelix'
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        KappaU          = Pars.KappaU;
        X(1,:)          = Xlatent;
        Tan(1,:)        = ones(1, N+1);
        X(2,:)          = log(Xlatent+3).*cos(3*pi*Xlatent);
        Tan(2,:)        = ((Xlatent+3).^(-1)).*cos(3*pi*Xlatent) - 3*pi*log(Xlatent+3).*sin(3*pi*Xlatent);
        X(3,:)          = log(Xlatent+3).*sin(3*pi*Xlatent);
        Tan(3,:)        = ((Xlatent+3).^(-1)).*sin(3*pi*Xlatent) + 3*pi*log(Xlatent+3).*cos(3*pi*Xlatent);
        parfor d = 4:D
            a           = KappaU*(randn(7,1)); a(6) = a(6)*3 ;a(7) = a(7)/3;
            %a           = Pars.KappaU*([ 0.1257, 0.5248,0.2198,0.2227, 0.6187,-0.2775, -0.6344]); %test choice1
            angfreq     = a(6)*((d+3)^0.5)*pi;
            dimfactor   = (d+3)^(0.5);
            h1          = a(1)+a(2)*Xlatent + a(3)/2*Xlatent.^2 + a(4)/6*Xlatent.^3;
            h2          = cos(a(5)+angfreq*Xlatent) + exp(a(7)*Xlatent/5);
            dh1         = a(2)+a(3)*Xlatent+ a(4)/2* Xlatent.^2;
            dh2         = -sin(a(5)+angfreq*Xlatent).*angfreq + exp(a(7)/5*Xlatent)*(a(7)/5);
            X(d,:)      = dimfactor * h1.* h2;
            Tan(d,:)    = dimfactor * dh1 .* h2 + dimfactor * h1 .* dh2 ;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    %standard Meyer and its variants
    case 'Meyer'
        delta           = 1/D;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = exp(-(xd-Xlatent).^2/(4*delta^2)) / ((2*pi)^0.25 * delta^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta^2);
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean    = X;
        gamma.Tan       = Tan;

    case 'Meyer-L2-CK'
        delta           = 1/(Pars.KappaU*D);
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = exp(-(xd-Xlatent).^2/(4*delta^2)) / ((2*pi)^0.25 * delta^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta^2);
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean    = X;
        gamma.Tan       = Tan;

    case 'Meyer-L1-C1'
        delta           = 1/D;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = exp(- (2*delta)^(-1) * (xd-Xlatent).^2 ./(2*delta + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta^0.25);
            Tan(d,:)    = (2*delta)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta) ./ (2*delta + abs(xd-Xlatent)).^2 ;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-L1-CK'
        delta           = 1/(Pars.KappaU*D);
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = exp(- (2*delta)^(-1) * (xd-Xlatent).^2 ./(2*delta + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta^0.25);
            Tan(d,:)    = (2*delta)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta) ./ (2*delta + abs(xd-Xlatent)).^2 ;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-C1'
        delta           = 1/D;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = cos(d+(Xlatent-xd)/delta) .* exp(-(xd-Xlatent).^2/(4*delta^2)) / ((2*pi)^0.25 * delta^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta^2) - X(d,:) .* tan(d+(Xlatent-xd)/delta)/delta;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-CK'
        delta           = 1/(Pars.KappaU*D);
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = cos(d+(Xlatent-xd)/delta) .* exp(-(xd-Xlatent).^2/(4*delta^2)) / ((2*pi)^0.25 * delta^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta^2) - X(d,:) .* tan(d+(Xlatent-xd)/delta)/delta;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-C1'
        delta           = 1/D;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = cos(1*d+(Xlatent-xd)/delta) .* exp(- (2*delta)^(-1) * (xd-Xlatent).^2 ./(2*delta + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta^0.25);
            Tan(d,:)    = (2*delta)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta) ./ (2*delta + abs(xd-Xlatent)).^2    -   X(d,:) .* tan(d+(Xlatent-xd)/delta)/delta;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-CK'
        delta           = 1/(Pars.KappaU*D);
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            X(d,:)      = cos(1*d+(Xlatent-xd)/delta) .* exp(- (2*delta)^(-1) * (xd-Xlatent).^2 ./(2*delta + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta^0.25);
            Tan(d,:)    = (2*delta)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta) ./ (2*delta + abs(xd-Xlatent)).^2    -   X(d,:) .* tan(d+(Xlatent-xd)/delta)/delta;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-C1-variant1'
        var_freq        = 10;
        var_amp1        = 0.3;
        var_amp2        = 0.3;
        var_d1          = 1;
        var_d2          = 1;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-C1-variant2'
        var_freq        = 10;
        var_amp1        = 0.35;
        var_amp2        = 0.4;
        var_d1          = 1;
        var_d2          = 2;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-C1-variant3'
        var_freq        = 10;
        var_amp1        = 0.4;
        var_amp2        = 0.5;
        var_d1          = 1;
        var_d2          = 3;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;


    case 'Meyer-Helix-L2-CK-variant1'
        var_freq        = 10;
        var_amp1        = 0.3;
        var_amp2        = 0.3;
        var_d1          = Pars.KappaU;
        var_d2          = Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-CK-variant2'
        var_freq        = 10;
        var_amp1        = 0.35;
        var_amp2        = 0.4;
        var_d1          = Pars.KappaU;
        var_d2          = 2*Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L2-CK-variant3'
        var_freq        = 10;
        var_amp1        = 0.4;
        var_amp2        = 0.5;
        var_d1          = Pars.KappaU;
        var_d2          = 3*Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2)  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-C1-variant1'
        var_freq        = 10;
        var_amp1        = 0.3;
        var_amp2        = 0.3;
        var_d1          = 1;
        var_d2          = 1;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-C1-variant2'
        var_freq        = 10;
        var_amp1        = 0.35;
        var_amp2        = 0.4;
        var_d1          = 1;
        var_d2          = 2;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(d+var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-C1-variant3'
        var_freq        = 10;
        var_amp1        = 0.4;
        var_amp2        = 0.5;
        var_d1          = 1;
        var_d2          = 3;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(2*d+var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    % case 'Meyer-Helix-L1-C1-variant4'
    %     var_freq        = 10;
    %     var_amp1        = 0.45;
    %     var_amp2        = 0.6;
    %     var_d1          = 1;
    %     var_d2          = 4;

    case 'Meyer-Helix-L1-CK-variant1'
        var_freq        = 10;
        var_amp1        = 0.3;
        var_amp2        = 0.3;
        var_d1          = Pars.KappaU;
        var_d2          = Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;


    case 'Meyer-Helix-L1-CK-variant2'
        var_freq        = 10;
        var_amp1        = 0.35;
        var_amp2        = 0.4;
        var_d1          = Pars.KappaU;
        var_d2          = 2*Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(d+var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-Helix-L1-CK-variant3'
        var_freq        = 10;
        var_amp1        = 0.4;
        var_amp2        = 0.5;
        var_d1          = 1*Pars.KappaU;
        var_d2          = 3*Pars.KappaU;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta1      = var_d1*(1+var_amp1*sin(var_freq*d))/D;
            delta2      = var_d2*(1+var_amp2*cos(2*d+var_freq*d))/D;
            X(d,:)      = cos(var_freq*d+(Xlatent-xd)/delta1) .* exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2  - X(d,:) .* tan(var_freq*d+(Xlatent-xd)/delta1) /delta1;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;


    %first order pseudo-space-filling curve, should have constant reach and length~2^D. For computational purpose, D should be small (e.g. less than 12)
    case 'SpaceFillingFst'
        m               = 64;
        N               = m*2^D-m/2-2;
        gamma.Xlatent   = linspace(0,1,N+1);
        dts             = gamma.Xlatent(2:N+1)-gamma.Xlatent(1:N);
        X               = zeros(D, N+2);
        X(1,1:2*m)      = [0.5+0.5*sin(linspace(-pi/2,pi/2,1.5*m)),ones(1,m/2)]; %myhelper(2^1 * linspace(0,1,m) - 2^(1-1));
        for currd = 2 : D
            X(1:currd-1, m*2^(currd-1)+1:m*2^currd -m/2) = flip(X(1:currd-1,1:m*2^(currd-1)-m/2),2);
            X(currd,1:m*2^currd) = [zeros(1,m*2^(currd-1)-1*m),0.5+0.5*sin(linspace(-pi/2,pi/2,1.5*m)),ones(1,m*2^(currd-1)-0.5*m)];
        end
        gamma.Tan       = (X(:,2:N+2)-X(:,1:N+1)) * N;
        gamma.Xclean         = X(:,1:N+1);

    case 'Meyer-L2-C1-variant1'
        var_freq        = 10;
        var_amp2        = 0.3;
        var_d2          = 1;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            xd          = (d-1)/(D-1);
            X(d,:)      = exp(-(xd-Xlatent).^2/(4*delta2^2)) / ((2*pi)^0.25 * delta2^0.5);
            Tan(d,:)    = X(d,:) .* (xd-Xlatent)/(2*delta2^2);
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean         = X;
        gamma.Tan       = Tan;

    case 'Meyer-L1-C1-variant1'
        var_freq        = 10;
        var_amp2        = 0.3;
        var_d2          = 1;
        Xlatent         = sort(rand(1, N+1));
        dts             = Xlatent(2:N+1)-Xlatent(1:N);
        X               = zeros(D, N+1);
        Tan             = zeros(D, N+1);
        parfor d = 1:D
            xd          = (d-1)/(D-1);
            delta2      = var_d2*(1+var_amp2*cos(var_freq*d))/D;
            X(d,:)      = exp(- (2*delta2)^(-1) * (xd-Xlatent).^2 ./(2*delta2 + abs(xd-Xlatent))) / ((2*pi)^0.25 * delta2^0.25);
            Tan(d,:)    = (2*delta2)^(-1) * X(d,:) .* (xd-Xlatent) .* (abs(xd-Xlatent)+4*delta2) ./ (2*delta2 + abs(xd-Xlatent)).^2 ;
        end
        gamma.Xlatent   = Xlatent;
        gamma.Xclean    = X;
        gamma.Tan       = Tan;

    otherwise
        warning('createCurve: curve of Name=%s not known.',Name)
end


%% curve data: Singular Value Decomposition
gamma.Xclean        = gamma.Xclean - mean(gamma.Xclean,2);
gamma.Diameter      = max(vecnorm(gamma.Xclean))*2;
[U,S,~]             = svd(gamma.Xclean,'econ'); %use econ when N>D
Xclean_SVD          = U' * gamma.Xclean ;
Tan_SVD             = U' * gamma.Tan;
gamma.SV            = diag(S/N^0.5);


%% compute reach
% warning: for Info_DimReduction other than 'None', the computation cost is extremely big.
if strcmp(Pars.Info_DimReduction,'None') || strcmp(Pars.Info_DimReduction,'none')
    flag_DimReduction_PCA = false;
    flag_DimReduction_RND = false;
elseif strcmp(Pars.Info_DimReduction,'PCA')
    flag_DimReduction_PCA = true;
    flag_DimReduction_RND = false;
elseif strcmp(Pars.Info_DimReduction,'RND')
    flag_DimReduction_PCA = false;
    flag_DimReduction_RND = true;
else
    flag_DimReduction_PCA = true;
    flag_DimReduction_RND = true;
end

[loc_reach_arr, appendix] = compute_reach(Xclean_SVD, Tan_SVD,dts, min(N,D^2*1e2),struct('flag_DimReduction_PCA',flag_DimReduction_PCA,'flag_DimReduction_RND',flag_DimReduction_RND));
gamma.Reach               = min(loc_reach_arr);
gamma.loc_reach_arr       = loc_reach_arr;


if strcmp(Pars.Info_DimReduction,'None') || strcmp(Pars.Info_DimReduction,'none')
    %do nothing
elseif strcmp(Pars.Info_DimReduction,'PCA')
    gamma.PCAproj_gamma_Len_arr_d   = appendix.PCAproj_gamma_Len_arr_d;
    gamma.PCAproj_gamma_Reach_arr_d = appendix.PCAproj_gamma_Reach_arr_d;
elseif strcmp(Pars.Info_DimReduction,'RND')
    gamma.RNDproj_gamma_Len_arr_d   = appendix.RNDproj_gamma_Len_arr_d;
    gamma.RNDproj_gamma_Reach_arr_d = appendix.RNDproj_gamma_Reach_arr_d;
else
    gamma.PCAproj_gamma_Len_arr_d   = appendix.PCAproj_gamma_Len_arr_d;
    gamma.PCAproj_gamma_Reach_arr_d = appendix.PCAproj_gamma_Reach_arr_d;
    gamma.RNDproj_gamma_Len_arr_d   = appendix.RNDproj_gamma_Len_arr_d;
    gamma.RNDproj_gamma_Reach_arr_d = appendix.RNDproj_gamma_Reach_arr_d;
end

%% curve data: calculate length, arclength, curvature scalar&vector, arclength, and normalize Len and Kapvec to unit vector
gamma.arclength = vecnorm(gamma.Tan(:, 1:N)) .* dts;
gamma.Tan       = normcols(gamma.Tan);
gamma.Len       = sum(gamma.arclength);
gamma.Kapvec    = (gamma.Tan(:,2:N+1)-gamma.Tan(:,1:N)) ./ gamma.arclength;
gamma.Kapval    = vecnorm(gamma.Kapvec);
gamma.Kapvec    = normcols(gamma.Kapvec);
gamma.Kap       = [min(gamma.Kapval), mean(gamma.Kapval), max(gamma.Kapval)];
gamma.Tan       = gamma.Tan(:, 1:N);
gamma.Xclean    = gamma.Xclean(:, 1:N);
gamma.Xlatent   = gamma.Xlatent(1:N);

%% rescale X so that (Diameter=1 or Reach=1*d^0.5)
switch Pars.rescale
    case 'None'
        scalefact       = 1;
    case 'Diameter'
        scalefact       = 1/gamma.Diameter ;
    case 'Reach'
        scalefact       = D^0.5 / gamma.Reach;
end

gamma.Xclean        = gamma.Xclean * scalefact;
gamma.Kapval        = gamma.Kapval / scalefact;
gamma.Kap           = gamma.Kap / scalefact;
gamma.Len           = gamma.Len * scalefact;
gamma.arclength     = gamma.arclength * scalefact;
gamma.Diameter      = gamma.Diameter * scalefact;
gamma.Reach         = gamma.Reach * scalefact;
gamma.loc_reach_arr = gamma.loc_reach_arr * scalefact;
if strcmp(Pars.Info_DimReduction,'None') || strcmp(Pars.Info_DimReduction,'none')
elseif strcmp(Pars.Info_DimReduction,'PCA')
    gamma.PCAproj_gamma_Len_arr_d   = gamma.PCAproj_gamma_Len_arr_d * scalefact;
    gamma.PCAproj_gamma_Reach_arr_d = gamma.PCAproj_gamma_Reach_arr_d * scalefact;
elseif strcmp(Pars.Info_DimReduction,'RND')
    gamma.RNDproj_gamma_Len_arr_d   = gamma.RNDproj_gamma_Len_arr_d * scalefact;
    gamma.RNDproj_gamma_Reach_arr_d = gamma.RNDproj_gamma_Reach_arr_d * scalefact;
else
    gamma.PCAproj_gamma_Len_arr_d   = gamma.PCAproj_gamma_Len_arr_d * scalefact;
    gamma.PCAproj_gamma_Reach_arr_d = gamma.PCAproj_gamma_Reach_arr_d * scalefact;
    gamma.RNDproj_gamma_Len_arr_d   = gamma.RNDproj_gamma_Len_arr_d * scalefact;
    gamma.RNDproj_gamma_Reach_arr_d = gamma.RNDproj_gamma_Reach_arr_d * scalefact;
end
%% curve data: add deviation in hyperplane normal to the tangent vector
Xclean              = gamma.Xclean ;
gamma.Xclean        = Xclean;
Tan                 = gamma.Tan;
Reach               = gamma.Reach;
loc_reach_arr       = gamma.loc_reach_arr;

if (~isfield(Pars,'normalSigma')) || (size(Pars.normalSigma,2)==1 && Pars.normalSigma==0)
    Pars.type_noise_off_curve  = 'none';
elseif ~isfield(Pars,'type_noise_off_curve') && size(Pars.normalSigma,2)==1 && Pars.normalSigma>0 && Pars.normalSigma<1
    Pars.type_noise_off_curve  = 'uniform & relative';
elseif ~isfield(Pars,'type_noise_off_curve') && size(Pars.normalSigma,2)==1 && Pars.normalSigma>=1
    Pars.type_noise_off_curve  = 'uniform_constant';
elseif ~isfield(Pars,'type_noise_off_curve') && size(Pars.normalSigma,2)==N
    Pars.type_noise_off_curve  = 'pointwise';
end


%for scalar value, standard deviation off the curve is normalSigma
if strcmp(Pars.type_noise_off_curve ,'uniform_constant')
    %with Z~N(0,Id) and |Z| truncated at 0.9*Reach/Pars.normalSigma, the distribution of |Z|^2
    pd              = truncate(makedist('Gamma','a',D/2,'b',2), 0, 0.81 * Reach^2 / Pars.normalSigma^2);
    %random array (truncated at 0.9 * Reach) of   Pars.normalSigma * |Z|
    rndrad          = Pars.normalSigma * (random(pd,1,N)).^0.5;
    noise           = bsxfun(@times, normcols(randn(D,N)), rndrad);
    gamma.X_UniCstDevSig  = Xclean + noise - bsxfun(@times, Tan, dot(noise,Tan));

% for array, standard deviation off the curve is each entry normalSigma times loc_reach_arr
elseif strcmp(Pars.type_noise_off_curve ,'relative_constant')
    %with Z~N(0,Id) and |Z| truncated at 0.9*Reach/normalSigma, the distribution of |Z|^2
    pd              = truncate(makedist('Gamma','a',D/2,'b',2), 0, 0.81 * Reach^2 / Pars.normalSigma^2);
    %random array (truncated at 0.9 * loc_reach) of  normalSigma *loc_reach/Reach * |Z|
    rndrad          = Pars.normalSigma * loc_reach_arr / Reach .* (random(pd,1,N)).^0.5;
    noise           = bsxfun(@times, normcols(randn(D,N)), rndrad);
    gamma.X_RelCstDevSig = Xclean + noise - bsxfun(@times, Tan, dot(noise,Tan));

elseif strcmp(Pars.type_noise_off_curve ,'uniform & relative')
    %with Z~N(0,Id) and |Z| truncated at 0.9*Reach/normalSigma, the distribution of |Z|^2
    pd              = truncate(makedist('Gamma','a',D/2,'b',2), 0, 0.81 * Reach^2 / Pars.normalSigma^2);
    %random array (truncated at 0.9 * Reach) of  normalSigma * |Z|
    rndrad_uni      = Pars.normalSigma * (random(pd,1,N)).^0.5;
    noise_uni       = bsxfun(@times, normcols(randn(D,N)), rndrad_uni);
    gamma.X_UniCstDevSig = Xclean + noise_uni - bsxfun(@times, Tan, dot(noise_uni,Tan));
    %random array (truncated at 0.9 * loc_reach) of  normalSigma *loc_reach/Reach * |Z|
    rndrad_rel      = Pars.normalSigma * loc_reach_arr / Reach .* (random(pd,1,N)).^0.5;
    noise_rel       = bsxfun(@times, normcols(randn(D,N)), rndrad_rel);
    gamma.X_RelCstDevSig = Xclean + noise_rel - bsxfun(@times, Tan, dot(noise_rel,Tan));

elseif strcmp(Pars.type_noise_off_curve ,'pointwise')
    %with Z~N(0,Id) and |Z| truncated at 0.9*Reach/normalSigma, the distribution of |Z|^2
    pd              = truncate(makedist('Gamma','a',D/2,'b',2), 0, 0.81 * Reach^2 / Pars.normalSigma^2);
    %random array (truncated at 0.9 * loc_reach) of  normalSigma * loc_reach/Reach * |Z|
    rndrad_pts      = loc_reach_arr / Reach .* Pars.normalSigma .*  (random(pd,1,N)).^0.5;
    noise_pts       = bsxfun(@times, normcols(randn(D,N)), rndrad_pts);
    gamma.X_PtsDevSig = Xclean + noise_pts - bsxfun(@times, Tan, dot(noise_pts,Tan));

end



gamma.normalSigma           = Pars.normalSigma;
gamma.type_noise_off_curve  = Pars.type_noise_off_curve;

delete(gcp('nocreate'))

return
