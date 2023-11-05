%% wals.m
% 
% version: 2.0
% created: 3 april 2010
% latest revision: 18 december 2013
% authors: J.R. Magnus, G. De Luca
%
% Description: Calculates WALS estimates and precisions
% when some regressors (X1) are always in the model, while model selection
% takes place over the remainder (X2) of the regressors.
%
% The WALS procedure was originally introduced in:
% Magnus, J.R., O. Powell, and P. Prufer (2010),
%    ``A comparison of two model averaging techniques with an application 
%    to growth empirics,'' Journal of Econometrics, 154, 139-153.
%
% An extension to the priors used in the WALS procedure was introduced in:
% Kumar, K. and J.R. Magnus (2013),
%    ``A characterization of Bayesian robustness for a
%     normal location parameter'', Sankhya (Series B) 75, 216–237.
%
%  Magnus, J.R. and G. De Luca (2014),
%    ``Weighted-average least squares (WALS): A survey'',
%    in progress.
%
% An extension to the weighting scheme was introduced in:
% De Luca, G. and J.R. Magnus (2011),
%    ``Bayesian model averaging and weighted average least squares:
%    equivariance, stability, and numerical issues,'' 
%    The Stata Journal, 11, 518--544.
%
% This code may be used freely, but reference should be made to:
% Magnus, Powell, and Prufer (2010); Kumar and Magnus (2011); 
% Magnus and De Luca (2014) and - if appropriate - to 
% De Luca and Magnus (2011). 
% 
% The authors accept no responsibility for any errors or malfunctions 
% in the program. Tested with Matlab 7.8.0 (R2009a).
%
% Comments and suggestions are welcome and should be addressed to 
% 'magnus@uvt.nl' and 'giuseppe.deluca@unipa.it' 
%
% Version 1.4 differs from previous versions  
%('est_wals.m' and 'ets_subbotin.m') in several respects. 
% The main differences are: 
% arguments required as input, 
% priors and other optional arguments that can be passed to the code,  
% default values for optional arguments, 
% addtional checks on the arguments passed as input,
% and display of the estimation results. 
%
%% Required inputs:
%       y       =   (n x 1) vector containing the observations on the dependent 
%                   variable "y". All observations must be real numbers.
%                   Missing values are not allowed. 
%       X1      =   (n x k1) matrix of observations on the focus regressors
%                   "X1". All observations must be real numbers.
%                   Missing values are not allowed.
%       X2      =   (n x k2) matrix of observations on the auxiliary regressors 
%                   "X2". All observations must be real numbers.
%                   Missing values are not allowed.
%
%     Note:    1)   (X1:X2) together should contain ALL the regressors,
%                   including the constant term (if present).
%              2)   'n' must be greater than k=k1+k2. 
%              3)   'k1' and 'k2' must be greater than 1
%% Optional inputs:
%
%       'prior' =    String containing the prior to be used in WALS estimation.
%                    The available 'prior' options are: 
%                    'Weibull' (the default), 'Subbotin', 'Laplace'. 
%                    Parameters of these prior distributions are always fixed
%                    to their minimax regret solutions under the neutrality 
%                    condition. 
%
%     'varname' =    (k x 1) string vector containing the names of the focus and 
%                    auxiliary variables, respecively. By default, focus
%                    variables are named by {X1_j, j=1...k1) and auxiliary
%                    variables are named by {X2_h, h=1...k2).
%
% 'postmoments' =    'weibullmom.xls' or 'subbotinmom.xls'. 
%                    This option defines the exel datafiles containing posterior means
%                    and variances under the weibull and subbotin priors, 
%                    respectively. 
%                    By default, the means m(x) and the variances v(x) of 
%                    the posterior distributions based on the Weibull and 
%                    Subbotin priors are computed by numerical integration 
%                    using the 'quadgk' function. 
%                    The 'postmoments' option avoids numerical integration 
%                    during the estimation process by interpolating m(x)
%                    and v(x) between the moments of the nearest
%                    x-values in the interval [0,100] with step 0.01. 
%                    For example, for x = 2.3571, we know from the table 
%                    m(2.35) and m(2.36) and thus approximate
%                    m(2.3571) = (29/100) x m(2.35) + (71/100) x m(2.36).
%                    The same for the posterior variances. For x>100, 
%                    posterior moments are approximated by m(100) 
%                    and v(100). 
%                    For the Laplace prior, this option is not active
%                    because moments of the resulting posterior distribution 
%                    can be computed accurately. 
%                    Notice that the 'postmoments' option defines implicitly the prior
%                    used in WALS estimation and thus it overwrites the prior specified 
%                    in the option 'prior'
%
%       'sigma' =    a non-negative real scalar for the standard deviation of the  
%                    disturbances. By default, the standard deviation is
%                    estimated by OLS in the unrestricted model. 
%
%     'results' =    determines whether or not the table of the estimation 
%                    results must be displayed. The available 'results' 
%                    options are: 
%                    'table' (the default) or 'no-table'.  
%
%% Outputs:
%
%     b         = vector of coefficient estimates.
%
%     se        = standard errors associated with b.
%
%     V         = variance matrix associated with b. 
%
%     exitflag  = some error occurred:
%               = 1    y, X1 and X2 have different number of observations
%               = 2    n<=k
%               = 3    the dimension of 'varnames' is different from k 
%               = 4    the diagonal matrix of eigenvalues is not positive definite.
%
%% Examples: see the file 'Example.m'
%
%% -----------------------------------------------------------
function [b] = wals(y, X1, X2, varargin)

% Parse syntax
p = inputParser;
addRequired(p,'y' ,@(x) validateattributes(x,{'numeric'},{'column','real','nonnan','finite','nonempty'}));
addRequired(p,'X1',@(x) validateattributes(x,{'numeric'},{'2d'    ,'real','nonnan','finite','nonempty'}));
addRequired(p,'X2',@(x) validateattributes(x,{'numeric'},{'2d'    ,'real','nonnan','finite','nonempty'}));
addOptional(p,'prior', 'weibull', @(x)any(strcmpi(x,{'weibull','subbotin','laplace'})));
addOptional(p,'varnames', '', @(x) validateattributes(x, {'cell', 'char'},{'column'}));
addOptional(p,'postmoments', '', @(x)any(strcmpi(x,{'weibullmom.xls','subbotinmom.xls'})));
addOptional(p,'sigma', 0, @(x) validateattributes(x,{'numeric'},{'scalar','nonnegative','real','nonnan','finite','nonempty'}));
addOptional(p,'results', 'table', @(x)any(strcmpi(x,{'table','no-table'})));
parse(p,y, X1, X2, varargin{:});

%% --- Check number of observations 
exitflag =0;
n          = size(y,1);
n1         = size(X1,1);
n2         = size(X2,1);
if (n ~= n1) || (n ~= n2) || (n1 ~= n2) 
   exitflag = 1; 
end

b1 = 1;
b2 = 1;
varb1 = 1;
varb2 = 1;
covb1b2 = 1;
s = 1;

%% --- Define number of focus and auxiliary regressors
k1         = size(X1,2);
k2         = size(X2,2);
k = k1 + k2;

%% --- Check on number of obs and number of regressors 
if (exitflag==0) && (n <= k)    
    exitflag = 2; 
end

%% --- Variable names
% Default
if (strcmpi('',p.Results.varnames)==1)
    varnames = 'X1_1';
    for j=2:k1
        temp = strcat('X1_',int2str(j));
        varnames = strvcat(varnames,temp);
    end
    for h=1:k2
        temp = strcat('X2_',int2str(h));
        varnames = strvcat(varnames,temp);
    end
    varnames=cellstr(varnames);
% By the user
else
    varnames=p.Results.varnames;
    dim=length(varnames);
    if (dim~=k)
    	exitflag = 3; 
    end
end

%% --- Table of posterior moments
if strcmpi('',p.Results.postmoments)==1 
   pmomtab=0;
else
   pmomtab=1; 
end    

%% --- Sigma known
sigknown=p.Results.sigma;

%% --- Output option
if strcmpi('no-table',p.Results.results)==1 
   table=0;
else
   table=1; 
end    

%% --- If checks on previous conditions are satisfied, then proceed
if (exitflag == 0)

    %% --- Define prior parameters and names 
    if pmomtab==0
        prior = strcmpi('weibull',p.Results.prior)+2.*strcmpi('subbotin',p.Results.prior) + 3.* strcmpi('laplace',p.Results.prior);
        prior_name=p.Results.prior;
    else
        if strcmpi('weibullmom.xls',p.Results.postmoments)==1
           prior=1;
           prior_name='weibull';
        end
        if strcmpi('subbotinmom.xls',p.Results.postmoments)==1
           prior=2;
           prior_name='subbotin';
        end    
    end
    if (prior==1) 
        q=0.887630085544086;
        alpha=1-q;
        c=log(2);
    end
    if (prior==2) 
        q=0.799512530172489;
        alpha=0;
        c=0.937673273794677;
    end
    if (prior==3) 
        q=1;
        alpha=0;
        c=log(2);
    end
    
    %% --- Step 2.a: Scaling X1 so that all diagonal elements of 
    %%     (X1*Delta1)'X1*Delta1 are all one
    d1 = diag(X1'*X1).^(-1/2);
    Delta1 = diag(d1);
    Z1 = X1 * Delta1;
    
    %% --- Step 2.b: Scaling X2 so that all diagonal elements of
    %%     (X2*Delta2)'M1*X2*Delta2 are all one
    Z2d     = X2'*X2;
    V1r     = inv(Z1'*Z1);
    VV12    = Z1'*X2;
    Z2d     = Z2d - VV12' * V1r * VV12;
    d2      = diag(Z2d).^(-1/2);
    Delta2  = diag(d2);
    Z2s     = Delta2 * Z2d * Delta2;

    %% --- Step 3: Semi-orthogonalization of Z2s
    [T, Xi]     = eig(Z2s);
    order       = max(size(Xi));
    eigv        = diag(Xi);
    tol         = eps; 
    rank        = sum(eigv>tol);
    if (rank ~= order) 
        exitflag = 4;
    end

    %% --- If Xi is positive definite, proceed
    if (exitflag ~= 4)

        %% --- Set up Z2 so that Z2'*M1*Z2 = I
        D2=Delta2 * T * diag(diag(Xi).^(-.5));
        Z2=X2 * D2;

        %% --- Step 4: OLS of unrestricted model
        Z                      = [Z1 Z2];
        [gamma_hat, ci, resid] = regress(y, Z);
        if (sigknown==0) 
                s2             = resid' * resid / (n-k);
                s              = s2^0.5;
        else
                s              =sigknown;
                s2             =s^2;
        end
        gamma2_hat             = gamma_hat(k1+1:k);
        x                      = gamma2_hat / s;

        %% --- Step 5: Compute the mean and variance of the posterior 
        m_post  = zeros(k2,1);    
        v_post  = zeros(k2,1);
        if pmomtab==0
            % Laplace prior
            if (prior==3)
                signx           = sign(x);
                absx            = abs(x);
                g0              = normcdf(-absx - c, 0, 1);
                g1              = normcdf( absx - c, 0, 1);
                g2              = normpdf( absx - c, 0, 1);
                psi0            = g0 ./ g1;
                psi1            = g2 ./ g1;
                psi2            = exp(2 * c * absx) .* psi0;
                hratio          = (1 - psi2) ./ (1 + psi2);
                m_post          = signx .* (absx - c * hratio);
                v_post          = 1 + c^2 .* (1-hratio.^2)  ...
                    - c .* (1+hratio) .* psi1;
            % Weibull and subbotin priors
            else
                m_post  = zeros(k2,1);    
                v_post  = zeros(k2,1);
                delta=(1-alpha)./ q;
                Prior= @(gamma) ((q .* c.^delta) ./ (2 .* exp(gammaln(delta)))) .* abs(gamma).^(-alpha) .* (exp(-c.*(abs(gamma).^q)));    
                for h=1:k2
                    xh=x(h);
                    A0=@(gamma) (                 normpdf(xh-gamma) +                  normpdf(xh+gamma)).*Prior(gamma);
                    A1=@(gamma) ( (xh-gamma)    .*normpdf(xh-gamma) +  (xh+gamma).*    normpdf(xh+gamma)).*Prior(gamma);
                    A2=@(gamma) (((xh-gamma).^2).*normpdf(xh-gamma) + ((xh+gamma).^2).*normpdf(xh+gamma)).*Prior(gamma);
                    int_A0 = quadgk(A0,0,inf);
                    int_A1 = quadgk(A1,0,inf);
                    int_A2 = quadgk(A2,0,inf);
                    psi1 = int_A1/int_A0;
                    psi2 = int_A2/int_A0;
                    m_post(h) = xh - psi1;                                    
                    v_post(h) = psi2 - psi1^2;
                end
            end
        else
            pmdata  = xlsread([p.Results.postmoments]);
            xtab=pmdata(:,1);           
            mtab=pmdata(:,2);
            vtab=pmdata(:,3);
            for h=1:k2
                signxh=sign(x(h));
                xh=abs(x(h));
                if xh<100 
                    xhl1=floor(xh*100)/100;
                    whl1=1-(floor(xh*10000)/10000-xhl1).*100;
                    xhl1_ind=find(xtab==xhl1);
                    m_post(h) =signxh .* (whl1 .* mtab(xhl1_ind) + (1-whl1) .* mtab(xhl1_ind+1));
                    v_post(h) =whl1 .* vtab(xhl1_ind) + (1-whl1) .* vtab(xhl1_ind+1);
                else
                    m_post(h)=signxh .* mtab(10001);
                    v_post(h)=vtab(10001);
                end
            end
        end
        
        %% --- Step 6: WALS estimates 
        c2          = s * m_post;
        c1          = V1r * Z1' * (y - Z2*c2);
        b1          = Delta1 * c1;
        %disp('b1')
        %disp(b1)
        b2          = D2 * c2;

        %% --- Step 7: WALS precisions
        varc2       = s2 * diag(v_post);
        varb2       = D2 * varc2 * D2';
        Q           = V1r * Z1' * Z2;
        varc1       = s2 * V1r + Q * varc2 * Q';
        varb1       = Delta1 * varc1 * Delta1';
        covc1c2     = -Q * varc2;
        covb1b2     = Delta1 * covc1c2 * D2';
    end     % exitflag ~=4

    b  = [b1; b2];
    se = [diag(varb1).^0.5; diag(varb2).^0.5];
    V  = vertcat(horzcat(varb1,covb1b2),horzcat(covb1b2',varb2));

    %% --- Display output
    if table==1 
        %%% --- Create output headers for rows and columns
        estimates = [b se];
        headers_cols    = {'variable' 'b         ' 'se        '};

        %%% --- Determine length of longest regressor name
        maxl = length(headers_cols{1});
        for ind=1:k
            curl = length(varnames{ind});
            if (curl>maxl)
                maxl = curl;
            end
        end

        %%% --- Add extra spacing to header for 1st column
        curl = length(headers_cols{1});
        for ind=curl:maxl+3
            headers_cols{1} = [headers_cols{1} ' '];
        end

        %%% --- Add extra spacing for regressor names
        for ind=1:k
            curl = length(varnames{ind});
            for ind2=curl:maxl-1
                varnames{ind} = [varnames{ind} ' '];
            end
        end

        %%% --- Display table of estimation results
        %fprintf('\n\nWALS estimates - Prior %s \n', prior_name);
        %fprintf('n      = %d\n', n);
        %fprintf('k1     = %d\n', k1);
        %fprintf('k2     = %d\n', k2);
        %fprintf('q      = %4.4f\n', q);
        %fprintf('alpha  = %4.4f\n', alpha);
        %fprintf('c      = %4.4f\n', c);
        %fprintf('sigma  = %4.4f\n', s);

        %fprintf('\n %s%s%s\n', headers_cols{1}, headers_cols{2}, headers_cols{3});
        for ind=1:k
            %fprintf('%s   % 7.4f   % 7.4f\n', varnames{ind}, estimates(ind,1), estimates(ind,2));
        end
    end           % table==1
else              % exitflag ~=0
    b  = nan; 
    se = nan;
    V  = nan;
    %fprintf('\n\nSome error has occured: \n');
    if exitflag==1 
        %fprintf('y, X1 and X2 must have the same number of observations\n'); 
    end
    if exitflag==2 
        %fprintf('The number of observations must be greater than the number of regressors\n'); 
    end
    if exitflag==3
        %fprintf('varnames must contain %d elements\n', k); 
    end
    if exitflag==4
        %fprintf('The diagonal matrix of eigenvalues is not positive definite\n'); 
    end
end

