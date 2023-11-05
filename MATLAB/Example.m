%% Example.m
%
% version: 1.0
% created: 18 December 2013
% authors: J.R. Magnus, G. De Luca
%
% Description: This Matlab code calls the 'wals.m' code which perform 
% weighted average least squares (WALS) estimation of linear regression 
% models with sperical errors. 
%
% Your output is written to the folder: "results".
% Our results, which should be the same, are given in the folder: "our-output".
%
% Comments and suggestions are welcome and should be addressed to 
% 'magnus@uvt.nl' and 'giuseppe.deluca@unipa.it' 
%% -------------------------------------------------------------


%% --- Example 1
clc;
clear all;
%{
diary off;
filename = ['your-output\Example1.txt'];
if (exist(filename,'file')==2)
    delete(filename);
end
diary(filename);
%}
%% --- Load data
[data,txt,raw]  = xlsread(['\data\M1.xls']);

%% --- Set your dependent variable and regressors
k       =size(data,2)-1;
k1      = 6; 
k2      = k-k1; 
y       = data(:,1);
X1      = data(:,2:k1+1);
X2      = data(:,k1+2:k+1);
namelist= txt(1,3:k+2)';

%% --- WALS estimation
%{
    % Default
    [b, se, V, exitflag] = wals(y,X1,X2);

     % Parsing variable names 
	[b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist);

    % Subbotin prior
    [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist, 'prior','Subbotin');

    % Laplace prior
    [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist, 'prior','Laplace');

    % Posterior moments from weibullmom.xls
    [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist, 'postmoments','weibullmom.xls');

    % Posterior moments from subbotinmom.xls
    [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist, 'postmoments','subbotinmom.xls');


    % Sigma estimated from the restricted model 
     [b,bint,r,rint,stats]=regress(y,X1);
     sigma_hat=sqrt(stats(4))
     [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist,'sigma',sigma_hat);
     

    % Option notable & post-estimation results 
     [b, se, V, exitflag] = wals(y,X1,X2,'varnames',namelist, 'postmoments','subbotinmom.xls', 'results','no-table');
     b
	 se
	 V

diary off;
%}

wals(y,X1(:,1),[X1(:,2:end) X2])
