%% Table_postmoments.m
%
% version: 1
% created: 11 December 2013
% authors: J.R. Magnus, G. De Luca
%
% Description: This Matlab code create the 'WeibullMom.txt' and 
% 'SubbotinMom.txt' files containing the means and the variance 
% of the posterior distribution under the Weibull and Subbotin 
% priors for given x-values in the [0 100] interval with step 0.01.
%% -------------------------------------------------------------
clc;
clear all;
format long;
Options=optimset('TolX',1e-6,'TolFun',1e-6,'Display','iter');

% Weibull
    q=0.887630085544086;
    alpha=1-q;
    c=0.693148182611436;
    diary off;
    filename = ['WeibullMom.txt'];
    if (exist(filename,'file')==2)
        delete(filename);
    end
    diary(filename);
    x = 0:.01:100;
    k=length(x);
    output= zeros(k,3);      
    fprintf('x        m             v        \n');        
    for h=1:k
        xh=x(h);
        [mh,vh]=postmoments(xh,q,alpha,c);
        fprintf('%6.2f   %11.9f   %11.9f \n', xh, mh, vh);        
    end
    diary off;

    
% Subbotin
    q    =0.799512530172489;
    alpha=0;
    c    =0.937673273794677;
    diary off;
    filename = ['SubbotinMom.txt'];
    if (exist(filename,'file')==2)
        delete(filename);
    end
    diary(filename);
    x = 0:.01:100;
    k=length(x);
    output= zeros(k,3);      
    fprintf('x        m             v        \n');        
    for h=1:k
        xh=x(h);
        [mh,vh]=postmoments(xh,q,alpha,c);
        fprintf('%4.2f   %11.9f   %11.9f \n', xh, mh, vh);        
    end
    diary off;
    