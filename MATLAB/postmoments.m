%% postmoments.m
%
% version: 1
% created: 11 December 2013
% authors: J.R. Magnus, G. De Luca
%
% Description: This Matlab code generate the mean and the variance 
% of the posterior distribution at a given x-value under a 
% reflected generalized gamma prior. 
%% -------------------------------------------------------------
function [m,v] = postmoments(x,q,alpha,c)    
    L=(1-alpha)./ q;
    A=(q .* c.^L) ./ (2 .* exp(gammaln(L)));
    prior= @(gamma) A .* abs(gamma).^(-alpha) .* (exp(-c.*(abs(gamma).^q)));
    A0=@(p1) arrayfun(@(eta)   (                normpdf(x+eta) +                 normpdf(x-eta)) .* prior(eta),p1);
    A1=@(p1) arrayfun(@(eta)   ( (x+eta)     .* normpdf(x+eta) +  (x-eta)     .* normpdf(x-eta)) .* prior(eta),p1);
    A2=@(p1) arrayfun(@(eta)   (((x+eta).^2) .* normpdf(x+eta) + ((x-eta).^2) .* normpdf(x-eta)) .* prior(eta),p1);    
    int_A0 = quadgk(A0,0,inf,'RelTol',1e-12,'AbsTol',1e-25);
    int_A1 = quadgk(A1,0,inf,'RelTol',1e-12,'AbsTol',1e-25);    
    int_A2 = quadgk(A2,0,inf,'RelTol',1e-12,'AbsTol',1e-25);        
    m=x - int_A1 ./ int_A0;    
    v=(int_A2 ./ int_A0) - ((int_A1 ./ int_A0).^2);
end

