%% Script to run all tests for this suite of functions
% The |runtests| command seems to switch folders into the subfolder to run
% the tests-- we must make sure that the functions we are testing are still
% on the MATLAB path when we do this:
addpath(pwd)

results = runtests(pwd, 'Recursively', true)

% To run an individual test, use a syntax like the following:
% addpath('UnitTesting')
% results1 = runtests('test_convertGoogleToYahooTickers.m')
% rmpath('UnitTesting')

rmpath(pwd)