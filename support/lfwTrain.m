%script file: lfwTrain.m

clear all
close all
clc

%
path = 'D:\Zhi\lfw';
file = 'D:\Zhi\lfw\pairsDevTrain.txt';
fid = fopen(file,'r');
numPairs = fscanf(fid,'%d');



fclose(fid);



