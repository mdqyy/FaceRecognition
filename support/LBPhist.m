%script file: LBPhist.m
clear all
clc

fid = fopen('../image/faces.bin', 'rb');
fseek(fid, 0, 'eof');
fsize = ftell(fid);
fclose(fid);
fid = fopen('../image/faces.bin', 'rb');
NUM = fsize/5121/4;  % 4byte - 32bit, id + 5120 features;
NUM_BIN = 5120;
id = zeros(NUM,1);
histLBP = zeros(NUM,NUM_BIN);



for ii = 1:NUM
    id(ii) = fread(fid, 1, 'int32');
    for jj = 1: NUM_BIN
        histLBP(ii, jj) = fread(fid, 1, 'float32=>uint8');
    end
end
    


fclose(fid);


% calculate distance
distTab = zeros(NUM,NUM);
for i = 1:NUM
    for j = i:NUM
        distTab(i,j) = sum(abs(histLBP(i,:) - histLBP(j,:)));
    end
end

distTab = distTab + distTab.';