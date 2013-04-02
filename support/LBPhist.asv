%script file: LBPhist.m
clear all
clc

%param
FEATURE_LEN = 5120;

fid = fopen('../image/faces.bin', 'rb');
fseek(fid, 0, 'eof');
fsize = ftell(fid);
fclose(fid);
fid = fopen('../image/faces.bin', 'rb');
NUM = fsize/(FEATURE_LEN+1)/4;  % 4byte - 32bit, id + 5120 features;
NUM_BIN = FEATURE_LEN;
id = zeros(NUM,1);
histLBP = zeros(NUM,NUM_BIN);



for ii = 1:NUM
    id(ii) = fread(fid, 1, 'int32');
    for jj = 1: NUM_BIN
        histLBP(ii, jj) = fread(fid, 1, 'float32=>uint8');
    end
end
    


fclose(fid);


% % calculate distance
% distTab = zeros(NUM,NUM);
% for i = 1:NUM
%     for j = i:NUM
%         denom = histLBP(i,:) + histLBP(j,:);
%         denom(denom == 0) = Inf;
%         distTab(i,j) = sum((histLBP(i,:) - histLBP(j,:)).^2 ./denom );
%     end
% end
% 
% distTab = distTab + distTab.';




        
                
        
        