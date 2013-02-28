%script: batchRename.m
%To rename and organize training and testing images

trainRoot = 'C:\Users\Zhang\Desktop\FaceRecognition\image\train\';
matchRoot = 'C:\Users\Zhang\Desktop\FaceRecognition\image\match\';

curNumTags = 60;

for ii = 1:curNumTags
    idxDir = sprintf('%d',ii);
    trainList = dir([trainRoot idxDir '\*.jpg']);
    matchList = dir([matchRoot idxDir '\*.jpg']);
    
    for jj = 1:size(trainList,1)
        trainRename = sprintf('%d_%d.jpg',ii,jj);
        movefile([trainRoot idxDir '\' trainList(jj).name], [trainRoot idxDir '\' trainRename]);
    end
    
    for jj = 1:size(matchList,1)
        matchRename = sprintf('%d_%d.jpg',ii,jj);
        movefile([matchRoot idxDir '\' matchList(jj).name], [matchRoot idxDir '\' matchRename]);
    end
end