%Script file: swapTrainMatch.m

TRAIN_PATH = 'C:\Users\Zhi\Desktop\temp\train\';
MATCH_PATH = 'C:\Users\Zhi\Desktop\temp\match\';

for ii = 1: 270
    trainPath = sprintf('%s%d%s', TRAIN_PATH, ii, '\*.jpg');
    matchPath = sprintf('%s%d%s', MATCH_PATH, ii, '\*.jpg');
    trainList = dir(trainPath);
    matchList = dir(matchPath);
    %move match to train
    source = sprintf('%s%d%s', MATCH_PATH, ii, ['\',matchList(1).name]);
    dest = sprintf('%s%d%s',TRAIN_PATH, ii, '\');
    movefile(source, dest);
    
    %move one train to match 'fb'
    for jj = 1:length(trainList)
        type = trainList(jj).name(6:7);
        if ( strcmp(type, 'fb'))
            source = sprintf('%s%d%s', TRAIN_PATH, ii, ['\', trainList(jj).name]);
            dest = sprintf('%s%d%s', MATCH_PATH, ii, '\');
            movefile(source, dest);
            break;
        end
    end
end