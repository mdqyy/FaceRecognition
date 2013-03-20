%Script File: processImageTag.m

SOURCE_PATH = 'C:\Users\Zhi\Desktop\temp\match\';
DEST_PATH = 'C:\Users\Zhi\Desktop\temp\ImgTag\';

for ii = 1:270
    path = sprintf('%s%d%s',SOURCE_PATH, ii, '\*.jpg');
    imageList = dir(path);
    source = sprintf('%s%d%s', SOURCE_PATH, ii, ['\' imageList(1).name]);
    dest = sprintf('%s%d%s', DEST_PATH, ii, '.jpg');
    copyfile(source, dest);
end