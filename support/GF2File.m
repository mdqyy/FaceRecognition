%write gabor filter coefficients to file

ffile = fopen('Gabor.bin', 'wb');
%write num of filters
fwrite(ffile, 20, 'int');
%write kernel size
fwrite(ffile, GaborH, 'int');

for i = 1:2:size(GaborImg,3)
    for j = 1:GaborH
        for k = 1:GaborW
            fwrite(ffile, GaborReal(j,k,i), 'double');
        end
    end
end

for i = 1:2:size(GaborImg,3)
    for j = 1:GaborH
        for k = 1:GaborW
            fwrite(ffile, GaborImg(j,k,i), 'double');
        end
    end
end


fclose(ffile);