%Script file: trainWeight.m

%train weight
NUM_REGION = 64;
NUM_CENTERS = 64;
meanIntra = zeros(NUM_REGION,1);
meanExtra = meanIntra;
varIntra = meanIntra;
varExtra = meanIntra;
numIntra = 0;
numExtra = 0;

for ii = 1:NUM
    for jj = ii+1:NUM
        dist = (histLBP(ii,:) - histLBP(jj,:)).^2 ./ (histLBP(ii,:) + (histLBP(jj,:)));
        dist(logical(isnan(dist))) = 0;
        if ( id(ii) == id(jj)) 
            %intra
            for kk = 1:NUM_REGION
                meanIntra(kk) = meanIntra(kk) + sum(dist((kk-1)*NUM_CENTERS+1:kk*NUM_CENTERS));
            end
            numIntra = numIntra + 1;
        else
            %extra
            for kk = 1:NUM_REGION
                meanExtra(kk) = meanExtra(kk) + sum(dist((kk-1)*NUM_CENTERS+1:kk*NUM_CENTERS));
            end
            numExtra = numExtra + 1;
        end
    end
end
meanIntra = meanIntra / numIntra;
meanExtra = meanExtra / numExtra;

%variance
for ii = 1:NUM
    for jj = ii+1:NUM
        dist = (histLBP(ii,:) - histLBP(jj,:)).^2 ./ (histLBP(ii,:) + (histLBP(jj,:)));
        dist(logical(isnan(dist))) = 0;
        if ( id(ii) == id(jj)) 
            %intra
            for kk = 1:NUM_REGION
                varIntra(kk) = varIntra(kk) + (sum(dist((kk-1)*NUM_CENTERS+1:kk*NUM_CENTERS)) - meanIntra(kk))^2;
            end
            %numIntra = numIntra + 1;
        else
            %extra
            for kk = 1:NUM_REGION
                varExtra(kk) = varExtra(kk) + (sum(dist((kk-1)*NUM_CENTERS+1:kk*NUM_CENTERS)) - meanExtra(kk))^2;
            end
            %numExtra = numExtra + 1;
        end
    end
end

varIntra = varIntra / numIntra;
varExtra = varExtra / numExtra;

weight = ( meanIntra - meanExtra).^2 ./ (varIntra + varExtra);
