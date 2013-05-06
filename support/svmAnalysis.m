num = size(svmDebug,1);
mess = zeros(num,1);
for ii = 1: num
    mess(ii) = sum( svmDebug(ii,2:end-1) > svmDebug(ii, svmDebug(ii,1)+1));
end
    
for ii = 1:num
    [~, idx] = max(svmDebug(ii,2:end-1));
    if ( length(idx) == 1)
      if (svmDebug(ii,1) == idx)
          continue;
      end
    end
    b = bar(svmDebug(ii,2:end-1));
    axis([1 60 0 100]);
    ch = get(b, 'children');
    color = repmat([0 0 1], 60, 1);
    color(svmDebug(ii,1),1) = 1;
    color(svmDebug(ii,1),3) = 0;
    set(ch, 'FaceVertexCData', color);
    saveas(gcf, num2str(ii), 'jpg');
end