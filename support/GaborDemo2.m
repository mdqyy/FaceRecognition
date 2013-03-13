% GaborReal, [GaborH,GaborW,40] 40?Gabor????
% GaborImg,  [GaborH,GaborW,40] 40?Gabor????
% ?????2???,???? Kmax=2.5*pi/2, f=sqrt(2), sigma=1.5*pi;
% GaborH, GaborW, Gabor????
% U,????{0,1,2,3,4,5,6,7}
% V,????{0,1,2,3,4}
% Kmax,f,sigma ????

GaborH=21;
GaborW=21;
Kmax=2.5*pi/2;
f=sqrt(2);
sigma=1.5*pi;
    
GaborReal = zeros( GaborH, GaborW, 40 );
GaborImg = zeros( GaborH, GaborW, 40 );

vnum=5;
unum=8;

% ??????

for v = 0 : (vnum-1)
    for u = 0 :(unum-1)
        [ GaborReal(:,:,v*8+u+1), GaborImg(:,:,v*8+u+1) ] = MakeGaborKernal( GaborH, GaborW, u, v, Kmax,f,sigma );
    end
end


G=cell(5,8);

for i = 1:5
    for j = 1:8
        G{i,j}=zeros(GaborH,GaborW);
    end
end
I=G;

for i = 1:5
    for j = 1:8
        G{i,j}=GaborReal(:,:,(i-1)*8+j);
        I{i,j}=GaborImg(:,:,(i-1)*8+j);
        %G{i,j}=G{i,j}.^2 + I{i,j}.^2;
    end
end

%plot
figure;
for i = 1:5
    for j = 1:8
        subplot(5,8,(i-1)*8+j);        
        %imshow(real(G{s,j})/2-0.5,[]);
        imshow(real(G{i,j}),[]);
    end
end
