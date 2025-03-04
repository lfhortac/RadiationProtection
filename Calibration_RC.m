%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% S. Viñals/C. Coelho: Oct-2024                       %%%%%
%%%%%%%                                                     %%%%%
%%%%%%% Programm to calibrate radiochromic films. Read the  %%%%%
%%%%%%% manual "ManualRadiocrómicas.pdf" for details about  %%%%%
%%%%%%% the execution and the results obtained.             %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize MATLAB
close all;
clc;
clear all;

for i=1:255
   redmap(i,:) = [255-uint8(i) 0 0];
   greenmap(i,:) = [0 255-uint8(i) 0];
   bluemap(i,:) = [0 0 255-uint8(i)];
    
end

myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.tiff'));
for k = 1:length(myFiles)
  clf
  baseFileName = myFiles(k).name;
  FileName = fullfile(myDir,baseFileName);
  pic = imread(FileName);
  figure;
  imshow(pic);
cut=imcrop(pic);
clf
figure;
set(gcf,'color','w');
subplot(1,2,1); imshow(pic);
title('Original image')
subplot(1,2,2); imshow(cut);
title('Analysed image')
cut_R=cut(:,:,1);
cut_G=cut(:,:,2);
cut_B=cut(:,:,3);
figure;
set(gcf,'color','w');
subplot(1,3,1); heatmap(cut_R, 'colormap', redmap);
title('Optic Density - RED')
subplot(1,3,2); heatmap(cut_G, 'colormap', greenmap);
title('Optic Density - GREEN')
subplot(1,3,3); heatmap(cut_B, 'colormap', bluemap);
title('Optic Density - BLUE')
   
prompt = 'Dose delivered: ' ;
results(k,1)= input(prompt);
results(k,2)= mean2(cut_R); results(k,3)=std2(cut_R);
results(k,4)= mean2(cut_G); results(k,5)=std2(cut_G);
results(k,6)= mean2(cut_B); results(k,7)=std2(cut_B);

close all;

end

%save('MeanValues.txt','results','-ascii')

%%%
myfit = fittype('a+b/(x-c)','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'})
xData_R = results(:,2);
xData_G = results(:,4);
xData_B = results(:,6);
yData   = results(:,1);

figure;
set(gcf,'color','w');

%%%%%
red_fit  = fit(xData_R,yData,myfit)
coef_R = coeffvalues(red_fit); r2_R  = confint(red_fit);
subplot(2,3,1);plot(red_fit,xData_R,yData);
xlabel('Pixel value');
ylabel('Dose (Gy)');
title('Red');
subplot(2,3,4);plot(red_fit,xData_R,yData,'residuals');
xlabel('Densidad óptica');
ylabel('Residuals');

%%%%%
green_fit  = fit(xData_G,yData,myfit)
coef_G = coeffvalues(green_fit); r2_G  = confint(green_fit);
subplot(2,3,2);plot(green_fit,xData_G,yData);
xlabel('Pixel value');
ylabel('Dose (Gy)');
title('Green');
subplot(2,3,5);plot(green_fit,xData_G,yData,'residuals');
xlabel('Densidad óptica');
ylabel('Residuals');

%%%%%
blue_fit  = fit(xData_B,yData,myfit)
coef_B = coeffvalues(blue_fit); r2_B  = confint(blue_fit);
subplot(2,3,3);plot(blue_fit,xData_B,yData);
xlabel('Pixel value');
ylabel('Dose (Gy)');
title('Blue');
subplot(2,3,6);plot(blue_fit,xData_B,yData,'residuals');
xlabel('Densidad óptica');
ylabel('Residuals');


paramCoef(:,1) = coef_R; paramCoef(:,2) = coef_G; paramCoef(:,3) = coef_B;
save('CalibParameters.txt','paramCoef','-ascii');
