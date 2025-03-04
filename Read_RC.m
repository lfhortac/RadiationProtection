%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% S. Viñals/C. Coelho: Oct-2024                       %%%%%
%%%%%%%                                                     %%%%%
%%%%%%% Programm to read and process radiochromic films.    %%%%%
%%%%%%% The programm reads the file with the calibration    %%%%%
%%%%%%% coefficients and analyze the picture to obtain      %%%%%
%%%%%%% the dose.                                           %%%%%
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
%%
%Open directory and load calibration parameters
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.tiff'));

caliFile = fopen('CalibParameters.txt'); %New calibration
sizePars = [3 3];
pars = fscanf(caliFile,'%f',sizePars);
pars = pars';
redCali = pars(:,1); greenCali = pars(:,2); blueCali = pars(:,3);
fclose(caliFile);


%%
%Analyze images
for k = 1:length(myFiles)
  clf
  baseFileName = myFiles(k).name;
  FileName = fullfile(myDir,baseFileName);
  pic = imread(FileName);
  figure;
  imshow(pic);
cut=imcrop(pic);
close all;
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
   

%%
%Calibrate image
dose(1) = redCali(1)+(redCali(2)/(mean2(cut_R)-redCali(3)));
dose(2) = greenCali(1)+(greenCali(2)/(mean2(cut_G)-greenCali(3)));
dose(3) = blueCali(1)+(blueCali(2)/(mean2(cut_B)-blueCali(3)));
std(1) = sqrt(((-(redCali(2)/((dose(1)-redCali(3))^2)))^2)*(std2(cut_R)^2));
std(2) = sqrt(((-(greenCali(2)/((dose(2)-greenCali(3))^2)))^2)*(std2(cut_G)^2));
std(3) = sqrt(((-(blueCali(2)/((dose(3)-blueCali(3))^2)))^2)*(std2(cut_B)^2));

dose_av = mean2(dose);
std_dose = std2(dose);

%%
%Correction factor for quenching (Sanchez-Parcerisa et al 2021)
s=2.2717;
q=-0.699;
p=1.75;
z=4; %Air gap that needs to be adjusted for each experiment
r=2.08;
En=10;
EW=((En-s*En^q)^p-z/r)^(1/p); %Taking into account the energy loss at the exit window
E=(En^p-z/r)^(1/p); %Without taking into account the energy loss at the exit window
a=4.1e5;
b=2.88;
c=22.5;
d=0.142;
LET=(a*exp(-b*E))+(c*exp(-d*E));
LETW=(a*exp(-b*EW))+(c*exp(-d*EW));
A=0.010;
B=1.09;
RE=1-A*LET^B;
REW=1-A*LETW^B;


%%
%Dose correction
dose_q(1) = dose(1)*RE;
dose_q(2) = dose(2)*RE;
dose_q(3) = dose(3)*RE;
Rstd_dose_q = std(1)*RE;
Gstd_dose_q = std(2)*RE;
Bstd_dose_q = std(3)*RE;
dose_av_q = dose_av*RE;
std_dose_q = std_dose*RE;
dose_W(1) = dose(1)*REW;
dose_W(2) = dose(2)*REW;
dose_W(3) = dose(3)*REW;
Rstd_dose_W = std(1)*REW;
Gstd_dose_W = std(2)*REW;
Bstd_dose_W = std(3)*REW;
dose_av_W = dose_av*REW;
std_dose_W = std_dose*REW;


%%
%Save data

structDose(k,1).picName = {baseFileName};
%structDose(k,1).AvDose = [dose_av]; %Whitout any correction
%structDose(k,1).CorrAvDose = [dose_av_q]; %Considering only the quenching effect
structDose(k,1).Corr2AvDose = [dose_av_W]; %Considering the quenching effect and the presence of the aluminium foil
%structDose(k,1).stdDose = [std_dose]; %Whitout any correction
%structDose(k,1).CorrstdDose = [std_dose_q]; %Considering only the quenching effect
structDose(k,1).Corr2stdDose = std_dose_W; %Considering the quenching effect and the presence of the aluminium foil
%structDose(k,1).RDose = dose(1); %Whitout any correction - RED CHANNEL
%structDose(k,1).CorrRDose = dose_q(1); %Considering only the quenching effect - RED CHANNEL
%structDose(k,1).Corr2RDose = dose_W(1); %Considering the quenching effect and the presence of the aluminium foil - RED CHANNEL
%structDose(k,1).RstdDose = std(1); %Whitout any correction  - RED CHANNEL
%structDose(k,1).RCorrstdDose = [Rstd_dose_q]; %Considering only the quenching effect - RED CHANNEL
%structDose(k,1).RCorr2stdDose = [Rstd_dose_W]; %Considering the quenching effect and the presence of the aluminium foil - RED CHANNEL
%structDose(k,1).GDose = dose(2); %Whitout any correction - GREEN CHANNEL
%structDose(k,1).CorrGDose = dose_q(2); %Considering only the quenching effect - GREEN CHANNEL
%structDose(k,1).Corr2GDose = dose_W(2); %Considering the quenching effect and the presence of the aluminium foil - GREEN CHANNEL
%structDose(k,1).GstdDose = std(2); %Whitout any correction - GREEN CHANNEL
%structDose(k,1).GCorrstdDose = [Gstd_dose_q]; %Considering only the quenching effect - GREEN CHANNEL
%structDose(k,1).GCorr2stdDose = [Gstd_dose_W]; %Considering the quenching effect and the presence of the aluminium foil - GREEN CHANNEL
%structDose(k,1).BDose = dose(3); %Whitout any correction - BLUE CHANNEL
%structDose(k,1).CorrBDose = dose_q(3); %Considering only the quenching effect - BLUE CHANNEL
%structDose(k,1).Corr2BDose = dose_W(3); %Considering the quenching effect and the presence of the aluminium foil - BLUE CHANNEL
%structDose(k,1).BstdDose = std(3); %Whitout any correction - BLUE CHANNEL
%structDose(k,1).BCorrstdDose = [Bstd_dose_q]; %Considering only the quenching effect - BLUE CHANNEL
%structDose(k,1).BCorr2stdDose = [Bstd_dose_W]; %Considering the quenching effect and the presence of the aluminium foil - BLUE CHANNEL

%close all;

end
%%
resultsTable = struct2table(structDose);
writetable(resultsTable,'DoseValues.txt');