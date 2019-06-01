close all
load cdata.mat
load dataindex.mat
load DataIndex5.mat
load DataIndex7.mat

OriginalData = {user1; user2; user3; user5; user6; user7; user8; user9};
PeakIndices = {DataIndex1; DataIndex2; DataIndex3; DataIndex5; DataIndex6; DataIndex7; DataIndex8; DataIndex9};

Fs_original = 32;
t_upsample = 0:1/Fs_original:(length(user1)-1)/Fs_original;
Fs_downsample = 0.2;
% t_downsample = 0:1/Fs_downsample:((length(user1)/(Fs_original/Fs_downsample))-1)/Fs_downsample;

NumUsers = size(OriginalData,1);

Reconstruction = cell(NumUsers,1);
Splines = cell(NumUsers,1);
SplinePeaks = cell(NumUsers,1);
DownSample = cell(NumUsers,1);
FP = cell(NumUsers,1);

w = waitbar(0,'Initializing waitbar...');

for i=1:NumUsers
    perc = fix((i-1)/NumUsers*100);
    waitbar(perc/100,w,sprintf('Reconstructing %.0f of %.0f',i,NumUsers));
    Reconstruction{i} = ReconstructingSCR_PeakDetection(OriginalData{i},Fs_original,Fs_downsample);
    DownSample{i} = downsample(OriginalData{i},Fs_original/Fs_downsample);
    t_downsample = 0:1/Fs_downsample:(length(DownSample{i})-1)/Fs_downsample;
    Splines{i} = pchip(t_downsample,DownSample{i},t_upsample);
%     [~,SplinePeaks{i}] = findpeaks(Splines{i});
end

close(w)

% [MSEReconstruction,NMAEReconstruction,NumMissedPeaksReconstuction,GeometricMeanReconstruction,FoundPeaksReconstruction] = PeakMSE(PeakIndices,Reconstruction,Fs_original,Fs_downsample);
% [MSESplines,NMAESplines,NumMissedPeaksSplines,GeometricMeanSplines,FoundPeaksSplines] = PeakMSE(PeakIndices,Splines,Fs_original,Fs_downsample);
% [MSEPD,NMAEPD,NumMissedPeaksPD,GeometricMeanPD,FoundPeaksPD] = PeakMSE(PeakIndices,DownSample,Fs_original,Fs_downsample,'DownSampled');

[MSEPeaksReconstruction,MSEOnsetsReconstruction,MSEAmplitudesReconstruction,FoundPeaksReconstruction] = PeakMSE(PeakIndices,Reconstruction,OriginalData,Fs_original,Fs_downsample);
[MSEPeaksSplines,MSEOnsetsSplines,MSEAmplitudesSplines,FoundPeaksSplines] = PeakMSE(PeakIndices,Splines,OriginalData,Fs_original,Fs_downsample);
[MSEPeaksPD,MSEOnsetsPD,MSEAmplitudesPD,FoundPeaksPD] = PeakMSE(PeakIndices,DownSample,OriginalData,Fs_original,Fs_downsample,'DownSampled');

%{
figure(1)
c = distinguishable_colors(3);
subplot(3,1,1)
b = bar([1;2;3;5;6;7;8;9],[[MSEReconstruction{:}]' [MSESplines{:}]' [MSEPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Labeled Peak Time and Calculated Peak Time')
% legend('Proposed Method', 'PCHIP', 'PeakDetection')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

subplot(3,1,2)
b = bar([1;2;3;5;6;7;8;9],[[NMAEReconstruction{:}]' [NMAESplines{:}]' [NMAEPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('NMAE Between Labeled Peak Time and Calculated Peak Time')
% legend('Proposed Method', 'PCHIP', 'PeakDetection')
xlabel('User')
ylabel('NMAE (samples at 0.2 Hz)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

subplot(3,1,3)
b = bar([1;2;3;5;6;7;8;9],[[NumMissedPeaksReconstuction{:}]' [NumMissedPeaksSplines{:}]' [NumMissedPeaksPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('Difference Between Number of Labeled Peaks and Calculated Peaks')
legend('Proposed Method', 'PCHIP', 'Peak Detection')
xlabel('User')
ylabel('Number of Missed Peaks')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

figure(2)
subplot(2,1,1)
b = bar([1;2;3;5;6;7;8;9],[[GeometricMeanReconstruction{:}]' [GeometricMeanSplines{:}]' [GeometricMeanPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('Geometric Mean of MSE and Number of Missed Peaks')
% legend('Proposed Method', 'PCHIP', 'Peak Detection')
xlabel('User')
ylabel('Geometric Mean')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

Geomeans = [geomean([GeometricMeanReconstruction{:}]) geomean([GeometricMeanSplines{:}]) geomean([GeometricMeanPD{:}])];
subplot(2,1,2)
hold on
for i = 1:3
    b = bar(i,Geomeans(i));
    b.FaceColor = c(i,:);
end
hold off
title('Geometric Mean Across All Users')
legend('Proposed Method', 'PCHIP', 'Peak Detection','Location','Northwest')
xticks([])
xticklabels({})
ylabel('Geometric Mean')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)
%}

figure(1)
set(gcf,'Units','Normalized','OuterPosition',[0,0,0.75,0.75])
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEPeaksReconstruction{:}]' [MSEPeaksSplines{:}]' [MSEPeaksPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Labeled Peak Time and Closest Detected Peak Time')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)
saveas(gcf,'Precision','png')

figure(2)
subplot(2,1,1)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEOnsetsReconstruction{:}]' [MSEOnsetsSplines{:}]' [MSEOnsetsPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Labeled Onset Time and Closest Detected Onset Time')
% legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

subplot(2,1,2)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEOnsetsReconstruction{:}]' [MSEOnsetsSplines{:}]' [MSEOnsetsPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('Zoomed In')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
ylim([0 750])
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

figure(3)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEAmplitudesReconstruction{:}]' [MSEAmplitudesSplines{:}]' [MSEAmplitudesPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between SCR Amplitude Based on Labels and Amplitude of Closet Reconstruction Feature')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (\muS^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

[MSEPeaksReconstruction,MSEOnsetsReconstruction,MSEAmplitudesReconstruction,FoundPeaksReconstruction] = RecallPeakMSE(PeakIndices,Reconstruction,OriginalData,Fs_original,Fs_downsample);
[MSEPeaksSplines,MSEOnsetsSplines,MSEAmplitudesSplines,FoundPeaksSplines] = RecallPeakMSE(PeakIndices,Splines,OriginalData,Fs_original,Fs_downsample);
[MSEPeaksPD,MSEOnsetsPD,MSEAmplitudesPD,FoundPeaksPD] = RecallPeakMSE(PeakIndices,DownSample,OriginalData,Fs_original,Fs_downsample,'DownSampled');

figure(4)
set(gcf,'Units','Normalized','OuterPosition',[0,0,0.75,0.75])
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEPeaksReconstruction{:}]' [MSEPeaksSplines{:}]' [MSEPeaksPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Detected Peak Time and Closest Labeled Peak Time')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)
saveas(gcf,'Recall','png')

figure(5)
subplot(2,1,1)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEOnsetsReconstruction{:}]' [MSEOnsetsSplines{:}]' [MSEOnsetsPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Detected Onset Time and Closet Labeled Onset Time')
% legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

subplot(2,1,2)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEOnsetsReconstruction{:}]' [MSEOnsetsSplines{:}]' [MSEOnsetsPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('Zoomed In')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (s^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
ylim([0 750])
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

figure(6)
c = distinguishable_colors(3);
b = bar([1;2;3;5;6;7;8;9],[[MSEAmplitudesReconstruction{:}]' [MSEAmplitudesSplines{:}]' [MSEAmplitudesPD{:}]']);
for i=1:size(c,1)
    b(i).FaceColor = c(i,:);
end
title('MSE Between Ampitude of Reconstruction SCR and Amplitude of Closet SCR Based on Labels')
legend('Proposed Method', 'PCHIP', 'PeakDetection','Location','Northwest')
xlabel('User')
ylabel('MSE (\muS^2)')
set(gca,'FontSize',16)
set(gca,'FontWeight','bold')
% set(gca,'YScale','log')
grid on
set(gca,'Gridalpha',0.6)

