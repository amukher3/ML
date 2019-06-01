function [MSEPeakTimes,MSEOnsetTimes,MSEAmplitudes,FoundPeaks] = RecallPeakMSE(PeakIndices,Reconstruction,Data,Fs_original,Fs_downsample,varargin)

% function [MSE,NormalizedMeanAbsoluteError,NumMissedPeaks,GeometricMean,FoundPeaks] = PeakMSE(PeakIndices,Data,Fs_original,Fs_downsample,varargin)

% Calculates the MSE between the peak times of a reconstruction and
% expertly labeled peak indices. Also gives the number of missed peaks
% labeled in the orignal data but not found in the reconstruction.

FoundPeaks = cell(1,size(Reconstruction,1));
FoundOnsets = cell(1,size(Reconstruction,1));
MSEPeakTimes = cell(1,size(Reconstruction,1));
MSEOnsetTimes = cell(1,size(Reconstruction,1));
MSEAmplitudes = cell(1,size(Reconstruction,1));
OnsetTimes = cell(1,size(PeakIndices,1));
% PeakAmplitudes = cell(1,size(Reconstruction,1));

for i=1:size(PeakIndices,1)
    PeakIndices{i} = sort(PeakIndices{i},'ascend');
    OnsetTimes{i} = PeakIndices{i}(2:2:size(PeakIndices{i},1));
    PeakIndices{i} = PeakIndices{i}(1:2:size(PeakIndices{i},1)-1);
    [~,FoundPeaks{i}] = findpeaks(Reconstruction{i});
    [~,FoundOnsets{i}] = findpeaks(-1*Reconstruction{i});
    if numel(FoundOnsets{i}) < numel(FoundPeaks{i})
        if size(FoundOnsets{i},1) == 1
            FoundOnsets{i} = horzcat(1,FoundOnsets{i});
        else 
            FoundOnsets{i} = vertcat(1,FoundOnsets{i});
        end
    end 
    if ~isempty(varargin)
        if varargin{1} == 'DownSampled'
            FoundPeaks{i} = FoundPeaks{i} *(Fs_original/Fs_downsample);
            FoundOnsets{i} = FoundOnsets{i} *(Fs_original/Fs_downsample);
        end
    end 
    MSEPeakTimes{i} = 0;
    MSEOnsetTimes{i} = 0;
    MSEAmplitudes{i} = 0;
    for j=1:size(FoundPeaks{i},1)
        [MinDistance,Index] = min(abs(PeakIndices{i}-FoundPeaks{i}(j)));
        if isempty(varargin)
            AmplitudeDifference = (Reconstruction{i}(FoundPeaks{i}(j))-Reconstruction{i}(FoundOnsets{i}(j)))-(Data{i}(PeakIndices{i}(Index))-Data{i}(OnsetTimes{i}(Index)));
        else
            AmplitudeDifference = (Reconstruction{i}((Fs_downsample/Fs_original)*FoundPeaks{i}(j))-Reconstruction{i}((Fs_downsample/Fs_original)*FoundOnsets{i}(j)))-(Data{i}(PeakIndices{i}(Index))-Data{i}(OnsetTimes{i}(Index)));
        end 
        MSEPeakTimes{i} = MSEPeakTimes{i}+(MinDistance/Fs_original)^2;
        MSEOnsetTimes{i} = MSEOnsetTimes{i}+((FoundOnsets{i}(j)-OnsetTimes{i}(Index))/Fs_original)^2;
        MSEAmplitudes{i} = MSEAmplitudes{i}+AmplitudeDifference^2;
    end
    MSEPeakTimes{i} = MSEPeakTimes{i}/(size(FoundPeaks{i},1));
    MSEOnsetTimes{i} = MSEOnsetTimes{i}/(size(FoundPeaks{i},1));
    MSEAmplitudes{i} = MSEAmplitudes{i}/(size(FoundPeaks{i},1));
end
end