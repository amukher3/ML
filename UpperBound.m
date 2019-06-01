clear all
close all
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\CData.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex5.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex7.mat');
Users= {(user1),(user2),(user3),(user5),(user6),(user7),(user8),(user9)};
DataIndexes = {(DataIndex1),(DataIndex2),(DataIndex3),(DataIndex5),(DataIndex6),(DataIndex7),(DataIndex8),(DataIndex9)};
%Users= {(user5),(user6)};
%DataIndexes = {(DataIndex5),(DataIndex6)};
for Num2=1:length(Users)
user=Users{Num2};
DataIndex=DataIndexes{Num2};
SamplingFreqn=32;
%[AlphaMLE,TauMLE,TauOneMLE,TauTwoMLE,FirstSample,SclTimeDomain,NumOfWindows,T]=EMForTheProblem(user,DataIndex);
[AlphaMLE,TauMLE,TauOneMLE,TauTwoMLE,FirstSample,SclTimeDomain,NumOfWindows,T]=EstimationProblem(user,DataIndex);
OnsetSample{Num2}=FirstSample;
AlphaEstimates{Num2} = AlphaMLE;
TauEstimates{Num2} = TauMLE;
TauOneEstimates{Num2}= TauOneMLE;
TauTwoEstimates{Num2}= TauTwoMLE;
EstimatedOnsetTimeInstants{Num2} = ((OnsetSample{Num2})/(1/T)) - TauEstimates{Num2}'; % in seconds 
EstimatedOnsetTimeInSamples{Num2}=round(EstimatedOnsetTimeInstants{Num2}*SamplingFreqn); %in samples
%% Reconstruction 

for j=1:NumOfWindows
TimeDurationOfEvents(j)= size(user,1)/SamplingFreqn; %in seconds
ts=0:1/SamplingFreqn:TimeDurationOfEvents(j); 
temp_prime(j,1:length(ts))=(AlphaMLE(j)*exp(-ts./TauOneMLE(j)) - AlphaMLE(j)*exp(-ts./TauTwoMLE(j)));
%temp_prime(j,1:length(ts))=(AlphaMLE(j)*exp(-ts./TauOneMLE) - AlphaMLE(j)*exp(-ts./TauTwoMLE));
SampleAmplitudes{Num2,j} = temp_prime(j,:);
[~,PeaksLocations{Num2,j}]= findpeaks(SampleAmplitudes{Num2,j});
end
temp=PeaksLocations{Num2};
EstimatedPeakSampleInstants{Num2} = EstimatedOnsetTimeInSamples{Num2} + temp;
% 
% %% Evaluation 
GroundPeakLocations=sort(DataIndex(1:2:length(DataIndex)-1));
GroundPeakLocationsTime{Num2} = GroundPeakLocations/SamplingFreqn;
PeaksLocationsTime{Num2}=EstimatedPeakSampleInstants{Num2}/SamplingFreqn;
UpperBoundMSE{Num2}=(sum((GroundPeakLocationsTime{Num2} - PeaksLocationsTime{Num2}).^2))/NumOfWindows;
end

