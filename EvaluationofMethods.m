%This function calculates the Precision and Recall with respect to a
%distance parameter from the event instats i.e Precision/Recall vs.
%distance. 
clear all
close all
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\CData.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex5.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex4.mat');
load('C:\Users\Abhishek Mukherjee\Downloads\DataIndex7.mat');
Users= {(user1),(user2),(user3),(user4),(user5),(user6),(user7),(user8),(user9)};
DataIndexes = {(DataIndex1),(DataIndex2),(DataIndex3),(DataIndex4),(DataIndex5),(DataIndex6),(DataIndex7),(DataIndex8),(DataIndex9)};
%% Evaluation metrics for the proposed algorithm. 

S_d=0;S_r=0;Delta=3;

for Num2=1:length(Users)
user = Users{Num2}; 
DataIndex = DataIndexes{Num2};
%[AlphaMLE,TauMLE,TauOneMLE,TauTwoMLE,FirstSample,SclTimeDomain,NumOfWindows,T,SamplingFreqn]=EMForTheProblem(user,DataIndex,Delta);
[alpha_mle,TauMLE,tauOne_mle,tauTwo_mle,FirstSample,SclTimeDomain,NumOfWindows,T,SamplingFreqn]=EstimationProblem(user,DataIndex,Delta);
DownsamplingFactor = T*SamplingFreqn;
locations{Num2} = FirstSample; 
Tau_MLE_locs{Num2} = TauMLE; 
DetectedLocations{Num2} = (Delta+(locations{Num2}-1)*T) - TauMLE ; % for low sampling rate
%DetectedLocations{Num2} = (Delta+1*T+(locations{Num2}-1)*WindowTimeLength)-(Tau_MLE_locs{Num2}); % for low sampling rate 
OnsetSampleInstants{Num2} = sort(DataIndex(2:2:length(DataIndex)));
OnsetTimeInstants{Num2}=(OnsetSampleInstants{Num2}-1)/SamplingFreqn;
RealLocations{Num2}=OnsetTimeInstants{Num2};
RealLocationsSamples{Num2}=OnsetSampleInstants{Num2}; % Onset Sample Instants(Handlabelled)
S_d=S_d+length(DetectedLocations{Num2});
S_r=S_r+length(RealLocations{Num2});  
end

dmax=1:0.5:3.5; % vary this to see performance at larger distances. 
Sdr=zeros(1,length(dmax)); 
Srd=zeros(1,length(dmax)); 

for i=1:length(dmax)
for Num2=1:length(Users)
temp_RealLocations = RealLocations{Num2};
temp_DetectedLocations = DetectedLocations{Num2};
for j=1:length(temp_DetectedLocations)
 diff_in_time{j} = temp_DetectedLocations(j) - temp_RealLocations(:);
 if (any(abs(diff_in_time{j})<=dmax(i)))
 Sdr(i) = Sdr(i)+1;
 end
end
for j=1:length(temp_RealLocations)
 diff_in_time{j} = temp_RealLocations(j) - temp_DetectedLocations(:);
 if (any(abs(diff_in_time{j})<=dmax(i)))
 Srd(i)=Srd(i)+1;
 end
end
end
end
preci=Sdr/S_d;
recall=Srd/S_r;
Fscore(:)=2*(preci(:).*recall(:))./(preci(:)+recall(:));

%% Evaluation metric using the peaks:
S_d_peak=0; S_r_peak=0;
Srd_peak=zeros(1,length(dmax)); 
Sdr_peak=zeros(1,length(dmax)); 
DownsamplingFactor = T*SamplingFreqn;
for Num2=1:length(Users)
user(:)= Users{Num2};
if(Delta==0)
DownUser = downsample(user,DownsamplingFactor);
else
DownUser = user(round(Delta*SamplingFreqn):DownsamplingFactor:length(user));
end
DownSampledUsers{Num2} = DownUser;
[~,locPeakDetection]= findpeaks(DownSampledUsers{Num2});
DetectedPeaks{Num2}= locPeakDetection;
%[~,locPeakDetection] = findpeaks(UpSampledUser{Num2,:}); 
DetectedLocations_peak{Num2} = ((locPeakDetection-1)/(1/T))+Delta;
DataIndex = DataIndexes{Num2};
PeakSamplingInstants{Num2} = sort(DataIndex(1:2:length(DataIndex)-1)); % comparing with the labeled peaks
PeakSamplingInstantsTemps = PeakSamplingInstants{Num2}; 
PeaksTimeInstants{Num2} =((PeakSamplingInstantsTemps-1)/SamplingFreqn);
RealLocations_peak{Num2} = PeaksTimeInstants{Num2};
S_d_peak=S_d_peak+length(DetectedLocations_peak{Num2});
S_r_peak= S_r_peak+length(RealLocations_peak{Num2});
end
for i=1:length(dmax)
for Num2=1:length(Users)
temp_RealLocations_peak = RealLocations_peak{Num2};
temp_DetectedLocations_peak = DetectedLocations_peak{Num2};
for j=1:length(temp_DetectedLocations_peak)
 diff_in_time_peak{j} = temp_DetectedLocations_peak(j) - temp_RealLocations_peak;
 if (any(abs(diff_in_time_peak{j})<=dmax(i))) 
 Sdr_peak(i)=Sdr_peak(i)+1;
 end
end
for j=1:length(temp_RealLocations_peak)
 diff_in_time_peak{j} = temp_RealLocations_peak(j) - temp_DetectedLocations_peak;
 if (any(abs(diff_in_time_peak{j})<=dmax(i)))
 Srd_peak(i)=Srd_peak(i)+1;
 end
end
end
end
preci_peak_detection=Sdr_peak/S_d_peak;
recall_peak_detection=Srd_peak/S_r_peak;
Fscore_peak_detection(:)=2*(preci_peak_detection(:).*recall_peak_detection(:))./(preci_peak_detection(:)+recall_peak_detection(:));
figure;
plot(dmax,preci,'--o');
hold on
plot(dmax,preci_peak_detection,'--o');
xlabel('dmax in sec');
ylabel('Precision');
legend('Proposed Method','Peak Detection');
figure;
plot(dmax,recall,'--o');
hold on
plot(dmax,recall_peak_detection,'--o');
xlabel('dmax in sec');
ylabel('Recall');
legend('Proposed Method','Peak Detection');
figure;
plot(dmax,Fscore,'--o');
hold on
plot(dmax,Fscore_peak_detection,'--o');
xlabel('dmax in sec');
ylabel('Fscore');
legend('Proposed Method','Peak Detection');