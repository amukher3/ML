function [WindowedSignal]=...
    MakingTimeWindows(TimeSig,WindowDuration,OverlapDuration,SamplingFreqn); 

%%% Making Time Windows 

WindowDuration=10; %Window time duration in seconds
WindowSize=WindowDuration*SamplingFreqn; %in samples


if(OverlapDuration==0)
    WindowedSignal = buffer(TimeSig,WindowSize);
else
    OverlapSize=OverlapDuration*SamplingFreqn; % in samples 
    WindowedSignal= buffer(TimeSig,WindowSize,OverlapSize);
end
end