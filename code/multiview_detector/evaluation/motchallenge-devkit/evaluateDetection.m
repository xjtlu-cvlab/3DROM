function detResults=evaluateDetection(res_fpath,gt_fpath, chlname)
%% evaluate detections using P. Dollar's script

filename = split(res_fpath,'/');
filename = char(filename(end));
if contains(filename,'train')
    splitStrLong='Training Set';
    if strcmpi(chlname,'wildtrack')
        frames = 0:5:1795;
    elseif strcmpi(chlname,'multiviewx')
        frames = 0:359;
    elseif strcmpi(chlname,'terrace')
        % frames = 35:25:4985;
        frames = 35:25:2510;
        frames = [frames 25:25:2500];
        frames = [frames 15:25:2515];
    end
% elseif contains(filename,'val')
%     splitStrLong='Validation Set';
%     if strcmpi(chlname,'wildtrack')
%         frames = [1800,1805];
%     elseif strcmpi(chlname,'multiviewx')
%         frames = [360,361];
%     end
elseif contains(filename,'test')
    splitStrLong='Test Set';
    if strcmpi(chlname,'wildtrack')
        frames = 1800:5:1995;
    elseif strcmpi(chlname,'multiviewx')
        frames = 360:399;
    elseif strcmpi(chlname,'terrace')
        % frames = 25:25:5000;
        frames = 2535:25:5000;
        frames = [frames 2525:25:5000];
    end
end

addpath(genpath('.'));
    

% read sequence map

fprintf('Challenge: %s\n',chlname);
fprintf('Set: %s\n',splitStrLong);


        
gtInfo=[];
gtInfo.X=[];




% Find out the length of each sequence
% and concatenate ground truth
gtAll={};
detAll={};
allFrCnt=0;
evalMethod=1;
gtAllMatrix=zeros(0,4);
detAllMatrix=zeros(0,4);
    
    gtRaw=readtable(gt_fpath);
    gtRaw=gtRaw{:,:};

    % if something (a result) is missing, we cannot evaluate this tracker
    detRaw=readtable(res_fpath);
    detRaw=detRaw{:,:};
    if isempty(detRaw)
        detRaw = [frames',ones(length(frames),2).*inf];
    end
    
    % 
    detOne = {};
    for t=frames
        allFrCnt=allFrCnt+1;
        
        exgt=find(gtRaw(:,1)==t);
        gtAll{allFrCnt}=[gtRaw(exgt,2:3) zeros(length(exgt),1)];
        
        ng = length(exgt);
        oneFrame=[allFrCnt*ones(ng,1), (1:ng)', gtRaw(exgt,2:3)]; % set IDs to 1..ng
%         oneFrame=[(1:ng)', gtRaw(exgt,2:3)];
        gtAllMatrix=[gtAllMatrix; oneFrame];

        exdet=find(detRaw(:,1)==t);
        bbox=detRaw(exdet,2:3);
        detAll{allFrCnt}=bbox;
        detOne{allFrCnt}=bbox;
        
        ng = length(exdet);
        oneFrame=[allFrCnt*ones(ng,1), (1:ng)', detRaw(exdet,2:3)]; % set IDs to 1..ng
%         oneFrame=[(1:ng)', detRaw(exdet,2:3)];
        detAllMatrix=[detAllMatrix; oneFrame];
        
    end
    

detResults=[];
mcnt=1;



try
    fprintf('Evaluating \n');
    

    detectorRuntime=0;
        if evalMethod
        fprintf('Ok, results are valid. EVALUATING...\n');
        
        gt0=gtAll;
        dt0=detAll;
        
        [detMetsAll, detMetsInfo, detMetsAddInfo]=CLEAR_MOD_HUN(gtAllMatrix,detAllMatrix);

        
        detResults(mcnt).detMets=detMetsAll;

        
        fprintf('*** Dataset: %s ***\n',chlname);  
        printMetrics(detMetsAll);


    else
        fprintf('WARNING: %s cannot be evaluated\n',tracker);
        % update mysql, delete row
    end
    
    
catch err
    fprintf('WARNING: cannot be evaluated: %s\n',err.message);
    getReport(err) 
end


