function [gain,saGain] = gainTerm(Q,planExp,params)

gain = cell(size(planExp));
saGain = nan(size(Q));
for i=1:numel(planExp)
    thisExp = planExp{i};
    gain{i} = nan(1,size(thisExp,1));
    for j=1:size(thisExp,1)
        Qmean = Q(thisExp(j,1),:); Qpre = Qmean;
        
        % Policy BEFORE backup
        pA_pre = pAct(Qmean,params.planPolicy,params);
        
         % Value of state stp1
        if strcmp(params.onVSoffPolicy,'on-policy')
            stp1Value = sum(Q(thisExp(end,4),:) .* pAct(Q(thisExp(end,4),:),params.planPolicy,params));
        else
            stp1Value = max(Q(thisExp(end,4),:),[],2);
        end
        
        actTaken = thisExp(j,2);
        steps2end = size(thisExp,1)-j;
        rew = params.gamma.^(0:steps2end) * thisExp(j:end,3);
        Qtarget = rew + (params.gamma.^(steps2end+1)) .* stp1Value;
        if params.copyQinPlanBkps
            Qmean(actTaken) = Qtarget;
        else
            Qmean(actTaken) = Qmean(actTaken) + params.alpha * (Qtarget-Qmean(actTaken));
        end
        
        % Policy AFTER backup
        pA_post = pAct(Qmean,params.planPolicy,params);
        
        % Calculate gain
        EVpre = sum(pA_pre .* Qmean,2);
        EVpost = sum(pA_post .* Qmean,2);
        gain{i}(j) = EVpost-EVpre;
        Qpost = Qmean;
        
        % Save on gain(s,a)
        saGain(thisExp(j,1),thisExp(j,2)) = max(saGain(thisExp(j,1),thisExp(j,2)),gain{i}(j));
    end
end
