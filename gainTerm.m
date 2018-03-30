function [gain,saGain] = gainTerm(Q,planExp,params)

gain = cell(size(planExp));
saGain = nan(size(Q));
for i=1:numel(planExp)
    thisExp = planExp{i};
    gain{i} = nan(1,size(thisExp,1));
    for j=1:size(thisExp,1)
        Qmean = Q(thisExp(j,1),:); Qpre = Qmean;
        
        % Calculate probabilities BEFORE backup
        pA_pre = pAct(Qmean,params.planPolicy,params); % Probabilities BEFORE backup
        
        stp1Value = max(Q(thisExp(end,4),:),[],2); % Value of state stp1
        
        actTaken = thisExp(j,2);
        steps2end = size(thisExp,1)-j;
        rew = params.gamma.^(0:steps2end) * thisExp(j:end,3);
        Qtarget = rew + (params.gamma.^(steps2end+1)) .* stp1Value;
        if params.copyQinGainCalc
            Qmean(actTaken) = Qtarget;
        else
            Qmean(actTaken) = Qmean(actTaken) + params.alpha * (Qtarget-Qmean(actTaken));
        end
        
        % Calculate probabilities AFTER backups
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
