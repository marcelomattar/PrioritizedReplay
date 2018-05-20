%% PARAMETERS
clear; close all;
setParams
policy = 'softmax';
Q = 0:0.001:1;
Qspacing = 0.1; % Spacing between Q-values before update
Qrange = 3*Qspacing; % Range of Q-vaues to plot (before and after the original Q)
softmaxInvT = [1 2 5 10 20 50];
%gainBaseline = 0.005;
gainBaseline = -Inf;
saveBool = true;
grayrange = [0.3 0.9]; % Range of grayscale to use when plotting
grayrange = [0.8 0.3]; % Range of grayscale to use when plotting
yRange = [-0.1 0.2];

%% INITIALIZE STUFF
addpath('../../../')
gainOPT = nan(numel(Q),numel(softmaxInvT));
gainNOPT = nan(numel(Q),numel(softmaxInvT));

%% CALCULATE STUFF

for i=1:numel(softmaxInvT)
    % LEARNING ABOUT OPTIMAL ACTION
    Qmean = [median(Q)+Qspacing median(Q)];
    % Probabilities BEFORE update
    pA_pre = exp(softmaxInvT(i) * Qmean) ./ sum(exp(softmaxInvT(i) * Qmean));
    % Probabilities AFTER update
    Qpost = [Q' repmat(Qmean(2),numel(Q),1)];
    pA_post = exp(softmaxInvT(i) * Qpost) ./ repmat(sum(exp(softmaxInvT(i) * Qpost),2),1,2);
    gainOPT(:,i) = sum(Qpost .* pA_post,2) - (Qpost * pA_pre');
    
    % LEARNING ABOUT NON-OPTIMAL ACTION
    Qmean = [median(Q)-Qspacing median(Q)];
    % Probabilities BEFORE update
    pA_pre = exp(softmaxInvT(i) * Qmean) ./ sum(exp(softmaxInvT(i) * Qmean));
    % Probabilities AFTER update
    Qpost = [Q' repmat(Qmean(2),numel(Q),1)];
    pA_post = exp(softmaxInvT(i) * Qpost) ./ repmat(sum(exp(softmaxInvT(i) * Qpost),2),1,2);
    gainNOPT(:,i) = sum(Qpost .* pA_post,2) - (Qpost * pA_pre');
end
gainOPT = max(gainOPT,gainBaseline);
gainNOPT = max(gainNOPT,gainBaseline);


%% PLOT STUFF

subplot(1,2,1);
grayvals = grayrange(1) + linspace(0,1,(numel(softmaxInvT))) * (grayrange(2)-grayrange(1));
legendvals = {};
for i=numel(softmaxInvT):-1:1
    h1=plot(Q,gainOPT(:,i)); hold on;
    if softmaxInvT(i)==max(softmaxInvT)
        set(h1,'Color','r','LineWidth',1);
    else
        set(h1,'Color',repmat(grayvals(i),1,3),'LineWidth',0.5);
    end
    legendvals = [legendvals ; sprintf('beta=%.2f',softmaxInvT(i))];
end
h1.Parent.XTick = (median(Q)+Qspacing-Qrange):Qspacing:(median(Q)+Qspacing+Qrange);
h1.Parent.XTickLabel = repmat({''},1,numel(h1.Parent.XTick));
h1.Parent.XTickLabel{abs(h1.Parent.XTick-(median(Q)+Qspacing))<0.000001} = 'Q(a_1)';
h1.Parent.XTickLabel{abs(h1.Parent.XTick-median(Q))<0.000001} = 'Q(a_2)';
h1.Parent.YTick = -0.1:0.1:0.3;
h1.Parent.YTickLabel = {'','','','','',''};
%h1.Parent.FontSize=6;
ylabel('Gain');
grid on
axis equal
ylim(yRange);
xlim([(median(Q)+Qspacing-Qrange) (median(Q)+Qspacing+Qrange)]);
l11=line([(median(Q)) (median(Q))],ylim);
l12=line([(median(Q)+Qspacing) (median(Q)+Qspacing)],ylim);
l13=line(xlim,[0 0]);
set(l11,'LineWidth',0.5,'Color','k');
set(l12,'LineWidth',0.5,'Color','k');
set(l13,'LineWidth',0.5,'Color','k');
legend(legendvals,'Location','northeastoutside');
title('Optimal action');

subplot(1,2,2);
for i=numel(softmaxInvT):-1:1
    h2=plot(Q,gainNOPT(:,i)); hold on;
    if softmaxInvT(i)==max(softmaxInvT)
        set(h2,'Color','r','LineWidth',1);
    else
        set(h2,'Color',repmat(grayvals(i),1,3),'LineWidth',0.5);
    end
end
h2.Parent.XTick = (median(Q)-Qspacing-Qrange):Qspacing:(median(Q)-Qspacing+Qrange);
h2.Parent.XTickLabel = repmat({''},1,numel(h2.Parent.XTick));
h2.Parent.XTickLabel{abs(h2.Parent.XTick-(median(Q)-Qspacing))<0.000001} = 'Q(a_k)';
h2.Parent.XTickLabel{abs(h2.Parent.XTick-median(Q))<0.000001} = 'Q(a_1)';
h2.Parent.YTick = -0.1:0.1:0.3;
h2.Parent.YTickLabel = {'','','','','',''};
h2.Parent.FontSize=6;
ylabel('Gain');
grid on
axis equal
ylim(yRange);
xlim([(median(Q)-Qspacing-Qrange) (median(Q)-Qspacing+Qrange)]);
l21=line([(median(Q)) (median(Q))],ylim);
l22=line([(median(Q)-Qspacing) (median(Q)-Qspacing)],ylim);
l23=line(xlim,[0 0]);
set(l21,'LineWidth',0.5,'Color','k');
set(l22,'LineWidth',0.5,'Color','k');
set(l23,'LineWidth',0.5,'Color','k');
legend(legendvals,'Location','northeastoutside');
title('Non-optimal action(s)');

%set(gcf,'Position',[64   768   511   192])
set(gcf,'Position',[-1919         241        1920        1104])


%% EXPORT FIGURE
if saveBool
    save gainTerm.mat
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end

