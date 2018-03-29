%% PARAMETERS
setParams
policy = 'softmax';
Q = 0:0.01:100;
%softmaxT = [0 logspace(log10(0.01),log10(2),10)];
softmaxT = [0 0.1 0.2 0.5 1];
gainBaseline = 0.005;
saveBool = true;

%% INITIALIZE STUFF
addpath('../../../')
Qvar = [0 0];
gainOPT = nan(numel(Q),numel(softmaxT));
gainNOPT = nan(numel(Q),numel(softmaxT));

%% CALCULATE STUFF

for i=1:numel(softmaxT)
    params.softmaxT = softmaxT(i);
    
    % LEARNING ABOUT OPTIMAL ACTION
    Qmean = [median(Q)+0.1 median(Q)];
    % Probabilities BEFORE update
    pA_pre = pAct(Qmean,Qvar,policy,params);
    % Probabilities AFTER update
    Qpost = [Q' repmat(Qmean(2),numel(Q),1)];
    pA_post = pAct(Qpost,zeros(numel(Q),2),policy,params);
    gainOPT(:,i) = sum(Qpost .* pA_post,2) - (Qpost * pA_pre');
    
    % LEARNING ABOUT NON-OPTIMAL ACTION
    Qmean = [median(Q)-0.1 median(Q)];
    % Probabilities BEFORE update
    pA_pre = pAct(Qmean,Qvar,policy,params);
    % Probabilities AFTER update
    Qpost = [Q' repmat(Qmean(2),numel(Q),1)];
    pA_post = pAct(Qpost,zeros(numel(Q),2),policy,params);
    gainNOPT(:,i) = sum(Qpost .* pA_post,2) - (Qpost * pA_pre');
end
gainOPT = max(gainOPT,gainBaseline);
gainNOPT = max(gainNOPT,gainBaseline);


%% PLOT STUFF

subplot(1,2,1);
grayrange = [0.3 0.8];
grayvals = grayrange(1) + linspace(0,1,(numel(softmaxT)-1)) * (grayrange(2)-grayrange(1));
legendvals = {};
for i=numel(softmaxT):-1:1
    h1=plot(Q,gainOPT(:,i)); hold on;
    if softmaxT(i)==0
        set(h1,'Color','r','LineWidth',1);
    else
        set(h1,'Color',repmat(grayvals(i-1),1,3),'LineWidth',0.5);
    end
    legendvals = [legendvals ; sprintf('tau=%.2f',softmaxT(i))];
end
h1.Parent.XTick = (median(Q)-0.3):0.1:(median(Q)+0.4);
h1.Parent.XTickLabel = {'','','','Q(a_2)','Q(a_1)','','',''};
h1.Parent.YTick = -0.1:0.1:0.3;
h1.Parent.YTickLabel = {'','','','','',''};
h1.Parent.FontSize=6;
ylabel('Gain');
grid on
axis equal
ylim([-0.1 0.3]);
xlim([(median(Q)-0.3) (median(Q)+0.4)]);
l11=line([(median(Q)) (median(Q))],ylim);
l12=line([(median(Q)+0.1) (median(Q)+0.1)],ylim);
l13=line(xlim,[0 0]);
set(l11,'LineWidth',0.5,'Color','k');
set(l12,'LineWidth',0.5,'Color','k');
set(l13,'LineWidth',0.5,'Color','k');
legend(legendvals,'Location','northeastoutside');

subplot(1,2,2);
for i=numel(softmaxT):-1:1
    h2=plot(Q,gainNOPT(:,i)); hold on;
    if softmaxT(i)==0
        set(h2,'Color','r','LineWidth',1);
    else
        set(h2,'Color',repmat(grayvals(i-1),1,3),'LineWidth',0.5);
    end
end
h2.Parent.XTick = (median(Q)-0.4):0.1:(median(Q)+3);
h2.Parent.XTickLabel = {'','','','Q(a_k)','Q(a_1)','','',''};
h2.Parent.YTick = -0.1:0.1:0.3;
h2.Parent.YTickLabel = {'','','','','',''};
h2.Parent.FontSize=6;
ylabel('Gain');
grid on
axis equal
ylim([-0.1 0.3]);
xlim([(median(Q)-0.4) (median(Q)+0.3)]);
l21=line([(median(Q)-0.1) (median(Q)-0.1)],ylim);
l22=line([(median(Q)) (median(Q))],ylim);
l23=line(xlim,[0 0]);
set(l21,'LineWidth',0.5,'Color','k');
set(l22,'LineWidth',0.5,'Color','k');
set(l23,'LineWidth',0.5,'Color','k');
legend(legendvals,'Location','northeastoutside');

set(gcf,'Position',[64   768   511   192])


%% EXPORT FIGURE
if saveBool
    save gainTerm.mat
    
    set(gcf, 'renderer', 'painters');
    export_fig(['../Parts/' mfilename], '-pdf', '-eps', '-q101', '-nocrop', '-painters');
    %print(filename,'-dpdf','-fillpage')
end

