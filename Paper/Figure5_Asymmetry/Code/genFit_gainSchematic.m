softmaxInvT = 3;
barWidth = 0.5;
barColor = [0 0 0];
Qpre = [1 0];
pA_pre = exp(softmaxInvT * Qpre) ./ sum(exp(softmaxInvT * Qpre));
gain = nan(3,1);
PS = nan(3,1);
figure(1); clf;

Qpost = [...
    -1 0;
    0.5 0;
    4 0];
for i=1:size(Qpost,1)
    pA_post = exp(softmaxInvT * Qpost(i,:)) ./ sum(exp(softmaxInvT * Qpost(i,:)));
    EVpre = sum(pA_pre .* Qpost(i,:),2);
    EVpost = sum(pA_post .* Qpost(i,:),2);
    gain(i) = EVpost-EVpre;
    PS(i) = abs(Qpost(i,1)-Qpre(1));
end
subplot(2,2,1); b1=bar(gain); ylim([-0.5 5]); b1.Parent.XTickLabel={'-1.0','0.5','4.0'}; b1.Parent.YTick = [0 1]; grid on; title('Update Q_1')
ylabel(sprintf('Prioritized\nReplay'))
ylabel(sprintf('Gain_{replay}'))
xlabel('Q-value after update')
subplot(2,2,3); b3=bar(PS); ylim([-0.5 5]); b3.Parent.XTickLabel={'-1.0','0.5','4.0'}; b3.Parent.YTick = [0 2]; grid on; %title('Update Q_1')
%ylabel(sprintf('Prioritized\nSweeping'))
ylabel(sprintf('Gain_{PE}'))
xlabel('Q-value after update')
set(b1,'FaceColor',barColor,'BarWidth',barWidth)
set(b3,'FaceColor',barColor,'BarWidth',barWidth)


Qpost = [...
    1 -1;
    1 0.5;
    1 4];
for i=1:size(Qpost,1)
    pA_post = exp(softmaxInvT * Qpost(i,:)) ./ sum(exp(softmaxInvT * Qpost(i,:)));
    EVpre = sum(pA_pre .* Qpost(i,:),2);
    EVpost = sum(pA_post .* Qpost(i,:),2);
    gain(i) = EVpost-EVpre;
    PS(i) = abs(Qpost(i,2)-Qpre(2));
end
subplot(2,2,2); b2=bar(gain); ylim([-0.5 5]); b2.Parent.XTickLabel={'-1.0','0.5','4.0'}; b2.Parent.YTick = [0 1]; grid on; title('Update Q_2')
xlabel('Q-value after update')
subplot(2,2,4); b4=bar(PS); ylim([-0.5 5]); b4.Parent.XTickLabel={'-1.0','0.5','4.0'}; b4.Parent.YTick = [0 2]; grid on; %title('Q_2=0')
xlabel('Q-value after update')
set(b2,'FaceColor',barColor,'BarWidth',barWidth)
set(b4,'FaceColor',barColor,'BarWidth',barWidth)
set(gcf, 'Position',[350   644   360   383])

%[ax4,h3]=suplabel('super Title'  ,'t');
suptitle('Original Q-values: [Q_1,Q_2] = [1,0]')
