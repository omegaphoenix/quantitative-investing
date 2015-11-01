function weight = strategy_rank_weight(universe, selected_ticker, check_day)

% convert [4,3,1,2] to [3,4,2,1] 
ranking = (1:length(selected_ticker))';
ranking(selected_ticker) = ranking;
[~,~,ranking] = unique(ranking);

weight = mean(ranking) - ranking;
%weight = sign(weight).*abs(weight).^(2);
weight = 2/sum(abs(weight))*weight;
