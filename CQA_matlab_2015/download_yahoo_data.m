function download_yahoo_data()
addpath ../dailyDataSuite
fid=fopen('r3000');
C=textscan(fid,'%s');
fclose(fid);

%
ticker={'^RUI','^GSPC'};
market=do_download(ticker);

%
ticker = C{1}(2:end);
stock={};
for i=1:length(ticker)
    ntry = 0;
    stock{i} = [];
    while(ntry < 3)
        try
            pause(ntry);
            stock{i}=do_download(ticker(i));
            ntry = inf;
        catch ME
            disp(ME);
            ntry = ntry + 1;
        end
    end
end

[~,per]=find(cellfun(@isempty,stock));
stock(per)=[];
ticker(per)=[];

tmp.data = cellfun(@(x) x.data, stock);
tmp.ticker = ticker;
stock = tmp;

stock.data = stock.data';

save Yahoo_20151030_r3000.mat stock market


function yahoo = do_download(symbols_yahoo)
data = getYahooDailyData(symbols_yahoo, '1950-01-01','2016-10-30','yyyy-mm-dd');
newdata={};
for i=1:length(symbols_yahoo)
    newdata{i} = data.(genvarname(symbols_yahoo{i}));
end
yahoo.data = newdata';
yahoo.ticker=symbols_yahoo;
