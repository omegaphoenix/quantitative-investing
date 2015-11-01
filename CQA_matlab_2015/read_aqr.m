function aqr = read_aqr()

MKT=xlsread('AQR/BAB_daily.xlsx','MKT');
RF =xlsread('AQR/BAB_daily.xlsx','RF');
BAB=xlsread('AQR/BAB_daily.xlsx','BAB Factors');

date=datetime(MKT(:,1),'ConvertFrom','excel');
aqr = fints(datenum(date),[MKT(:,25),RF(:,2), BAB(:,25)],{'mkt','rf','bab'},1);
