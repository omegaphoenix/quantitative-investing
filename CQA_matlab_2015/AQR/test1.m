clear all;

QMJ=xlsread('QMJ_daily.xlsx','QMJ Factors');
MKT=xlsread('QMJ_daily.xlsx','MKT');
RF=xlsread('QMJ_daily.xlsx','RF');

date=datetime(QMJ(:,1),'ConvertFrom','excel');

qmj_usa=QMJ(:,25);
mkt_usa=MKT(:,25);
rf_usa=RF(:,2);