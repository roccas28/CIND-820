proc import
out=diabetes_2021
datafile="/home/u63254551/sasuser.v94/diabetes_2021.csv"
dbms=csv replace;
getnames=yes;
proc print data=diabetes_2021 (obs=15);
title "Diabetes Health Indicators";
run;

ods graphics on;

proc hpsplit data=diabetes_2021;
class Diabetes HighBP HighChol CholCheck Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income Race BMICat CurrSmoke CurrESmoke Alcohol30;
model Diabetes = HighBP HighChol CholCheck Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income Race BMICat CurrSmoke CurrESmoke Alcohol30 ;
grow entropy;
prune costcomplexity;
run;


ods graphics on;

proc hpsplit data=diabetes_2021;
class Diabetes HighBP HighChol CholCheck Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income Race BMICat CurrSmoke CurrESmoke Alcohol30;
model Diabetes = HighBP HighChol CholCheck Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies HvyAlcoholConsump AnyHealthcare NoDocbcCost GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income Race BMICat CurrSmoke CurrESmoke Alcohol30 ;
grow gini;
prune costcomplexity;
run;

 