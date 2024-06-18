#YDATA Profiling
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021_v21.csv")
profile = ProfileReport(df, title="Profiling Report")
profile.to_file(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\ydata_report_2021_v3.html")



#SweetViz Profiling
import sweetviz as sv
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021_v21.csv")

report = sv.analyze(df)

report.show_html(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\sweetviz_report_2021_v3.html")


#import xport.v56

#xport (r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2022XPT\LLCP2022.XPT") > (r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2022XPT\LLCP2022.csv")