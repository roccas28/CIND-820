#YDATA Profiling
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r'C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\diabetes_binary_health_indicators_BRFSS2015.csv')
profile = ProfileReport(df, title="Profiling Report")
profile.to_file(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\ydata_report.html")



#SweetViz Profiling
import sweetviz as sv
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\diabetes_binary_health_indicators_BRFSS2015.csv')

report = sv.analyze(data)

report.show_html(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\sweetviz_report.html")
