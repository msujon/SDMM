#/usr/bin/python
import os
import glob
import numpy as np
import pandas as pd

#set panda option
pd.set_option('display.max_colwidth', -1)

prntrows=10


#path=r'./'
#all_files = glob.glob(path+"/*.csv")
all_files = glob.glob("*.csv")
#for fl in os.listdir(os.getcwd()) : 
for fl in all_files : 
    if fl.endswith(".csv") :
        #print "Summary of csv file = " + str(fl)
#spdata = pd.read_csv(r'../N100k-N200k_M256_a2b2_pt.csv');
        spdata = pd.read_csv(fl);
#print spdata.head()
#print spdata[['FILENAME', 'Speedup_exe_time']]  
        max1 = spdata['Speedup_exe_time'].max() 
        min1 = spdata['Speedup_exe_time'].min() 
        mean1 = spdata['Speedup_exe_time'].mean() 
        median1 = spdata['Speedup_exe_time'].median() 
        count1 = spdata['Speedup_exe_time'].count() 
#count_slow = spdata[spdata['Speedup_exe_time'] < 1.0 ].count() 
        count_slow = sum(spdata['Speedup_exe_time'] < 1.0) 
#count_fast = spdata[spdata['Speedup_exe_time'] > 1.0 ].count() 
        count_fast = sum(spdata['Speedup_exe_time'] > 1.0) 
       
        #writing on a file 
        outf = "Summary_"+os.path.splitext(fl)[0]+".txt" 
        f = open(outf, "w+")
        f.write ("Summary of csv file = " + str(fl)) 
        f.write("\n")
        f.write ("================================================\n")
        f.write ("Speedup data\n")
        #print "max = "  + str(max1)
        f.write("   max = %.2f\n" % (max1))
        #print "min = "  + str(min1)
        f.write("   min = %.2f\n" % (min1))
        #print "avg = " + str(mean1)
        f.write("   avg = %.2f\n" % (mean1))
        #print "median = " + str(median1)
        f.write("   median = %.2f\n" % (median1))
        #print "count = " + str(count1) 
        f.write("   count = %d\n" % (count1))
        #print "count slow = " + str(count_slow) 
        f.write("   count slow = %d\n" % (count_slow))
        #print "count fast = " + str(count_fast) 
        f.write("   count fast = %d\n" % (count_fast))
        #print "Percentage of speedup = " + str(count_fast*100/count1) + "%"
        if count1 != 0 :
            f.write("   Perc of matices which speedup = %.2f %%\n" % (count_fast*100/count1))
        f.write("\n")
        sp = spdata.sort_values(by=['Speedup_exe_time'], ascending=True) 
        f.write("Print by worst performing matrices (%d)\n" % prntrows)
        f.write ("================================================\n")
        f.write(sp[['FILENAME', 'Speedup_exe_time']].head(prntrows).to_string(header=False, index = False))  
        f.write("\n")
        sp = spdata.sort_values(by=['Speedup_exe_time'], ascending=False) 
        f.write("Print by best performing matrices (%d)\n" % prntrows)
        f.write ("================================================\n")
        f.write(sp[['FILENAME', 'Speedup_exe_time']].head(prntrows).to_string(header=False, index = False))  
        f.write("\n")
        f.write("END of FILE\n") 
