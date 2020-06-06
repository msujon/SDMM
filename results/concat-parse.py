import glob
import pandas as pd

#set panda option
pd.set_option('display.max_colwidth', -1)

prntrows=10

#get data files 
path=r'./'
filenames= glob.glob(path+ "/*.csv")

dat = [] 
for filename in filenames : 
    dat.append(pd.read_csv(filename))

spdata = pd.concat(dat, ignore_index=True) 


max1 = spdata['Speedup_exe_time'].max() 
min1 = spdata['Speedup_exe_time'].min() 
mean1 = spdata['Speedup_exe_time'].mean() 
median1 = spdata['Speedup_exe_time'].median() 
count1 = spdata['Speedup_exe_time'].count() 
count_slow = sum(spdata['Speedup_exe_time'] < 1.0) 
count_fast = sum(spdata['Speedup_exe_time'] > 1.0) 

print "max = " + str(max1)
print "min = " + str(min1)
print "mean = " + str(mean1)
print "count = " + str(count1)
print "count_slow = " + str(count_slow)
print "count_fast = " + str(count_fast)

if count1 != 0 :
    print ("   Perc of matices which speedup = %.2f %%\n" % (count_fast*100/count1))

sp = spdata.sort_values(by=['Speedup_exe_time'], ascending=True) 
print("Print by worst performing matrices (%d)\n" % prntrows)
print ("================================================\n")
print (sp[['FILENAME', 'Speedup_exe_time']].head(prntrows).to_string(header=False, index = False))  
print ("\n")
sp = spdata.sort_values(by=['Speedup_exe_time'], ascending=False) 
print ("Print by best performing matrices (%d)\n" % prntrows)
print ("================================================\n")
print (sp[['FILENAME', 'Speedup_exe_time']].head(prntrows).to_string(header=False, index = False))  
print ("\n")
print ("END of FILE\n") 
