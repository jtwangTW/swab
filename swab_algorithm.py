#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
Created on 01.03.2017

@author: Q416435
'''
import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def calculate_error(s1, s2):
    '''
    A function which takes in a time series and returns the approximation
    error of the linear segment approximation of it
    
    BESSER TODO: Use https://en.wikipedia.org/wiki/Least_trimmed_squares
    weil normaler Error failen kann!
    
    '''
    # cost of merging segment s1 and s2
    fuse = pd.concat([s1, s2])
    
    # define line
    times = fuse.index.astype(numpy.int64)
    values = [f[0] for f in fuse.values.astype(numpy.float64)]
    approximated_values = numpy.poly1d(numpy.polyfit(times, values, 1))(times)

    # Error measure
    mean_distance = (abs(values - approximated_values)).mean(axis=0)   
    return mean_distance

def bottom_up(T, max_error):
    seg_ts = []    
    if(len(T)<3):
        return [T]
    
    for i in range(0, len(T), 2):
        seg_ts = seg_ts + [T.iloc[i:i+2]]
        
    merge_cost = [0]* (len(seg_ts)-1)
    for i in range(0, len(seg_ts)-1):
        merge_cost[i] = calculate_error(seg_ts[i], seg_ts[i+1])

    while min(merge_cost) < max_error:
        index = merge_cost.index(min(merge_cost)) # find cheapest pair to merge
        seg_ts[index] = pd.concat([seg_ts[index], seg_ts[index+1]])
        del(seg_ts[index+1])
        del(merge_cost[index])
        if(len(seg_ts)==1):
            #print("\n\nSEG "+ str(seg_ts))
            #print("\nCosts "+ str(merge_cost))
            break
        if (index+1)< len(seg_ts): merge_cost[index] = calculate_error(seg_ts[index], seg_ts[index+1])
        merge_cost[index-1] = calculate_error(seg_ts[index-1], seg_ts[index]);

        #print("\n\nSEG "+ str(seg_ts))
        #print("\nCosts "+ str(merge_cost))
    
        
    return seg_ts

def best_line(max_error, input_df, w, start_idx, upper_bound):
    ''' 
    finds data corresponding to a single segment using the 
    (relatively poor) Sliding Windows and gives it to the buffer.
    '''
    error = 0
    idx = start_idx + len(w)
    S_prev = w
    if(idx >= len(input_df)):
        return S_prev
    
    while error <= max_error:

        # ADD ONE Point as long as smaller than error
        S = pd.concat([S_prev, input_df.iloc[idx:idx+1]])
        idx += 1
        
        times = S.index.astype(numpy.int64)
        values = [f[0] for f in S.values.astype(numpy.float64)]
        
        # line approximation
        approximated_values = numpy.poly1d(numpy.polyfit(times, values, 1))(times)
        # curve y(x) = a[0] * x + a[1]
        
        # determine error = mean of distance between points
        error = (abs(values - approximated_values)).mean(axis=0)
        
        if error <= max_error:
            S_prev = S
        
        if(len(S_prev)>upper_bound) or len(input_df.iloc[idx:idx+1])==0:
            break
        # todo: gebe 2 Punkte zurueck die die Linie darstellt fuer diesen 
        # Abschnitt wegen Zeit passt es trotzdem normal

    # return line of best fit approximated_values
    #S_prev = pd.TimeSeries([approximated_values[0], approximated_values[-2]], index=[S_prev.index[0], S_prev.index[-1]]).to_frame()
    return S_prev

def approximated_segment(in_seg):
    if len(in_seg)<3:
        in_seg['avg_value'] =-10# numpy.mean([f[0] for f in in_seg.values.astype(numpy.float64)])        
        return in_seg
        
    times = in_seg.index.astype(numpy.int64)
    values = [f[0] for f in in_seg.values.astype(numpy.float64)]
    approximated_values = numpy.poly1d(numpy.polyfit(times, values, 1))(times)
    new_seg = pd.TimeSeries([approximated_values[0], approximated_values[-2]], index=[in_seg.index[0], in_seg.index[-1]]).to_frame()
    new_seg['avg_value'] =-10# numpy.mean(values)
    return new_seg

def swab(input_df, max_error, seg_num, in_window_size):

    # 1. read in w data points     
    cur_nr = 0
    window_size = in_window_size
    w_nr = cur_nr + window_size
    tot_size = len(input_df)
    w = input_df.iloc[cur_nr:w_nr]
    lower_bound = w_nr/2
    upper_bound = 2*w_nr
    seg_ts = []
    
    
    plt.plot(input_df.index, input_df.values, marker="o", linestyle="None")
    
    print("Processing from "+ str(cur_nr) +" to "+ str(w_nr))
    print("From "+ str(w.iloc[0:1]) + " to "+ str(w.iloc[-2:-1]))
    last_run = False
    
    while True:
        T = bottom_up(w, max_error)
        
        # creates new approximated segment for T[0] and adds it
        seg_ts += [approximated_segment(T[0])] # add this segment represented by a line
        
        plt.plot(seg_ts[-1].index, seg_ts[-1].values)
        
        
        # finished if cur_nr > length of input
        if cur_nr >= tot_size or last_run:            
            if T[1:]: 
                seg_ts += [approximated_segment(T[1])]
            break
        
        # remove elements of T[0] from w
        cur_nr += len(T[0])-1 # overlap
        w_nr = cur_nr + window_size
        
        if (len(input_df) <= w_nr): 
            w_nr = -1
            last_run = True
            w  = input_df.iloc[cur_nr:]
        else:
            w  = input_df.iloc[cur_nr:w_nr] # - 4
        w = w.sort()
        w = best_line(max_error, input_df, w, cur_nr, upper_bound) # == w + best_line add further data points (=variable window)
        # adjust depending on lower and upper bound
        if len(w)>upper_bound:
            w = w.iloc[:upper_bound]
            print("Processing from "+ str(cur_nr) +" to "+ str(cur_nr+len(w)) + " \nTotal: " + str(tot_size))
        else: 
            print("Processing from "+ str(cur_nr) +" to "+ str(w_nr+len(w)) + " \nTotal: " + str(tot_size))
        if cur_nr > 1350:
            pass

    ''' iterate further!'''
    return seg_ts

    
def swab_alg(df, max_error = 0.1, window_size = 10, label_time="timestamp", label_value="mid", unit_time="s", thr_steady=0.1, thr_steep=1.75, plot_it = True):
    
    # 1. Load data
    df= df.sort(label_time)
    pre_df = pd.TimeSeries(df[label_value].get_values(), index=pd.to_datetime(df[label_time], unit = 's')).to_frame()

    # 2. SWAB Algorithm by Eamonn Keogh
    error_bound = (max(pre_df.icol(0)) - min(pre_df.icol(0)))*max_error
    res_df1 = swab(pre_df, error_bound, 10, window_size)#[0]
    
    # 3. assign value now depending on slope of curve
    # 1. plot points
    if plot_it:
        plt.plot(pre_df.index, pre_df.values, marker="o", linestyle="None")
        plt.show()
    
    # pass result segments to dataframe
    res_df = []
    first = True
    seg_id = 0
    for sub_df in res_df1: 
        seg_id += 1
        if len(sub_df) < 2: continue           
        slope = 1000000000*(sub_df.values[1][0] - sub_df.values[0][0])/(sub_df.index.astype(numpy.int64)[1]-sub_df.index.astype(numpy.int64)[0])
        assign = "steady" # -16 to 16 degrees
        if slope > thr_steady and slope <= thr_steep: assign = "increase" # 10 to 60 degrees
        if slope > thr_steep: assign = "steep_increase" # 60 to 90 degrees
        if slope < -thr_steady and slope >= -thr_steep: assign = "decrease"  # -10 to -60 degrees
        if slope < -thr_steep: assign = "steep_decrease" # -60 to -90 degrees
        sub_df["trend"] = pd.Series([assign, assign], index=sub_df.index)
        sub_df["seg_id"] = seg_id
        sub_df[label_time] = sub_df.index.astype(numpy.int64)[0]
        sub_df[label_time+"_end"] = sub_df.index.astype(numpy.int64)[1]
        
        # segment 
        if first:
            res_df = sub_df.head(1)
            first = False
        else: 
            ''' if same as previous remove redundant information NEU? '''
            res_df = pd.concat([res_df, sub_df.head(1)])

    res_df[label_value] = res_df[res_df.columns[0]]
    return res_df[[label_time, label_time+"_end", label_value, "trend", "seg_id"]]
    

if __name__ == '__main__':
    ''' USAGE '''
    input_df = pd.read_csv("example.csv", delimiter =",")
    swab_alg(input_df, max_error = 0.1, window_size = 10, label_time="timestamp", label_value="mid", unit_time="s", plot_it=True)