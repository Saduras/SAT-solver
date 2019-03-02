# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:37:14 2019

@author: Victor Zuanazzi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle

def titleNAxis(fig_plot, title, label_x, label_y):
    """enter title and axis labels with standard font sizes.
    """
    fig_plot.set_title(title, fontsize = 40)
    fig_plot.set_xlabel(label_x, fontsize = 30)
    fig_plot.set_ylabel(label_y, fontsize = 30)
    
    return fig_plot
    
def savePlot(fig_plot, fig_name, path = ".//figures//"):
    """saves a seaborn figure
    """
    try:
        fig = fig_plot.get_figure()
        print(fig, fig_plot)
        file_path = path + fig_name + ".png"
        print(file_path)
        fig.savefig(file_path)
        return file_path
    except:
        
        return False   
    
def plotAllCategoricals(df_exp, x_labels, y_labels, hue_labels = None, 
                     path = "..//figures//", plot_type = "all", 
                     name  = "", plot_size = (15, 7)):
    """plot and saves Categorical plots.
    Inputs:
        df_exp: (pd.DataFrame) containing the data to be ploted.
        x_labels: (list(str)) name of the columns to be used as x axis.
        y_labels:  (list(str)) name of the columns to be used as y axis.
        hue_labels: (list(str)) name of the columns to be used as hue.
        path = (str) folder to save the plots.
        plot_type (str or list(str)) if all use "all" otherwise specify a list
            of plot types. Plots currently supported are "strip", "swarm", 
            "point", "bar"
        name: (str) string to be added to the name of the plot. This does not 
            replace automatically generated name.
        plot_size: (tuple(float, float)) x and y sizes for the plot.
    Output:
        fails: (dict(list(int, list(tuple(str))))) 
    """
    
    if (plot_type == "all")  | (plot_type == ["all"]):\
        #all suported categorical plots
        cat_plots = ["strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"]
    else:
        #only use the specified plots
        cat_plots = plot_type
        
    fails = {x: [] for x in cat_plots}
    
    #plot and save everyting
    for cp in cat_plots:
        fails[cp] = plotCategoricals(df_exp, 
                                     x_labels, 
                                     y_labels, 
                                     hue_labels, 
                                     path, 
                                     plot_type, 
                                     name,
                                     plot_size)
        
    return fails
            
def allInOne(df_exp, hue, labels,
             path = "..//figures//", name  = "", plot_size = (5, 5)):
    
    #size of the figure
    size_x = plot_size[0]
    size_y = plot_size[1]
    
    all_labels = labels + [hue]
    f_plot = sns.pairplot(df_exp[all_labels], hue = hue, height= size_y, aspect=size_x/size_y)
    f_plot.fig.suptitle(name)
    
    try: 
        #saves the plot using a unique name.
        file_name = path + name + ".png"
        f_plot.savefig(file_name)
        file_path = file_name
    except:
        #if the file is not saved, the file path is returned as False.
        file_path = False
        
    return file_path
    
    
def plotCategoricals(df_exp, x_labels, y_labels, hue_labels = None, 
                     path = "..//figures//", plot_type = "bar", 
                     name  = "", plot_size = (15, 7)):
    """plot and saves Categorical plots.
    Inputs:
        df_exp: (pd.DataFrame) containing the data to be ploted.
        x_labels: (list(str)) name of the columns to be used as x axis.
        y_labels:  (list(str)) name of the columns to be used as y axis.
        hue_labels: (list(str)) name of the columns to be used as hue.
        path = (str) folder to save the plots.
        plot_type (str) name of the type of plot. Plots currently supported are
            "strip", "swarm", "point", "bar"
        name: (str) string to be added to the name of the plot. This does not 
            replace automatically generated name.
        plot_size: (tuple(float, float)) x and y sizes for the plot.
    Output:
        fails: (list(int, list(tuple(str)))) fails[0] outputs the number of 
            problematic plots and fails[1] ouputs a tuple containing 
            information about the problematic plots. The information is wraped 
            in a tuple in the sequence: x_label, y_label, hue_label, file_path.
        files: figures of extention .png
    """

    sns.set(style = "ticks")
    
    #size of the figure
    size_x = plot_size[0]
    size_y = plot_size[1]
    
    #Makes None iterable:
    if hue_labels == None:
        hue_labels = [hue_labels]
    
    #for loging problems
    fails = [0, []]  
    file_path = []
    
    for label_i in x_labels:
        for label_j in y_labels:
            for label_k in hue_labels:
                title = str(label_i) + " vs " + str(label_j) 
                try:
                    #actually plots stuff
                    f_plot = sns.catplot(x = label_i, 
                                         y = label_j, 
                                         hue = label_k, 
                                         data = df_exp, 
                                         height= size_y,
                                         aspect=size_x/size_y,
                                         kind = plot_type)   
                    f_plot.fig.suptitle(title)
                    try: 
                        #saves the plot using a unique name.
                        file_name = path + name + title + " " + str(label_k) + " " + plot_type + ".png"
                        f_plot.savefig(file_name)
                        file_path.append(file_name)
                    except:
                        #if the file is not saved, the file path is returned as False.
                        file_path.append(False)
                    
                except:
                    #logs the problem that failed to save the plot.
                    fails[0] += 1
                    fails[1].append((label_i, label_j, label_k, file_path))
        
    return fails

if __name__ == "__main__":

         #load saved experiments   
    filename = '..//data//experiment_stats.csv'
    df_exp = pd.read_csv(filename)
    
    x_categoricals = ["heuristic"]
                
    
    x_numericals = ["DP_calls",
                    "backtracks",
                    "split_calls", 
                    "unit_clause_calls"]
    
    y_labels = ["DP_calls",
                "split_calls",
                "backtracks",
                "unit_clause_calls",
                "solved_sudoku",
                "split_time", 
                "assign_calls",  
                "assign_time",
                "unit_clause_time",         
                "solve_time"]
    
    heuristics =["random", 
                 "next", 
                 "DLIS", 
                 "DLIS_max", 
                 "BOHM", 
                 "paretoDominant", 
                 "RF"]
#    
    hue_labels = [None]
    
    #working plot types: "strip", "swarm", "point", "bar", box, boxen, violin
    fails = plotCategoricals(df_exp, x_categoricals, y_labels, 
                             name = "",
                             plot_type = "boxen")
    print(fails)
    
#    fails = allInOne(df_exp, 
#                     hue = "heuristic", 
#                     labels = x_numericals, 
#                     name  = "categorical_pair_plot")
#    
#    print(fails)
#    
#    
#    
    
    