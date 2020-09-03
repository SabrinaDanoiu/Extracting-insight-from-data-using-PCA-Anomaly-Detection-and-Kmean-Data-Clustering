#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import easygui
import tkinter as Tkinter
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkscrolledframe import ScrolledFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
from random import randrange



inputFilenameCSV = ""
importedCSV = None
csvCleanData = None

dataFromCSV = []
titlesOfCSV = []

# for anomaly detection
name1 = None
name2 = None
choice = None

#for k-mean
name1Kmean = None
name2Kmean = None
choiceKmean = None
noClusters = None



def errorWarning(errorDetails):
    err = Tkinter.Tk()
    err.title("Error")
    err.geometry("600x300")
    myFont = Tkinter.font.Font(family='Courier', size=400, weight='bold')
    label = Tkinter.Label(err, text=errorDetails, fg="red")
    label['font']= myFont
    label.place(x = 100, y = 120)
    
    return None
    
    
    
    
def getInputCSV():
    
    global importedCSV
    global inputFilenameCSV
    global dataFromCSV
    global titlesOfCSV
    inputFilenameCSV = easygui.fileopenbox(msg = 'Please locate the CSV file',
                    title = 'Specify File',
                    filetypes = '*.csv')
    
    if inputFilenameCSV[-4:] != ".csv" : 
        
        inputFilenameCSV = ""
        
        dataFromCSV = np.array([], dtype='object')
        titlesOfCSV = np.array([], dtype='object')
        setCSV()
        
        errorWarning("Please import a CSV file!!!")
        
        return None
        
    importedCSV = pd.read_csv(inputFilenameCSV)
    
    dataFromCSV = importedCSV.values
    titlesOfCSV = importedCSV.columns

    setCSV()
    cleanData()
    
    return None




def cleanData():
    global importedCSV
    global csvCleanData
    global dataFromCSV
    global titlesOfCSV
    
    csvCleanData = importedCSV.copy()
    
    csvValues = list(csvCleanData.values)
    okCsvValues = list()
    
    for row in csvValues:
        rowOk = True
        for value in row:
            if str(value).replace(" ", "") == "" or str(value) == "NaN" or str(value) == "nan":
                rowOk = False
                break
        if rowOk:
            okCsvValues.append(row)
            
    csvCleanData = pd.DataFrame(data=okCsvValues, columns=csvCleanData.columns)
    
    values = csvCleanData.values
    garbageColumns = list()
    
    for _ in range(3):
        inc = 0
        for test in values[randrange(10)]:
            try:
                aux = float(test)
            except:
                garbageColumns.append(inc)
            inc += 1
    columns = csvCleanData.columns
    for i in list(set(garbageColumns)):
            csvCleanData[columns[i]] = pd.factorize((csvCleanData[columns[i]]))[0]
            
    dataFromCSV = csvCleanData.values
    titlesOfCSV = csvCleanData.columns
    
    return None




def doPCA():
    
    global inputFilenameCSV
    global dataFromCSV
    global titlesOfCSV
    
    if(inputFilenameCSV == ""):
        errorWarning("No CSV file was imported!!!")
        return None
    
    csvCleanData = pd.DataFrame(data = dataFromCSV
             , columns = titlesOfCSV)
    pca = PCA(n_components=2)
    resultPCA = pca.fit_transform(csvCleanData)
    resultPCAdataframe = pd.DataFrame(data = resultPCA
             , columns = ['principal component 1', 'principal component 2'])
    
    
    dataFromCSV = np.array([], dtype='object')
    titlesOfCSV = np.array([], dtype='object')
 
    dataFromCSV = resultPCAdataframe.values
    titlesOfCSV = resultPCAdataframe.columns
    setCSV()
    
    rootPCA = Tkinter.Tk()
    rootPCA.title("Principal Component Analysis")
    rootPCA.geometry("1640x800")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('principal component 1', fontsize = 15)
    ax.set_ylabel('principal component 2', fontsize = 15)
    
    ax.set_title('2 component PCA', fontsize = 20)
    colors = ['r']
    for color in colors:
        ax.scatter(resultPCAdataframe.loc[:, 'principal component 1']
                   , resultPCAdataframe.loc[:, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.grid()
    
    canvas = FigureCanvasTkAgg(fig, master=rootPCA)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, rootPCA)
    toolbar.update()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
    
    rootPCA.mainloop()
    
    return None




def displayKmeanResult():

    global name1Kmean
    global name2Kmean
    
    if name1Kmean.get() == name2Kmean.get():
        errorWarning("Must choose 2 different columns!!!!")
        return None
        
    global choiceKmean
    choiceKmean.destroy()
     
    global dataFromCSV
    global titlesOfCSV
    global noClusters
    
    csvCleanData = pd.DataFrame(data = dataFromCSV
             , columns = titlesOfCSV)
     
    resultKmean = KMeans(n_clusters=noClusters.get(), random_state=0).fit(csvCleanData)
    csvCleanData['cluster_index'] = resultKmean.labels_
    dataFromCSV = csvCleanData.values
    titlesOfCSV = csvCleanData.columns
    setCSV()
    
    clusters = []
    targets = []
    moreColors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', "#7FFF38", '#D36838', '#00807E'
                 , '#CD807E', '#CD187E', '#CD18FF', '#000075', '#CDA7F1', '#00A7F1', '#560000', '#826162', '#D18F34', '#008000']
    rootKmean = Tkinter.Tk()
    rootKmean.title("K-mean Clustering")
    rootKmean.geometry("1640x800")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(name1Kmean.get(), fontsize = 15)
    ax.set_ylabel(name2Kmean.get(), fontsize = 15)
    ax.set_title(name1Kmean.get() + " x " + name2Kmean.get() + " Representation", fontsize = 20)
    for i in range(noClusters.get()):
        clusters.append("Cluster" + str(i))
        targets.append(i)
    colors = moreColors[0:noClusters.get()]
    for target, color in zip(targets,colors):
        indicesToKeep = csvCleanData['cluster_index'] == target
        ax.scatter(csvCleanData.loc[indicesToKeep, name1Kmean.get()]
                   , csvCleanData.loc[indicesToKeep, name2Kmean.get()]
                   , c = color
                   , s = 50)
    ax.legend(clusters)
    ax.grid()
    
    canvas = FigureCanvasTkAgg(fig, master=rootKmean)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, rootKmean)
    toolbar.update()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
    
    rootKmean.mainloop()
    
    return None




def decideKmeanType():
    
    global name1Kmean
    global name2Kmean
    global choiceKmean
    global noClusters
    
    columnName1 = None
    columnName2 = None
    
    choiceKmean = Tkinter.Tk()
    choiceKmean.title("K-mean Representation parameters ")
    myFont = Tkinter.font.Font(family='Courier', size=70, weight='bold')
    label = Tkinter.Label(choiceKmean, text="What should be used for K-mean representation? ")
    label['font'] = myFont
    label.grid( column = 0, row = 0 )
    labelNumberOfClusters = Tkinter.Label(choiceKmean, text="Number of clusters: ")
    labelNumberOfClusters['font'] = myFont
    labelNumberOfClusters.grid( column = 0, row = 3 )
    
    name1Kmean = Tkinter.StringVar(choiceKmean)
    name2Kmean = Tkinter.StringVar(choiceKmean)
    noClusters = Tkinter.IntVar(choiceKmean)
    
    options = list(titlesOfCSV)
    optionsClusterNumbers = range(2,21)
    
    name1Kmean.set(options[0])
    name2Kmean.set(options[1])
    noClusters.set(optionsClusterNumbers[0])
    
    columnName1Combo = Tkinter.OptionMenu(choiceKmean, name1Kmean ,*options)
    columnName2Combo = Tkinter.OptionMenu(choiceKmean, name2Kmean ,*options)
    clustersNumberCombo = Tkinter.OptionMenu(choiceKmean, noClusters ,*optionsClusterNumbers)
 
    columnName1Combo['font'] = myFont
    columnName2Combo['font'] = myFont
    clustersNumberCombo['font'] = myFont
    columnName1Combo.grid(column=0, row=1)
    columnName2Combo.grid(column=0, row=2)
    clustersNumberCombo.grid(column=0, row=4)
    
    Submit = Tkinter.Button(choiceKmean, text="OK", command=displayKmeanResult)
    Submit['font'] = myFont
    Submit.grid(column=0, row=5)
    
    choiceKmean.mainloop()
    
    return None




def doKmean():
    
    global inputFilenameCSV
    if(inputFilenameCSV == ""):
        errorWarning("No CSV file was imported!!!")
        return None
    
    decideKmeanType()
    
    return None




def displayAnomalyResult():
    
    global name1
    global name2
    
    if name1.get() == name2.get():
        errorWarning("Must choose 2 different columns!!!!")
        return None
        
    global choice
    choice.destroy()
    
    global dataFromCSV
    global titlesOfCSV
    
    csvCleanData = pd.DataFrame(data = dataFromCSV
             , columns = titlesOfCSV)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=10,contamination=0.005)
    lofFitPrediction = lof.fit_predict(csvCleanData)
    lofNegativeOutlierFactor = lof.negative_outlier_factor_
    prediction = pd.Series(lofFitPrediction).replace([-1,1],[1,0])
    anomalies = csvCleanData[prediction == 1]
    nonAnomalies = csvCleanData[prediction == 0]
        
    dataFromCSV = np.array([], dtype='object')
    titlesOfCSV = np.array([], dtype='object')
         
    csvCleanData['is_anomaly'] = prediction
        
    dataFromCSV = csvCleanData.values
    titlesOfCSV = csvCleanData.columns
    setCSV()
    
    root = Tkinter.Tk()
    root.title("Anomaly Detection")
    root.geometry("1640x800")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(name1.get(), fontsize = 15)
    ax.set_ylabel(name2.get(), fontsize = 15)
    ax.set_title(name1.get() + " x " + name2.get() + " Representation", fontsize = 20)
    targets = [1, 0]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = prediction == target 
        ax.scatter(csvCleanData.loc[indicesToKeep, name1.get()]
                   , csvCleanData.loc[indicesToKeep, name2.get()]
                   , c = color
                   , s = 50)
    ax.legend(["anomaly", "non-anomaly"])
    ax.grid()
    
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
    
    root.mainloop()
    
    return None
    
    
    
    
def decideAnomalyDetectionType():
    
    global name1
    global name2
    global choice
    
    columnName1 = None
    columnName2 = None
    
    choice = Tkinter.Tk()
    choice.title("Anomaly Detection parameters ")
    myFont = Tkinter.font.Font(family='Courier', size=70, weight='bold')
    label = Tkinter.Label(choice, text="What should be used for Anomaly detection? ")
    label['font'] = myFont
    label.grid( column = 0, row = 0 )
    
    name1 = Tkinter.StringVar(choice)
    name2 = Tkinter.StringVar(choice)
    
    options = list(titlesOfCSV)
    
    name1.set(options[0])
    name2.set(options[1])
    
    columnName1Combo = Tkinter.OptionMenu(choice, name1 ,*options)
    columnName2Combo = Tkinter.OptionMenu(choice, name2 ,*options)
 
    columnName1Combo['font'] = myFont
    columnName2Combo['font'] = myFont
    columnName1Combo.grid(column=0, row=1)
    columnName2Combo.grid(column=0, row=2)
    
    Submit = Tkinter.Button(choice, text="OK", command=displayAnomalyResult)
    Submit['font'] = myFont
    Submit.grid(column=0, row=4)
    
    choice.mainloop()
    
    return None
    

    
    
def doAnomalyDetection():
    global inputFilenameCSV
    if(inputFilenameCSV == ""):
        errorWarning("No CSV file was imported!!!")
        return None
    
    decideAnomalyDetectionType()
    
    return None




def setCSV():
    
    global inner_frame_csv
    global dataFromCSV
    global titlesOfCSV
    
    inner_frame_csv = sf.display_widget(Tkinter.Frame)
    row = column = 0
    if len(dataFromCSV) > 10:
        head = list(dataFromCSV[:5].copy())
        bot = list(dataFromCSV[-5:].copy())
        head_bot = head + [['...']] + bot
    else:
        head_bot = dataFromCSV.copy()
 
    head_bot = [titlesOfCSV] + head_bot
    
    for line in head_bot:
        for value in line:
            value_formated = value
            if row == 0 and type(value) == str:
                value_formated = ("\n".join(value.split())).replace("\"", "")
            w = Tkinter.Label(inner_frame_csv, width=15,
                      height=5,
                      borderwidth=2,
                      relief="groove",
                      anchor="center",
                      justify="center",
                      text=str(value_formated))

            w.grid(row=row,
                   column=column,
                   padx=4,
                   pady=4)
            if row == 0:
                w.config(background="black", foreground="white")
            column+=1
        column = 0
        row += 1
        
    return None




def downloadCSV():
    
    global inputFilenameCSV
    global dataFromCSV
    global titlesOfCSV
    
    if(inputFilenameCSV == ""):
        errorWarning("No CSV file was imported!!!")
        return None
    
    outputCSV = inputFilenameCSV
    outputCSV = outputCSV.replace(".csv","")
    import time;
    timeStamp = time.time()
    
    csvCleanData = pd.DataFrame(data = dataFromCSV
             , columns = titlesOfCSV)
    
    csvCleanData.to_csv(outputCSV+'_modified_'+ str(timeStamp) +'.csv', index=False)
    
    return None




# Main app

app = Tkinter.Tk()
app.title("Data representation Application")
app.geometry("1810x800")

myFont = Tkinter.font.Font(family='Courier', size=20, weight='bold')

importButton = Tkinter.Button(app, text='Import CSV Data', width=20, height=3, bg='#0052cc', fg='#ffffff',
                      activebackground='#0052cc', activeforeground='#aaffaa', command = getInputCSV)
importButton['font'] = myFont
importButton.place(x = 0, y = 0)

showPCAButton = Tkinter.Button(app, text='Use PCA on Data', width=20, height=3, bg='#0052cc', fg='#ffffff', 
                       activebackground='#0052cc', activeforeground='#aaffaa', command = doPCA)
showPCAButton['font'] = myFont
showPCAButton.place(x = 330, y = 0)

showKmeanButton = Tkinter.Button(app, text='Use K-mean on Data', width=20, height=3, bg='#0052cc', fg='#ffffff', 
                         activebackground='#0052cc', activeforeground='#aaffaa', command = doKmean)
showKmeanButton['font'] = myFont
showKmeanButton.place(x = 660, y = 0)

showAnomalyDetectionButton = Tkinter.Button(app, text='Use Anomaly Detection on Data', width=30, height=3, bg='#0052cc', fg='#ffffff', 
                                    activebackground='#0052cc', activeforeground='#aaffaa', command = doAnomalyDetection)
showAnomalyDetectionButton['font'] = myFont
showAnomalyDetectionButton.place(x = 990, y = 0)

downloadButton = Tkinter.Button(app, text='Download CSV', width=20, height=3, bg='#0052cc', fg='#ffffff',
                      activebackground='#0052cc', activeforeground='#aaffaa', command = downloadCSV)
downloadButton['font'] = myFont
downloadButton.place(x =1480, y = 0)

sf = ScrolledFrame(app, width=1780, height=660)
sf.place(x = 5, y = 120)

# Bind the arrow keys and scroll wheel
sf.bind_arrow_keys(app)
sf.bind_scroll_wheel(app)

inner_frame_csv = sf.display_widget(Tkinter.Frame)

app.mainloop()

