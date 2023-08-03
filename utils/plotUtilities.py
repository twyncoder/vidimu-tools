import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils.signalProcessing as sp
import utils.fileProcessing as fp

#VIDEO

def plotVideoPoseAllSubjectsOneActivity(inpath,outpath,subjects,activity,activity_legend,outputfilename=None):
    ncols = 6
    nrows = len(subjects)

    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*3)) #40,nrows*3
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)

    for i,subject in enumerate(subjects):
        inpathfolder = os.path.join(inpath,subject)
        dfcsv = None
        # 1) Compute video joint's angle signals
        for trial in ["01","02","03","04","05"]:
            trialname='T'+trial
            csvsubjacttrial = subject+"_"+activity+"_"+trialname+'.csv'
            inpathmotfull = os.path.join(inpathfolder,csvsubjacttrial)
            if not os.path.exists(inpathmotfull):
                continue
            else:
                #print("Reading: ",inpathmotfull)
                dfcsv=fp.readCSV(inpathfolder,subject,activity,trialname)
                break #limit to the first existing trial
        if dfcsv is None:
            print("None")
            continue

        arm_flex_r,arm_flex_l,elbow_flex_r,elbow_flex_l,knee_angle_r,knee_angle_l=fp.getMainJointAnglesFromCSV2(dfcsv)
        arm_flex_r_filt = sp.applyMedianFilter(arm_flex_r,11)
        arm_flex_l_filt = sp.applyMedianFilter(arm_flex_l,11)
        elbow_flex_r_filt = sp.applyMedianFilter(elbow_flex_r,11)
        elbow_flex_l_filt = sp.applyMedianFilter(elbow_flex_l,11)        
        knee_angle_r_filt = sp.applyMedianFilter(knee_angle_r,11)
        knee_angle_l_filt = sp.applyMedianFilter(knee_angle_l,11)

        bodyjoints = ['right shoulder', 'right elbow','right knee','left shoulder','left elbow','left knee']
        for j in range(ncols):
            if j == 0:
                color = 'r'
                imujoint="arm_flex_r"
                X=np.arange(0,arm_flex_r_filt.shape[0])
                axes[i][j].set_ylim([-90,180])
                axes[i][j].plot(X,arm_flex_r_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==3:
                color = 'r'
                imujoint="arm_flex_l"
                X=np.arange(0,arm_flex_l_filt.shape[0])
                axes[i][j].set_ylim([-90,180])
                axes[i][j].plot(X,arm_flex_l_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==1:
                color = 'g'
                imujoint="elbow_flex_r"
                X=np.arange(0,elbow_flex_r_filt.shape[0])
                axes[i][j].set_ylim([-10,180])
                axes[i][j].plot(X,elbow_flex_r_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==4:
                color = 'g'
                imujoint="elbow_flex_l"
                X=np.arange(0,elbow_flex_l_filt.shape[0])
                axes[i][j].set_ylim([-10,180])
                axes[i][j].plot(X,elbow_flex_l_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==2:
                color = 'b'
                imujoint="knee_angle_r"
                X=np.arange(0,knee_angle_r_filt.shape[0])
                axes[i][j].set_ylim([-20,120])
                axes[i][j].plot(X,knee_angle_r_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==5:
                color = 'b'
                imujoint="knee_angle_l"
                X=np.arange(0,knee_angle_l_filt.shape[0])
                axes[i][j].set_ylim([-20,120])
                axes[i][j].plot(X,knee_angle_l_filt,color)
                axes[i][j].set_title(csvsubjacttrial + ' (Estimated angle: '+str(bodyjoints[j])+')')
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")

    title="Activity "+activity+": "+activity_legend+ " (one subject per row)"
   
    plt.suptitle(title,fontsize=18, verticalalignment='top',y=1.0)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)

    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath,outputfilename+'.svg'),format='svg')
        plt.savefig(os.path.join(outpath,outputfilename+'.pdf'),format='pdf')
        #plt.savefig(os.path.join(outpath,outputfilename+'.png'),format='png',dpi=600)
    
    plt.show()

# IMUS RAW

def plotRawQuaternionsPerActivity(inpath,outpath,subjects,activity,activity_legend,imu_list,outputfilename=None):    
    ncols = len(imu_list)
    nrows = len(subjects)

    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*3))
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)

    for i,subject in enumerate(subjects):
        #Try to open the first existing trial for that subject
        for trial in ["01","02","03","04","05"]:
            trialname='T'+trial
            rawsubjacttrial = subject+"_"+activity+"_"+trialname
            rawfilename = subject+"_"+activity+"_"+trialname+'.raw'
            inpathtxtfull = os.path.join(inpath,subject,rawfilename)
            if not os.path.exists(inpathtxtfull):
                continue
            else:
                dfraw = pd.read_csv(inpathtxtfull,sep=',')
                break #limit to the first existing trial

        for j,imu in enumerate(imu_list):
            dfnew = dfraw[dfraw['QUAT']==imu]
            dfnew = dfnew.reset_index(drop=True)

            #Remove first line (N-pose)
            dfnew = dfnew.iloc[1:]

            axes[i][j].set_title(rawfilename + ' (IMU: '+str(imu)+')')
            axes[i][j].plot(dfnew['w'],color='black')
            axes[i][j].plot(dfnew['x'],color='red')
            axes[i][j].plot(dfnew['y'],color='green')
            axes[i][j].plot(dfnew['z'],color='blue')
            axes[i][j].legend(['w','x','y','z'])
            axes[i][j].set_xlabel("Samples (50 Hz)")

    title="Activity "+activity+": "+activity_legend+ " (one subject per row)"
    plt.suptitle(title,fontsize=18, verticalalignment='top',y=1.0)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath,outputfilename+'.svg'),format='svg')
        plt.savefig(os.path.join(outpath,outputfilename+'.pdf'),format='pdf')
        #plt.savefig(os.path.join(outpath,outputfilename+'.png'),dpi=600))
    plt.show()

#IMUS MOT

def plotMotJointAnglesPerActivity(inpath,outpath,subjects,activity,activity_legend,motsignals,motrange,outputfilename=None):
    ncols = len(motsignals)
    nrows = len(subjects)

    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*3))
    prefixname = 'ik_'
    for i,subject in enumerate(subjects):

        dfmot = None
        #Try to open the first existing trial for that subject
        for trial in ["01","02","03","04","05"]:
            trialname='T'+trial
            motsubjacttrial = subject+"_"+activity+"_"+trialname
            motfilename = prefixname+motsubjacttrial+".mot"
            inpathmotfull = os.path.join(inpath,subject,motfilename)
            #print(inpathmotfull)
            if not os.path.exists(inpathmotfull):
                continue
            else:
                dfmot = pd.read_csv(inpathmotfull,skiprows=6,sep='\t')
                break #limit to the first existing trial
        if dfmot is None:
            continue

        for j in range(ncols):
            if j > len(motsignals)-1:
                continue
            #idxsignal=signalsToPlot[i*ncols+j]
            imujoint=motsignals[j]
            yaxis0=motrange[imujoint][0]
            yaxis1=motrange[imujoint][1]
            if imujoint == '': 
            #Para alinear gráficas
                continue
            jointangle_imus = dfmot[imujoint].to_numpy().flatten()[2:]
            X=np.arange(0,jointangle_imus.shape[0])
            if 'pelvis' in imujoint:
                color = 'r'
            if 'hip' in imujoint:
                color = 'g'
            if 'knee' in imujoint:
                color = 'b'
            if 'lumbar' in imujoint:
                color = 'r'
            if 'arm' in imujoint:
                color = 'g'
            if 'elbow' in imujoint:
                color = 'b'
            if 'wrist' in imujoint:
                color = 'r'
            axes[i][j].plot(X,jointangle_imus,color)
            axes[i][j].set_ylim([yaxis0, yaxis1])
            axes[i][j].set_title(imujoint+' '+motsubjacttrial)
            axes[i][j].set_title(motfilename + ' ('+str(imujoint)+')')
            axes[i][j].set_ylabel("Degrees")
            axes[i][j].set_xlabel("Samples (50 Hz)")

    title="Activity "+activity+": "+activity_legend+ " (one subject per row)"
    plt.suptitle(title,fontsize=18, verticalalignment='top',y=1.0)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    if outputfilename:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath,outputfilename+'.svg'),format='svg')
        plt.savefig(os.path.join(outpath,outputfilename+'.pdf'),format='pdf')
        #plt.savefig(os.path.join(outpath,outputfilename+'.png'),format='png',dpi=600))

    plt.show()
    
# OTHER UTILITIES (SYNC)

def plot_both(jointangle_imus,jointangle_video,title=None,outputfile=None):
    X=np.arange(0,jointangle_imus.shape[0])
    plt.plot(X,jointangle_imus,color='r',label='imus')
    X=np.arange(0,jointangle_video.shape[0])
    plt.plot(X,jointangle_video,color='b',label='video')
    plt.legend()
    if title:
        plt.suptitle(title)
    if outputfile:
        plt.savefig(os.path.join(outputfile+'.svg'),format='svg')
        plt.savefig(os.path.join(outputfile+'.pdf'),format='pdf')
    plt.show()
