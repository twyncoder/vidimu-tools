import os
import pandas as pd
import numpy as np
import re
import utils.signalProcessing as sp


### .CSV UTILITIES

def readCSV(folder,subject,activity,trial):
    csvfilename = subject+"_"+activity+"_"+trial+".csv"
    inpath_csv = os.path.join(folder,csvfilename)
    try:
        dfcsv = pd.read_csv(inpath_csv)
    except FileNotFoundError:    
        dfcsv = None
    return dfcsv

def getJointAngleCsvAsNP(bonesCSV):
    bone1,bone2,bone3,bone4 = bonesCSV
    segmA = bone1-bone2
    segmB = bone3-bone4
    jointangle_video = np.zeros((segmA.shape[0]))
    index=0
    for v1,v2 in zip(segmA,segmB):
        angle = sp.angle_between(v2,v1)
        jointangle_video[index]=angle
        index+=1
    return jointangle_video

def getMainJointAnglesFromCSV2(dfcsv):
    torso = dfcsv[['torso_x', 'torso_y', 'torso_z']].to_numpy()
    neck = dfcsv[['neck_x', 'neck_y', 'neck_z']].to_numpy()
    rshoulder = dfcsv[['right_shoulder_x','right_shoulder_y', 'right_shoulder_z']].to_numpy()
    lshoulder = dfcsv[['left_shoulder_x','left_shoulder_y', 'left_shoulder_z']].to_numpy()
    relbow = dfcsv[['right_elbow_x', 'right_elbow_y', 'right_elbow_z']].to_numpy()
    lelbow = dfcsv[['left_elbow_x', 'left_elbow_y', 'left_elbow_z']].to_numpy()
    rwrist = dfcsv[['right_wrist_x', 'right_wrist_y', 'right_wrist_z']].to_numpy()
    lwrist = dfcsv[['left_wrist_x', 'left_wrist_y', 'left_wrist_z']].to_numpy()
    rhip = dfcsv[['right_hip_x', 'right_hip_y', 'right_hip_z']].to_numpy()
    lhip = dfcsv[['left_hip_x', 'left_hip_y', 'left_hip_z']].to_numpy()
    rknee = dfcsv[['right_knee_x', 'right_knee_y', 'right_knee_z']].to_numpy()
    lknee = dfcsv[['left_knee_x', 'left_knee_y', 'left_knee_z']].to_numpy()
    rankle = dfcsv[['right_ankle_x', 'right_ankle_y', 'right_ankle_z']].to_numpy()
    lankle = dfcsv[['left_ankle_x', 'left_ankle_y', 'left_ankle_z']].to_numpy()  
    arm_flex_r = getJointAngleCsvAsNP([rshoulder,relbow,neck,torso])   
    arm_flex_l =  getJointAngleCsvAsNP([lshoulder,lelbow,neck,torso])   
    elbow_flex_r = getJointAngleCsvAsNP([rshoulder,relbow,relbow,rwrist])        
    elbow_flex_l = getJointAngleCsvAsNP([lshoulder,lelbow,lelbow,lwrist])  
    knee_angle_r = getJointAngleCsvAsNP([rhip,rknee,rknee,rankle])
    knee_angle_l = getJointAngleCsvAsNP([lhip,lknee,lknee,lankle])

    return arm_flex_r,arm_flex_l,elbow_flex_r,elbow_flex_l,knee_angle_r,knee_angle_l

def getJointAngleCsvAsNP(bonesCSV):
    bone1,bone2,bone3,bone4 = bonesCSV
    segmA = bone1-bone2
    segmB = bone3-bone4
    jointangle_video = np.zeros((segmA.shape[0]))
    index=0
    for v1,v2 in zip(segmA,segmB):
        #print(v1,v2)
        angle = sp.angle_between(v2,v1)
        jointangle_video[index]=angle
        index+=1
    return jointangle_video

### .OUT UTILITIES

def findNumberFrames(fileinfullpath,skipelines=2):
    with open(fileinfullpath) as f:
        for i in range(skipelines):
            next(f)
        content=f.read()
        frames= re.split("3d KeyPoints: |KeyPoints: ",content)
        nframes = int((len(frames)-1)/2)
    return nframes

def getJoints3DFromFile(fileinfullpath,jointlist3D,skiplines=2):
    with open(fileinfullpath) as f:
        lines = f.readlines()
        idxframe3D = 0
        nframes = findNumberFrames(fileinfullpath,skiplines)
        joints3D = np.ndarray(shape=(nframes,len(jointlist3D),3), dtype=float)
        for idxline,line in enumerate(lines):
            if idxline < skiplines:
                continue
            if "3d KeyPoints: " in line:
                #print("found ",idxline)
                for idxjoint,joint in enumerate(jointlist3D):
                    linecoords = lines[idxline+idxjoint+1]
                    #xyz = [float(item) for item in linecoords.split()]
                    xyz = [item for item in linecoords.split()]
                    if len(xyz)<3:
                        #For example line containign: 308.3-1012.3 6637.8
                        x = linecoords[0:7]
                        y = linecoords[7:14] 
                        z = linecoords[14:21]
                        xyz = [x,y,z]
                        #print(idxline,' ',linecoords,' ',idxframe3D,' ',len(xyz))
                        #print(xyz)
                    else:
                        xyz = [float(item) for item in xyz]
                    joints3D[idxframe3D,idxjoint]=xyz
                    if idxframe3D>nframes:
                        break
                #print(joints3D[idxframe3D,:,:])
                idxframe3D = idxframe3D + 1
    return joints3D,nframes

def convertJoints3DToDataframe(joints3D, jointlist3D):
    listcoords = ['_x','_y','_z']
    df= pd.DataFrame()
    for idxjoint,joint in enumerate(jointlist3D):
        for idxcoord,coord in enumerate(listcoords):
            df2= pd.DataFrame()
            df2[joint+coord]=joints3D[:,idxjoint,idxcoord]
            df=pd.concat([df,df2],axis=1)
        df = df.astype(float)
    return df

### .MOT UTILITIES

def getJointAngleMotAsNP(dfmot,jointMot):
    jointangle_mot = dfmot[jointMot].to_numpy().flatten()[1:]
    return jointangle_mot

def readMOTandCSV(folder,subject,activity,trial):
    motprefixname = 'ik_'
    motfilename = motprefixname+subject+"_"+activity+"_"+trial+".mot"
    inpath_mot = os.path.join(folder,motfilename)
    try:
        dfmot = pd.read_csv(inpath_mot,skiprows=6,sep='\t')
    except FileNotFoundError:    
        dfmot = None    
    csvfilename = subject+"_"+activity+"_"+trial+".csv"
    inpath_csv = os.path.join(folder,csvfilename)
    try:
        dfcsv = pd.read_csv(inpath_csv)
    except FileNotFoundError:    
        dfcsv = None
    #print("READ: "+str(inpath_mot)+", "+str(inpath_csv))
    return dfmot, dfcsv

# SYNC UTILITIES
def remove_insidelines_file(infile,idxstart,linestoremove,outfile):
    try:
        with open(infile, 'r') as fr:
            lines = fr.readlines()
            ptr=1
            with open(outfile, 'w') as fw:
                for line in lines:
                    if ptr < idxstart or ptr > (idxstart+linestoremove-1):
                        fw.write(line)
                    ptr = ptr + 1
    except:
        print("Oops! error in remove_insidelines ", infile, idxstart,linestoremove,outfile)


def remove_endlines_file(infile,linestoremove,outfile):
    try:
        with open(infile, 'r') as fr:
            lines = fr.readlines()
            with open(outfile, 'w') as fw:
                for line in lines[:-linestoremove]:
                    fw.write(line)
    except:
        print("Oops! error in remove_endlines ", infile,linestoremove,outfile)
