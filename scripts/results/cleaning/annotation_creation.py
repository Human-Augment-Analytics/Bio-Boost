# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Imports
import pandas as pd
import subprocess
import cv2

# Command Variables
cmd1 = "rclone ls "
cmd2="rclone copy "
cmd3="rm -r "
path='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/PatrickTesting/MaleFemale/'
trialpaths='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
trackfile='/MasterAnalysisFiles/AllTrackedFish.csv'
Annotationpath='CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/'
videos='/Videos/'
here=' ./'
classes=['Male', 'Female']
#get names of tracks needed.

# Metadata Extraction for Females
femalelist=subprocess.check_output(cmd1+path+classes[1], shell=True).decode("utf-8").split(' ')

# Cleaning subprocess output
fdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex'])
for i in femalelist:
    try:
        int(i)
    except:
        if i!='':
            fdf.loc[len(fdf.index)]=i.split('\n')[0].strip('.mp4').split('__')+[classes[1]]

# Metadata Extraction for Males
malelist=subprocess.check_output(cmd1+path+classes[0], shell=True).decode("utf-8").split(' ')

# Cleaning subprocess output
mdf=pd.DataFrame(columns=['trial', 'base_name', 'track_id', 'sex'])
for i in malelist:
    try:
        int(i)
    except:
        if i!='':
            mdf.loc[len(mdf.index)]=i.split('\n')[0].strip('.mp4').split('__')+[classes[0]]

# Concat Results
track_annotations = pd.concat([fdf, mdf], axis=0)

# Find Length
# Tracks pulled from 30 trials 
len(track_annotations.trial.unique())

# Iterate over unique trials
tracknum=0
for i in track_annotations.trial.unique():
    sdf=track_annotations[track_annotations.trial==i]
    videolist=sdf.base_name.unique()
    print('starting trial:  '+str(i))

    # Copy necessary files for the trial
    subprocess.run(cmd2+trialpaths+i+trackfile+here, shell=True)
    tdf = pd.read_csv ('./AllTrackedFish.csv')

    # Process each video in the trial
    for j in videolist:
        ssdf=sdf[sdf.base_name==j]
        tracks=[int(k) for k in ssdf.track_id]
        tdf=tdf[tdf.base_name==j]
        tdf=tdf.loc[tdf.track_id.isin(tracks)]
        print('Downloading Video '+str(j)+'of '+str(videolist[-1]))
        subprocess.run(cmd2+trialpaths+i+videos+j+'.mp4'+here, shell=True)

        # Process each track in the video
        for t in tracks:
            tracknum+=1
            ttdf=tdf[tdf.track_id==t]
            line=list(ssdf[ssdf.track_id==str(t)].iloc[0])
            cap = cv2.VideoCapture('./'+j+'.mp4')
            cap.set(cv2.CAP_PROP_POS_FRAMES, ttdf.frame.min())
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            current_frame = ttdf.frame.min()
            count=0
            for c,  current_track in ttdf.iterrows():
                ret, frame = cap.read()
                while current_track.frame != current_frame:
                    ret, frame = cap.read()
                    current_frame += 1
                outFile = './'+line[3]+'/'+line[0]+ '__' + line[1] + '__' + line[2] + '__'+str(count)+'.jpg'
                
                delta_xy=(int((1/2)*max(int(current_track.w)+1, int(current_track.h)+1)))+10
                frame = frame[int(max(0, current_track.yc - delta_xy)):int(min(current_track.yc + delta_xy,height)) , int(max(0, current_track.xc - delta_xy)):int(min(current_track.xc + delta_xy, width))]
                frame=cv2.resize(frame, (100, 100))
                cv2.imwrite(outFile, frame) 
                current_frame += 1
                count+=1
        print('images created!!')
        print('moving images to cloud')
        subprocess.run('rclone  copy ./Female/ CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/BreeTesting/Female', shell=True)
        subprocess.run('rclone  copy ./Male/ CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/BreeTesting/Male', shell=True)
        print('clearing directory for next trial')
        subprocess.run(cmd3+'./Female/*', shell=True)
        subprocess.run(cmd3+'./Male/*', shell=True)
        subprocess.run(cmd3+'./'+j+'.mp4', shell=True)
    
    # Cleanup after each trial
    subprocess.run(cmd3+'./AllTrackedFish.csv', shell=True)

# Final steps after all trials
print('All trials completed')
print('moving images back to working directory')
subprocess.run('rclone  copy CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/BreeTesting/Female ./Female/', shell=True)
subprocess.run('rclone  copy CichlidPiData:BioSci-McGrath/Apps/CichlidPiData/__AnnotatedData/BreeTesting/Male ./Male/ ', shell=True)

# Notify of Completion
print('All commands terminated')