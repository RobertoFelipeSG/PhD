import numpy as np
import mne
import matplotlib

from Preprocessing_Burst import *

### PILOTS
# path01 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Estelle.vhdr'
# Preprocessing_Burst('Estelle', path01)
# path02 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Jacob.vhdr'
# Preprocessing_Burst('Jacob', path02)
# path03 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Michele.vhdr'
# Preprocessing_Burst('Michele', path03)
# path04 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Fab.vhdr'
# Preprocessing_Burst('Fab', path04)
# path05 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Pie.vhdr'
# Preprocessing_Burst('Pie', path05)
# path06 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Gas.vhdr'
# Preprocessing_Burst('Gas', path06)
# path07 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Jul.vhdr'
# Preprocessing_Burst('Jul', path07)
# path08 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Es.vhdr'
# Preprocessing_Burst('Es', path08)
# path09 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Mih.vhdr'
# Preprocessing_Burst('Mih', path09)
# path10 = '../../../Documents/EEG_Data/Exp3_Pilot_Task_Florian.vhdr'
# Preprocessing_Burst('Fl1', path10)
# path11 = '../../../Documents/EEG_Data/Exp3_Pilot_TASK2_Florian.vhdr'
# Preprocessing_Burst('Fl2', path11)

### SUBJECTS
# path01 = '../../../Documents/Exp3_TASK/P01_TASK.vhdr'
# Preprocessing_Burst('P01_Bursts', path01)
# path02 = '../../../Documents/Exp3_TASK/P02_TASK.vhdr'
# Preprocessing_Burst('P02_Bursts', path02)
# path03 = '../../../Documents/Exp3_TASK/P03_TASK.vhdr'
# Preprocessing_Burst('P03_Bursts', path03)
# path04 = '../../../Documents/Exp3_TASK/P04_TASK.vhdr'
# Preprocessing_Burst('P04_Bursts', path04)
# path05 = '../../../Documents/Exp3_TASK/P05_TASK.vhdr'
# Preprocessing_Burst('P05_Bursts', path05)
# path06 = '../../../Documents/Exp3_TASK/P06_TASK.vhdr'
# Preprocessing_Burst('P06_Bursts', path06)
# path07 = '../../../Documents/Exp3_TASK/P07_TASK.vhdr'
# Preprocessing_Burst('P07_Bursts', path07)
# path08 = '../../../Documents/Exp3_TASK/P08_TASK.vhdr'
# Preprocessing_Burst('P08_Bursts', path08)
# path09 = '../../../Documents/Exp3_TASK/P09_TASK.vhdr'
# Preprocessing_Burst('P09_Bursts', path09)
# path10 = '../../../Documents/Exp3_TASK/P10_TASK.vhdr'
# Preprocessing_Burst('P10_Bursts', path10)
# path11 = '../../../Documents/Exp3_TASK/P11_TASK.vhdr'
# Preprocessing_Burst('P11_Bursts', path11)
# path12 = '../../../Documents/Exp3_TASK/P12_TASK.vhdr'
# Preprocessing_Burst('P12_Bursts', path12)
path13 = '../../../Documents/Exp3_TASK/P13_TASK.vhdr'
Preprocessing_Burst('P13_Bursts', path13)
# path14 = '../../../Documents/Exp3_TASK/P14_TASK.vhdr'
# Preprocessing_Burst('P14_Bursts', path14)
# path15 = '../../../Documents/Exp3_TASK/P15_TASK.vhdr'
# Preprocessing_Burst('P15_Bursts', path15)
# path16 = '../../../Documents/Exp3_TASK/P16_TASK.vhdr'
# Preprocessing_Burst('P16_Bursts', path16)
# path17 = '../../../Documents/Exp3_TASK/P17_TASK.vhdr'
# Preprocessing_Burst('P17_Bursts', path17)
# path18 = '../../../Documents/Exp3_TASK/P18_TASK.vhdr'
# Preprocessing_Burst('P18_Bursts', path18)
# path19 = '../../../Documents/Exp3_TASK/P19_TASK.vhdr'
# Preprocessing_Burst('P19_Bursts', path19)
# path20 = '../../../Documents/Exp3_TASK/P20_TASK.vhdr'
# Preprocessing_Burst('P20_Bursts', path20)
# path21 = '../../../Documents/Exp3_TASK/P21_TASK.vhdr'
# Preprocessing_Burst('P21_Bursts', path21)
# path22 = '../../../Documents/Exp3_TASK/P22_TASK.vhdr'
# Preprocessing_Burst('P22_Bursts', path22)
# path23 = '../../../Documents/Exp3_TASK/P23_TASK.vhdr'
# Preprocessing_Burst('P23_Bursts', path23)
# path24 = '../../../Documents/Exp3_TASK/P24_TASK.vhdr'
# Preprocessing_Burst('P24_Bursts', path24)
# path25 = '../../../Documents/Exp3_TASK/P25_TASK.vhdr'
# Preprocessing_Burst('P25_Bursts', path25)
# path26 = '../../../Documents/Exp3_TASK/P26_TASK.vhdr'
# Preprocessing_Burst('P26_Bursts', path26)
# path27 = '../../../Documents/Exp3_TASK/P27_TASK.vhdr'
# Preprocessing_Burst('P27_Bursts', path27)
# path28 = '../../../Documents/Exp3_TASK/P28_TASK.vhdr'
# Preprocessing_Burst('P28_Bursts', path28)
# path29 = '../../../Documents/Exp3_TASK/P29_TASK.vhdr'
# Preprocessing_Burst('P29_Bursts', path29)
# path30 = '../../../Documents/Exp3_TASK/P30_TASK.vhdr'
# Preprocessing_Burst('P30_Bursts', path30)
