import numpy as np
import mne
import matplotlib

from Preprocessing import *

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mne.viz import iter_topography
from mne.beamformer import make_dics, apply_dics_csd

from time import time


#path01 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/01EC_taskBsl.vhdr'
#Preprocessing('P01', path01)
#path02 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/02HR_taskBsl.vhdr'
#Preprocessing('P02', path02)
#path03 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/03FP_taskBsl.vhdr'
#Preprocessing('P03', path03)
#path04 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/04TH_taskBsl.vhdr'
#Preprocessing('P04', path04)
#path05 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/05FD_taskBsl2.vhdr'
#Preprocessing('P05_Bsl', path05)
#path06 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/06FC_taskBsl.vhdr'
#Preprocessing('P06_Bsl', path06)
#path07 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/07HS_taskBsl.vhdr'
#Preprocessing('P07_Bsl', path07)
#path08 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/08GP_taskBsl.vhdr'
#Preprocessing('P08_Bsl', path08)
#path09 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/09ML_taskBsl.vhdr'
#Preprocessing('P09_Bsl', path09)
#path10 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/10JN_taskBsl.vhdr'
#Preprocessing('P10_Bsl', path10)
#path11 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/11PG_taskBsl.vhdr'
#Preprocessing('P11_Bsl', path11)
#path12 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/12VB_taskBsl.vhdr'
#Preprocessing('P12_Bsl', path12)
#path13 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/13IS_taskBsl.vhdr'
#Preprocessing('P13_Bsl', path13)
#path14 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/14AM_taskBsl.vhdr'
#Preprocessing('P14_Bsl', path14)
#path15 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/15LP_taskBsl.vhdr'
#Preprocessing('P15_Bsl', path15)
#path16 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/16HD_taskBsl.vhdr'
#Preprocessing('P16_Bsl', path16)
#path17 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/17AM_taskBsl.vhdr'
#Preprocessing('P17_Bsl', path17)
#path18 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/18IC_taskBsl.vhdr'
#Preprocessing('P18_Bsl', path18)
#path19 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/19VB_taskBsl.vhdr'
#Preprocessing('P19_Bsl', path19)
#path20 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/20AM_taskBsl.vhdr'
#Preprocessing('P20_Bsl', path20)
#path21 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/21II_taskBsl.vhdr'
#Preprocessing('P21_Bsl', path21)
#path22 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/22AP_taskBsl.vhdr'
#Preprocessing('P22_Bsl', path22)
#path23 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/23JT_taskBsl.vhdr'
#Preprocessing('P23_Bsl', path23)
#path24 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/24SC_taskBsl.vhdr'
#Preprocessing('P24_Bsl', path24)
#path25 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/25LG_taskBsl.vhdr'
#Preprocessing('P25_Bsl', path25)
#path26 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/26FB_taskBsl.vhdr'
#Preprocessing('P26_Bsl', path26)
#path27 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/27EB_taskBsl.vhdr'
#Preprocessing('P27_Bsl', path27)
#path28 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/28DP_taskBsl.vhdr'
#Preprocessing('P28_Bsl', path28)
#path29 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/29FD_taskBsl.vhdr'
#Preprocessing('P29_Bsl', path29)
#path30 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/30VR_taskBsl.vhdr'
#Preprocessing('P30_Bsl', path30)
#path31 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/29FD_taskBsl.vhdr'
#Preprocessing('P31_Bsl', path31)
#path32 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/32AC_taskBsl.vhdr'
#Preprocessing('P32_Bsl', path32)
#path33 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/33EA_taskBsl.vhdr'
#Preprocessing('P33_Bsl', path33)
#path34 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/34DG_taskBsl.vhdr'
#Preprocessing('P34_Bsl', path34)
#path35 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/35GE_taskBsl.vhdr'
#Preprocessing('P35_Bsl', path35)
#path36 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/36MR_taskBsl.vhdr'
#Preprocessing('P36_Bsl', path36)
#path37 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/37GN_taskBsl.vhdr'
#Preprocessing('P37_Bsl', path37)
#path38 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/38DK_taskBsl.vhdr'
#Preprocessing('P38_Bsl', path38)
#path39 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/39MH_taskBsl.vhdr'
#Preprocessing('P39_Bsl', path39)
#path40 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/40UP_taskBsl.vhdr'
#Preprocessing('P40_Bsl', path40)
#path41 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/41MM_taskBsl.vhdr'
#Preprocessing('P41_Bsl', path41)
#path42 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/42PE_taskBsl.vhdr'
#Preprocessing('P42_Bsl', path42)
#path43 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/43CC_taskBsl.vhdr'
#Preprocessing('P43_Bsl', path43)
#path44 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/44SN_taskBsl.vhdr'
#Preprocessing('P44_Bsl', path44)
#path45 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/45MF_taskBsl.vhdr'
#Preprocessing('P45_Bsl', path45)
#path46 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/46LG_taskBsl.vhdr'
#Preprocessing('P46_Bsl', path46)
#path47 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/47CB_taskBsl.vhdr'
#Preprocessing('P47_Bsl', path47)
#path48 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/48HM_taskBsl.vhdr'
#Preprocessing('P48_Bsl', path48)
#path491 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/49MM_taskBsl.vhdr'
#Preprocessing('P49_Bsl', path491)
#path501 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/50JB_taskBsl.vhdr'
#Preprocessing('P50_Bsl', path501)
#path51 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/51XW_taskBsl.vhdr'
#Preprocessing('P51_Bsl', path51)
#path52 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/52CC_taskBsl.vhdr'
#Preprocessing('P52_Bsl', path52)
#path53 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/53AC_taskBsl.vhdr'
#Preprocessing('P53_Bsl', path53)
#path54 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/54VG_taskBsl.vhdr'
#Preprocessing('P54_Bsl', path54)
#path55 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/55YW_taskBsl.vhdr'
#Preprocessing('P55_Bsl', path55)
#path56 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/56GD_taskBsl.vhdr'
#Preprocessing('P56_Bsl', path56)
#path57 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/57CA_taskBsl.vhdr'
#Preprocessing('P57_Bsl', path57)
#path58 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/58AK_taskBsl.vhdr'
#Preprocessing('P58_Bsl', path58)
#path59 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/59AV_taskBsl.vhdr'
#Preprocessing('P59_Bsl', path59)
#path60 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/60CJ_taskBsl.vhdr'
#Preprocessing('P60_Bsl', path60)
#path61 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/61EC_taskBsl.vhdr'
#Preprocessing('P61_Bsl', path61)
# path62 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/62AF_taskBsl.vhdr'
# Preprocessing('P62_Bsl', path62)
# path63 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/63MF_taskBsl.vhdr'
# Preprocessing('P63_Bsl', path63)
# path64 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/64SC_taskBsl.vhdr'
# Preprocessing('P64_Bsl', path64)
#path65 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/65JG_taskBsl.vhdr'
#Preprocessing('P65_Bsl', path65)
#path66 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/66IC_taskBsl.vhdr'
#Preprocessing('P66_Bsl', path66)
#path67 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/67GH_taskBsl.vhdr'
#Preprocessing('P67_Bsl', path67)
#path68 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/68AS_taskBsl.vhdr'
#Preprocessing('P68_Bsl', path68)
# path69 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/69AS_taskBsl.vhdr'
# Preprocessing('P69_Bsl', path69)
#path70 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/70JP_taskBsl.vhdr'
#Preprocessing('P70_Bsl', path70)
# path71 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/71AE_taskBsl.vhdr'
# Preprocessing('P71_Bsl', path71)
# path72 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/72AI_taskBsl.vhdr'
# Preprocessing('P72_Bsl', path72)
# path73 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/73AG_taskBsl.vhdr'
# Preprocessing('P73_Bsl', path73)
# path74 = '/home/robertofelipe_sg/Documents/EEG_Data/74LL_taskBsl.vhdr'
# Preprocessing('P74_Bsl', path74)
# path75 = '/home/robertofelipe_sg/Documents/EEG_Data/75EB_taskBsl.vhdr'
# Preprocessing('P75_Bsl', path75)
# path76 = '/home/robertofelipe_sg/Documents/EEG_Data/76AK_taskBsl.vhdr'
# Preprocessing('P76_Bsl', path76)
# path77 = '/home/robertofelipe_sg/Documents/EEG_Data/77MB_taskBsl.vhdr'
# Preprocessing('P77_Bsl', path77)
# path78 = '/home/robertofelipe_sg/Documents/EEG_Data/78AF_taskBsl.vhdr'
# Preprocessing('P78_Bsl', path78)
# path79 = '/home/robertofelipe_sg/Documents/EEG_Data/79CR_taskBsl.vhdr'
# Preprocessing('P79_Bsl', path79)
# path80 = '/home/robertofelipe_sg/Documents/EEG_Data/80SA_taskBsl.vhdr'
# Preprocessing('P80_Bsl', path80)
# path81 = '/home/robertofelipe_sg/Documents/EEG_Data/81AR_taskBsl.vhdr'
# Preprocessing('P81_Bsl', path81)

### POST 10 ###
#path01 = '/home/robertofelipe_sg/Desktop/eeg/01EC_taskPost1.vhdr'
#Preprocessing('P01', path01)
#path02 = '/home/robertofelipe_sg/Desktop/eeg/02HR_taskPost1.vhdr'
#Preprocessing('P02', path02)
#path03 = '/home/robertofelipe_sg/Desktop/eeg/03FR_taskPost10.vhdr'
#Preprocessing('P03', path03)
#path04 = '/home/robertofelipe_sg/Desktop/eeg/04TH_taskPost10.vhdr'
#Preprocessing('P04', path04)
#path05 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/05FD_taskPost10.vhdr'
#Preprocessing('P05_Post10', path05)
#path06 = '/home/robertofelipe_sg/Desktop/eeg/06FC_taskPost10.vhdr'
#Preprocessing('P06', path06)
#path07 = '/home/robertofelipe_sg/Desktop/eeg/07HS_taskPost10.vhdr'
#Preprocessing('P07', path07)
#path08 = '/home/robertofelipe_sg/Desktop/eeg/08GP_taskPost10.vhdr'
#Preprocessing('P08', path08)
#path09 = '/home/robertofelipe_sg/Desktop/eeg/09ML_taskPost10.vhdr'
#Preprocessing('P09', path09)
#path10 = '/home/robertofelipe_sg/Desktop/eeg/10JN_taskPost10.vhdr'
#Preprocessing('P10', path10)
#path11 = '/home/robertofelipe_sg/Desktop/eeg/11PG_taskPost10.vhdr'
#Preprocessing('P11', path11)
#path12 = '/home/robertofelipe_sg/Desktop/eeg/12VB_taskPost10.vhdr'
#Preprocessing('P12', path12)
#path13 = '/home/robertofelipe_sg/Desktop/eeg/13IS_taskPost10.vhdr'
#Preprocessing('P13', path13)
#path14 = '/home/robertofelipe_sg/Desktop/eeg/14AM_taskPost10.vhdr'
#Preprocessing('P14', path14)
#path15 = '/home/robertofelipe_sg/Desktop/eeg/15LP_taskPost10.vhdr'
#Preprocessing('P15', path15)
#path16 = '/home/robertofelipe_sg/Desktop/eeg/16HD_taskPost10.vhdr'
#Preprocessing('P16', path16)
#path17 = '/home/robertofelipe_sg/Desktop/eeg/17AM_taskPost10.vhdr'
#Preprocessing('P17', path17)
#path18 = '/home/robertofelipe_sg/Desktop/eeg/18IC_taskPost10.vhdr'
#Preprocessing('P18', path18)
#path19 = '/home/robertofelipe_sg/Desktop/eeg/19VB_taskPost10.vhdr'
#Preprocessing('P19', path19)
#path20 = '/home/robertofelipe_sg/Desktop/eeg/20AM_taskPost10.vhdr'
#Preprocessing('P20', path20)
#path21 = '/home/robertofelipe_sg/Desktop/eeg/21II_taskPost10.vhdr'
#Preprocessing('P21', path21)
#path22 = '/home/robertofelipe_sg/Desktop/eeg/22AP_taskPost10.vhdr'
#Preprocessing('P22', path22)
#path23 = '/home/robertofelipe_sg/Desktop/eeg/23JT_taskPost10.vhdr'
#Preprocessing('P23', path23)
#path24 = '/home/robertofelipe_sg/Desktop/eeg/24SC_taskPost10.vhdr'
#Preprocessing('P24', path24)
#path25 = '/home/robertofelipe_sg/Desktop/eeg/25LG_taskPost10.vhdr'
#Preprocessing('P25', path25)
#path26 = '/home/robertofelipe_sg/Desktop/eeg/26FB_taskPost10.vhdr'
#Preprocessing('P26', path26)
#path27 = '/home/robertofelipe_sg/Desktop/eeg/27EB_taskPost10.vhdr'
#Preprocessing('P27', path27)
#path28 = '/home/robertofelipe_sg/Desktop/eeg/28DP_taskPost10.vhdr'
#Preprocessing('P28', path28)
#path29 = '/home/robertofelipe_sg/Desktop/eeg/29FD_taskPost10.vhdr'
#Preprocessing('P29', path29)
#path30 = '/home/robertofelipe_sg/Desktop/eeg/30VR_taskPost10.vhdr'
#Preprocessing('P30', path30)
#path32 = '/home/robertofelipe_sg/Desktop/eeg/32AC_taskPost10.vhdr'
#Preprocessing('P32', path32)
#path33 = '/home/robertofelipe_sg/Desktop/eeg/33EA_taskPost10.vhdr'
#Preprocessing('P33', path33)
#path34 = '/home/robertofelipe_sg/Desktop/eeg/34DG_taskPost10.vhdr'
#Preprocessing('P34', path34)
#path35 = '/home/robertofelipe_sg/Desktop/eeg/35GE_taskPost10.vhdr'
#Preprocessing('P35', path35)
#path36 = '/home/robertofelipe_sg/Desktop/eeg/36MR_taskPost10.vhdr'
#Preprocessing('P36', path36)
#path37 = '/home/robertofelipe_sg/Desktop/eeg/37GN_taskPost10.vhdr'
#Preprocessing('P37', path37)
#path38 = '/home/robertofelipe_sg/Desktop/eeg/38DK_taskPost10.vhdr'
#Preprocessing('P38', path38)
#path39 = '/home/robertofelipe_sg/Desktop/eeg/39MH_taskPost10.vhdr'
#Preprocessing('P39', path39)
#path40 = '/home/robertofelipe_sg/Desktop/eeg/40UP_taskPost10.vhdr'
#Preprocessing('P40', path40)
#path41 = '/home/robertofelipe_sg/Desktop/eeg/41MM_taskPost10.vhdr'
#Preprocessing('P41', path41)
#path42 = '/home/robertofelipe_sg/Desktop/eeg/42PE_taskPost10.vhdr'
#Preprocessing('P42', path42)
#path43 = '/home/robertofelipe_sg/Desktop/eeg/43CC_taskPost10.vhdr'
#Preprocessing('P43', path43)
#path44 = '/home/robertofelipe_sg/Desktop/eeg/44SN_taskPost10.vhdr'
#Preprocessing('P44', path44)
#path451 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/45MF_taskPost10.vhdr'
#Preprocessing('P45_Post10', path451)
#path461 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/46LG_taskPost10.vhdr'
#Preprocessing('P46_Post10', path461)
#path471 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/47CB_taskPost10.vhdr'
#Preprocessing('P47_Post10', path471)
#path481 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/48HM_taskPost10.vhdr'
#Preprocessing('P48_Post10', path481)
#path492 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/49MM_taskPost10.vhdr'
#Preprocessing('P49_Post10', path492)
#path502 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/50JB_taskPost10.vhdr'
#Preprocessing('P50_Post10', path502)
#path511 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/51XW_taskPost10.vhdr'
#Preprocessing('P51_Post10', path511)
#path521 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/52CC_taskPost10.vhdr'
#Preprocessing('P52_Post10', path521)
#path531 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/53AC_taskPost10.vhdr'
#Preprocessing('P53_Post10', path531)
#path541 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/54VG_taskPost10.vhdr'
#Preprocessing('P54_Post10', path541)
#path551 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/55YW_taskPost10.vhdr'
#Preprocessing('P55_Post10', path551)
#path561 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/56GD_taskPost10.vhdr'
#Preprocessing('P56_Post10', path561)
#path571 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/57CA_taskPost10.vhdr'
#Preprocessing('P57_Post10', path571)
#path581 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/58AK_taskPost10.vhdr'
#Preprocessing('P58_Post10', path581)
#path591 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/59AV_taskPost10.vhdr'
#Preprocessing('P59_Post10', path591)
#path601 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/60CJ_taskPost10.vhdr'
#Preprocessing('P60_Post10', path601)
#path611 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/61EC_taskPost10.vhdr'
#Preprocessing('P61_Post10', path611)
#path621 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/62AF_taskPost10.vhdr'
#Preprocessing('P62_Post10', path621)
# path631 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/63MF_taskPost10.vhdr'
# Preprocessing('P63_Post10', path631)
# path641 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/64SC_taskPost10.vhdr'
# Preprocessing('P64_Post10', path641)
#Path651 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/65JG_taskPost10.vhdr'
#Preprocessing('P65_Post10', path651)
#path661 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/66IC_taskPost10.vhdr'
#Preprocessing('P66_Post10', path661)
#path671 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/67GH_taskPost10.vhdr'
#Preprocessing('P67_Post10', path671)
#path681 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/68AS_taskPost10.vhdr'
#Preprocessing('P68_Post10', path681)
# path691 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/69AS_taskPost10.vhdr'
# Preprocessing('P69_Post10', path691)
#path701 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/70JP_taskPost10.vhdr'
#Preprocessing('P70_Post10', path701)
# path711 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/71AE_taskPost10.vhdr'
# Preprocessing('P71_Post10', path711)
# path721 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/72AI_taskPost10.vhdr'
# Preprocessing('P72_Post10', path721)
# path731 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/73AG_taskPost10.vhdr'
# Preprocessing('P73_Post10', path731)
# path74 = '/home/robertofelipe_sg/Documents/EEG_Data/74LL_taskPost10.vhdr'
# Preprocessing('P74_Post10', path74)
# path75 = '/home/robertofelipe_sg/Documents/EEG_Data/75EB_taskPost10.vhdr'
# Preprocessing('P75_Post10', path75)
# path76 = '/home/robertofelipe_sg/Documents/EEG_Data/76AK_taskPost10.vhdr'
# Preprocessing('P76_Post10', path76)
# path77 = '/home/robertofelipe_sg/Documents/EEG_Data/77MB_taskPost10.vhdr'
# Preprocessing('P77_Post10', path77)
# path78 = '/home/robertofelipe_sg/Documents/EEG_Data/78AF_taskPost10.vhdr'
# Preprocessing('P78_Post10', path78)
# path79 = '/home/robertofelipe_sg/Documents/EEG_Data/79CR_taskPost10.vhdr'
# Preprocessing('P79_Post10', path79)
# path80 = '/home/robertofelipe_sg/Documents/EEG_Data/80SA_taskPost10.vhdr'
# Preprocessing('P80_Post10', path80)
# path81 = '/home/robertofelipe_sg/Documents/EEG_Data/81AR_taskPost10.vhdr'
# Preprocessing('P81_Post10', path81)

### POST 30 ###
#path01 = '/home/robertofelipe_sg/Desktop/eeg/01EC_taskPost2.vhdr'
#Preprocessing('P01', path01)
#path02 = '/home/robertofelipe_sg/Desktop/eeg/02HR_taskPost30.vhdr'
#Preprocessing('P02', path02)
#path03 = '/home/robertofelipe_sg/Desktop/eeg/03FR_taskPost30b.vhdr'
#Preprocessing('P03', path03)
#path04 = '/home/robertofelipe_sg/Desktop/eeg/04TH_taskPost30.vhdr'
#Preprocessing('P04', path04)
#path053 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/05FD_taskPost30.vhdr'
#Preprocessing('P05_Post30', path053)
#path06 = '/home/robertofelipe_sg/Desktop/eeg/06FC_taskPost30.vhdr'
#Preprocessing('P06', path06)
#path07 = '/home/robertofelipe_sg/Desktop/eeg/07HS_taskPost30.vhdr'
#Preprocessing('P07', path07)
#path08 = '/home/robertofelipe_sg/Desktop/eeg/08GP_taskPost30.vhdr'
#Preprocessing('P08', path08)
#path09 = '/home/robertofelipe_sg/Desktop/eeg/09ML_taskPost30_2.vhdr'
#Preprocessing('P09', path09)
#path10 = '/home/robertofelipe_sg/Desktop/eeg/10JN_taskPost30.vhdr'
#Preprocessing('P10', path10)
#path11 = '/home/robertofelipe_sg/Desktop/eeg/11PG_taskPost30b.vhdr'
#Preprocessing('P11', path11)
#path12 = '/home/robertofelipe_sg/Desktop/eeg/12VB_taskPost30.vhdr'
#Preprocessing('P12', path12)
#path13 = '/home/robertofelipe_sg/Desktop/eeg/13IS_taskPost30.vhdr'
#Preprocessing('P13', path13)
#path14 = '/home/robertofelipe_sg/Desktop/eeg/14AM_taskPost30.vhdr'
#Preprocessing('P14', path14)
#path15 = '/home/robertofelipe_sg/Desktop/eeg/15LP_taskPost30.vhdr'
#Preprocessing('P15', path15)
#path16 = '/home/robertofelipe_sg/Desktop/eeg/16HD_taskPost30.vhdr'
#Preprocessing('P16', path16)
#path17 = '/home/robertofelipe_sg/Desktop/eeg/17AM_taskPost30.vhdr'
#Preprocessing('P17', path17)
#path18 = '/home/robertofelipe_sg/Desktop/eeg/18IC_taskPost30.vhdr'
#Preprocessing('P18', path18)
#path19 = '/home/robertofelipe_sg/Desktop/eeg/19VB_taskPost30.vhdr'
#Preprocessing('P19', path19)
#path20 = '/home/robertofelipe_sg/Desktop/eeg/20AM_taskPost30.vhdr'
#Preprocessing('P20', path20)
#path21 = '/home/robertofelipe_sg/Desktop/eeg/21II_taskPost30.vhdr'
#Preprocessing('P21', path21)
#path22 = '/home/robertofelipe_sg/Desktop/eeg/22AP_taskPost30.vhdr'
#Preprocessing('P22', path22)
#path23 = '/home/robertofelipe_sg/Desktop/eeg/23JT_taskPost30.vhdr'
#Preprocessing('P23', path23)
#path24 = '/home/robertofelipe_sg/Desktop/eeg/24SC_taskPost30.vhdr'
#Preprocessing('P24', path24)
#path25 = '/home/robertofelipe_sg/Desktop/eeg/25LG_taskPost30.vhdr'
#Preprocessing('P25', path25)
#path26 = '/home/robertofelipe_sg/Desktop/eeg/26FB_taskPost30.vhdr'
#Preprocessing('P26', path26)
#path27 = '/home/robertofelipe_sg/Desktop/eeg/27EB_taskPost30.vhdr'
#Preprocessing('P27', path27)
#path28 = '/home/robertofelipe_sg/Desktop/eeg/28DP_taskPost30.vhdr'
#Preprocessing('P28', path28)
#path29 = '/home/robertofelipe_sg/Desktop/eeg/29FD_taskPost30.vhdr'
#Preprocessing('P29', path29)
#path30 = '/home/robertofelipe_sg/Desktop/eeg/30VR_taskPost30.vhdr'
#Preprocessing('P30', path30)
#path32 = '/home/robertofelipe_sg/Desktop/eeg/32AC_taskPost30.vhdr'
#Preprocessing('P32', path32)
#path33 = '/home/robertofelipe_sg/Desktop/eeg/33EA_taskPost30.vhdr'
#Preprocessing('P33', path33)
#path34 = '/home/robertofelipe_sg/Desktop/eeg/34DG_taskPost30.vhdr'
#Preprocessing('P34', path34)
#path35 = '/home/robertofelipe_sg/Desktop/eeg/35GE_taskPost30.vhdr'
#Preprocessing('P35', path35)
#path36 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/36MR_taskPost30.vhdr'
#Preprocessing('P36_Post30', path36)
#path37 = '/home/robertofelipe_sg/Desktop/eeg/37GN_taskPost30.vhdr'
#Preprocessing('P37', path37)
#path38 = '/home/robertofelipe_sg/Desktop/eeg/38DK_taskPost30.vhdr'
#Preprocessing('P38', path38)
#path39 = '/home/robertofelipe_sg/Desktop/eeg/39MH_taskPost30.vhdr'
#Preprocessing('P39', path39)
#path40 = '/home/robertofelipe_sg/Desktop/eeg/40UP_taskPost30.vhdr'
#Preprocessing('P40', path40)
#path41 = '/home/robertofelipe_sg/Desktop/eeg/41MM_taskPost30.vhdr'
#Preprocessing('P41', path41)
#path42 = '/home/robertofelipe_sg/Desktop/eeg/42PE_taskPost30.vhdr'
#Preprocessing('P42', path42)
#path43 = '/home/robertofelipe_sg/Desktop/eeg/43CC_taskPost30.vhdr'
#Preprocessing('P43', path43)
#path44 = '/home/robertofelipe_sg/Desktop/eeg/44SN_taskPost30.vhdr'
#Preprocessing('P44', path44)
#path452 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/45MF_taskPost30.vhdr'
#Preprocessing('P45_Post30', path452)
#path462 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/46LG_taskPost30.vhdr'
#Preprocessing('P46_Post30', path462)
#path472 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/47CB_taskPost30.vhdr'
#Preprocessing('P47_Post30', path472)
#path482 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/48HM_taskPost30.vhdr'
#Preprocessing('P48_Post30', path482)
#path503 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/50JB_taskPost30.vhdr'
#Preprocessing('P50_Post30', path503)
#path512 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/51XW_taskPost30.vhdr'
#Preprocessing('P51_Post30', path512)
#path522 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/52CC_taskPost30.vhdr'
#Preprocessing('P52_Post30', path522)
#path532 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/53AC_taskPost30.vhdr'
#Preprocessing('P53_Post30', path532)
#path542 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/54VG_taskPost30.vhdr'
#Preprocessing('P54_Post30', path542)
#path552 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/55YW_taskPost30.vhdr'
#Preprocessing('P55_Post30', path552)
#path552 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/56GD_taskPost30.vhdr'
#Preprocessing('P56_Post30', path552)
#path572 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/57CA_taskPost30.vhdr'
#Preprocessing('P57_Post30', path572)
#path582 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/58AK_taskPost30.vhdr'
#Preprocessing('P58_Post30', path582)
#path592 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/59AV_taskPost30.vhdr'
#Preprocessing('P59_Post30', path592)
#path602 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/60CJ_taskPost30.vhdr'
#Preprocessing('P60_Post30', path602)
#path612 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/61EC_taskPost30.vhdr'
#Preprocessing('P61_Post30', path612)
# path622 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/62AF_taskPost30.vhdr'
# Preprocessing('P62_Post30', path622)
# path632 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/63MF_taskPost30.vhdr'
# Preprocessing('P63_Post30', path632)
# path642 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/64SC_taskPost30.vhdr'
# Preprocessing('P64_Post30', path642)
#path652 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/65JG_taskPost30.vhdr'
#Preprocessing('P65_Post30', path652)
#path662 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/66IC_taskPost30.vhdr'
#Preprocessing('P66_Post30', path662)
#path672 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/67GH_taskPost30.vhdr'
#Preprocessing('P67_Post30', path672)
#path682 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/68AS_taskPost30.vhdr'
#Preprocessing('P68_Post30', path682)
# path692 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/69AS_taskPost30.vhdr'
# Preprocessing('P69_Post30', path692)
#path702 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/70JP_taskPost30.vhdr'
#Preprocessing('P70_Post30', path702)
# path712 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/71AE_taskPost30.vhdr'
# Preprocessing('P71_Post30', path712)
# path722 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/72AI_taskPost30.vhdr'
# Preprocessing('P72_Post30', path722)
# path732 = '/home/robertofelipe_sg/Documents/MATLAB/EEG_Data/73AG_taskPost30.vhdr'
# Preprocessing('P73_Post30', path732)
# path74 = '/home/robertofelipe_sg/Documents/EEG_Data/74LL_taskPost30.vhdr'
# Preprocessing('P74_Post30', path74)
path75 = '/home/robertofelipe_sg/Documents/EEG_Data/75EB_taskPost30.vhdr'
Preprocessing('P75_Post30', path75)
path76 = '/home/robertofelipe_sg/Documents/EEG_Data/76AK_taskPost30.vhdr'
Preprocessing('P76_Post30', path76)
path77 = '/home/robertofelipe_sg/Documents/EEG_Data/77MB_taskPost30.vhdr'
Preprocessing('P77_Post30', path77)
path78 = '/home/robertofelipe_sg/Documents/EEG_Data/78AF_taskPost30.vhdr'
Preprocessing('P78_Post30', path78)
path79 = '/home/robertofelipe_sg/Documents/EEG_Data/79CR_taskPost30.vhdr'
Preprocessing('P79_Post30', path79)
path80 = '/home/robertofelipe_sg/Documents/EEG_Data/80SA_taskPost30.vhdr'
Preprocessing('P80_Post30', path80)
path81 = '/home/robertofelipe_sg/Documents/EEG_Data/81AR_taskPost30.vhdr'
Preprocessing('P81_Post30', path81)

