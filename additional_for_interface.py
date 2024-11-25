# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:32:56 2024

@author: Natalia
"""

import re
import numpy as np
import tqdm



#==============================================================================
# Defino el promedio movil. El especial hace un promedio movil logaritmico.
#==============================================================================  
def moving_average(a, n=3,Especial=False) :
    if Especial==False:
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    if Especial==True:
        espacios= int(np.floor(np.log10(len(a))))
        ret= np.nancumsum(a, dtype=float)
        ret0=ret
        ret0[n:] = ret0[n:] - ret0[:-n]
        ret0=ret0[n - 1:11] / n
        for i in range(espacios):
            ret1=np.nancumsum(a, dtype=float)
            n0=10**(i+1)+1
            n1=n0+10**(i+2)
            if n1>len(a):
                n1=len(a)
            ret1[n0:]= ret1[n0:] - ret1[:-n0]
            # ret1=ret1[n0 + int((10**(i+1)+10**(i))/2)-1:n1-1] / n0
            ret1=ret1[n0+int(n0/2)-1:n1-1] / n0
            ret0=np.append(ret0,ret1)
        return ret0
    else:
        print('Especifique el modo Especial como True o False')
        
def Line_time (total_num_lines, total_num_columns, tp, tr=0):
    '''
    
    Parameters
    ----------
    total_num_lines : int

    total_num_columns : int

    tp : float
        pixel dwell time.
        
    tr : float, optional
        Return time in a real microscope. 
        If you are working with simulations tr is 0 by default. In a simulation the line time is tp*total_num_columns
        If you are working with microscope adquired data:
            -) tr would be 0 if tp already has incorporated the microscope pixel return time.
            -) tr would not be 0 if is just the pixel dwell time does not consider the microscope pixel return time.
    
    Returns
    -------
    1d-array: .

    '''
    sampling_frequency = 1/(tp*total_num_columns+tr)
    Time=[]
    
    for i in range (1,total_num_lines+1):
                     Time.append(i*1/sampling_frequency)
                     
    return Time


def Kimogram(Lines, pixel):
    '''
    
    Parameters
    ----------
    Lines : int16 1D-array.
        Lines is an 1D-array of consecutives lines.
    pixel : int
        number of pixel in a line.

    Returns
    -------
    kimogram : int16 2D-array
        Kimogram of lines.
        Each column is a pixel
        Each row is a line

    '''
    
    if type(len(Lines)/pixel) == float:   ## avoid possible uncompleted lines
        last_line = int(len(Lines)/pixel)*pixel
        Lines = Lines[0:last_line]                      
        
    kimogram = (Lines).reshape(int(len(Lines)/pixel),pixel)                                 
    
    return kimogram

def line_pCF_analysis(Kimogram, C0, C1, tp, reverse_PCF, 
                      return_time=0, Tiempo_imagen=0, logtime=False, Movil_log=0):

    '''
    Parameters
    ----------
    Kimogram of intensity: ndarray
        Tipically 100k rows and 256 columns.
    
    C0 : int
        First column to be correlated.
    
    C1 : int
        Second column to be correlated.

    tp: float
        pixel dwell time.
        
    return_time : float, optional
        Line time return. The default is 0.
        If pixel dwell time already includes the return pixel time of the microscope,
        then the return_time parameter can be ignored. 
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : 1D-array
        Correlation between columns C0 and C1
        
    Tau : 1D-array
        Correlation time
    
    '''
    
    # First column to be correlated
    C0=Kimogram [ : ,C0]
    # Second column to be correlated
    C1=Kimogram [ : ,C1]    
    
    if reverse_PCF:
        C0, C1 = C1, C0

    ######################################
    
    ## based on https://www.cgohlke.com/ipcf/ --> linear correlation
    
    """Return linear correlation of two arrays using DFT."""
    size = C0.size
    
    # subtract mean and pad with zeros to twice the size
    C0_mean = C0.mean()
    C1_mean = C1.mean()
    C0 = np.pad(C0-C0_mean, C0.size//2, mode='constant')
    C1 = np.pad(C1-C1_mean, C1.size//2, mode='constant')

    # forward DFT
    C0 = np.fft.rfft(C0)
    C1 = np.fft.rfft(C1)
    # multiply by complex conjugate
    G = C0.conj() * C1
    # reverse DFT
    G = np.fft.irfft(G)
    # positive delays only
    G = G[:size // 2]
        
    # normalize with the averages of a and b
    G /= size * C0_mean * C1_mean
    
    ######################################
    
    Tau = np.asarray(Line_time(Kimogram.shape[0], Kimogram.shape[1], tp))
    
    
    Tau = Tau[:len(Tau)//2] #$### por que lo dividiria a la mitad???


    #Apply MAV
    if Movil_log==0:

        if logtime==False:
            return np.array(G), np.array (Tau)
        if logtime==True:
            Tau=np.log10 (Tau)
            return np.array(G), np.array(Tau)
    else:

        G=moving_average(G, Movil_log , Especial=True)
    
        if logtime==False:
            Tau=moving_average(Tau, Movil_log, Especial=True )
        if logtime==True:
            Tau=np.log10(moving_average(Tau, Movil_log, Especial=True ))
    
    return np.array(G), np.array(Tau)

def pCF(Kimogram , tp, dr=0, reverse_PCF=False, return_time=0, logtime=False, Movil_log=0):
    '''
    Parameters
    ----------
    Kimogram : ndarray tipically (100k rows, 256 columns)
        Kimogram = Kimogram or B Kimogram  
        
    dr : int
        pCF distance.
    
    tp : TYPE
        DESCRIPTION.

    return_time : float, optional
        line time return. The default is 0.
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.
    
    ACF norm: Bool, optional
        Normalizes the amplitud of correlation by the G(0)

    Returns
    -------
    G : ndarray 
        Matrix of pCF analisys between at distance.
        -) Columns are the pixel position
        -) Rows are the correlation analysis
        
    Tau : ndarray
          Correlation time
          -) Columns are the pixel position
          -) Rows are the delay time
    '''
    

    Size = len(Kimogram[0]) #cantidad de píxeles en una linea

    #calculo todas las correlaciones
    G=[]
    T=[]
    for i in tqdm.trange(Size-dr):
        result = line_pCF_analysis(Kimogram ,i ,i+dr, tp, reverse_PCF,
                                         return_time, logtime, Movil_log=Movil_log)
        
        G.append(result[0])
        T.append(result[1])

    return np.array(G).transpose(), np.array(T).transpose()


def line_crosspCF_analysis(Kimogram, Kimogram2, C0, C1, linetime, reverse_PCF, 
                      return_time=0, Tiempo_imagen=0, logtime=False, Movil_log=0):

    '''
    Parameters
    ----------
    Kimogram of intensity: ndarray
        Tipically 100k rows and 256 columns.
    
    C0 : int
        First column to be correlated.
    
    C1 : int
        Second column to be correlated.

    linetime: float
        time taken to adquire each line.
        
    return_time : float, optional
        Line time return. The default is 0.
        If pixel dwell time already includes the return pixel time of the microscope,
        then the return_time parameter can be ignored. 
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : 1D-array
        Correlation between columns C0 and C1
        
    Tau : 1D-array
        Correlation time
    
    '''
    
    # First column to be correlated
    C0=Kimogram [ : ,C0]
    # Second column to be correlated
    C1=Kimogram2 [ : ,C1]    
    
    if reverse_PCF:
        C0, C1 = C1, C0

    ######################################
    
    ## based on https://www.cgohlke.com/ipcf/ --> linear correlation
    
    """Return linear correlation of two arrays using DFT."""
    size = C0.size
    
    # subtract mean and pad with zeros to twice the size
    C0_mean = C0.mean()
    C1_mean = C1.mean()
    C0 = np.pad(C0-C0_mean, C0.size//2, mode='constant')
    C1 = np.pad(C1-C1_mean, C1.size//2, mode='constant')

    # forward DFT
    C0 = np.fft.rfft(C0)
    C1 = np.fft.rfft(C1)
    # multiply by complex conjugate
    G = C0.conj() * C1
    # reverse DFT
    G = np.fft.irfft(G)
    # positive delays only
    G = G[:size // 2]
        
    # normalize with the averages of a and b
    G /= size * C0_mean * C1_mean
    
    ######################################
    Tau=[]
    
    for i in range (1,Kimogram.shape[0]+1):
                     Tau.append(i*linetime)
    #Tau = Tau[:len(Tau)//2]


    #Apply MAV
    if Movil_log==0:

        if logtime==False:
            return np.array(G), np.array (Tau)
        if logtime==True:
            Tau=np.log10 (Tau)
            return np.array(G), np.array(Tau)
    else:

        G=moving_average(G, Movil_log , Especial=True)
    
        if logtime==False:
            Tau=moving_average(Tau, Movil_log, Especial=True )
        if logtime==True:
            Tau=np.log10(moving_average(Tau, Movil_log, Especial=True ))
    
    return np.array(G), np.array(Tau)


def crosspCF(Kimogram, Kimogram2, linetime, dr=0, reverse_PCF=False, return_time=0, logtime=False, Movil_log=0):
    '''
    Parameters
    ----------
    Kimogram : ndarray tipically (100k rows, 256 columns)
        Kimogram = Kimogram or B Kimogram  
        
    dr : int
        pCF distance.
    
    tp : TYPE
        DESCRIPTION.

    return_time : float, optional
        line time return. The default is 0.
        
    logtime : Bool, optional
        Allows logarithmic separate values for Tau. The default is False.
    
    Movil_log : int, optional
         The default is 0. If Movil_log is not zero, then a logarithmic moving average its done.

    Returns
    -------
    G : ndarray 
        Matrix of pCF analisys between at distance.
        -) Columns are the pixel position
        -) Rows are the correlation analysis
        
    Tau : ndarray
          Correlation time
          -) Columns are the pixel position
          -) Rows are the delay time
    '''
    
        
    # if Movil_log==0:
        # print('MAV = 0')
    # else:
        # print('MAV = %s' % (Movil_log))
    Size = len(Kimogram[0]) #cantidad de píxeles en una linea
    print(f"Starting crosspCF with Size: {Size}, dr: {dr}, line time: {linetime}")  # Debug print

    #calculo todas las correlaciones
    G=[]
    T=[]
    for i in tqdm.trange(Size-dr):
        #print(f"Processing index: {i}")  # Debug print
        pCF_analysis = line_crosspCF_analysis(Kimogram ,Kimogram2, i ,i+dr, linetime, reverse_PCF,
                                         return_time, logtime, Movil_log=Movil_log)

        G.append(pCF_analysis[0])
        T.append(pCF_analysis[1])

    return np.array(G).transpose(), np.array(T).transpose()