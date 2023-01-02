# Rainfall DSD Modelling
# K. Ekerete, 10-Feb-2018

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import scipy.io as sio
import math
import os
import datetime
#import time
import functions
import neuralNet3Layer
from netCDF4 import Dataset
from sklearn.mixture import GaussianMixture


# Open data files
stAnswer = input("Enter Start Date (yyyymmdd): ")
enAnswer = input("Enter End Date (yyyymmdd): ")

stY = stAnswer[:4]; stM = stAnswer[4:6]; stD = stAnswer[6:]
enY = enAnswer[:4]; enM = enAnswer[4:6]; enD = enAnswer[6:]

startDate = stY + '-' + stM + '-' + stD
endDate = enY + '-' + enM + '-' + enD

startDate = datetime.date(int(stY),int(stM),int(stD))
endDate = datetime.date(int(enY),int(enM),int(enD))

dateRange = pd.date_range(startDate, endDate)

stBin = 34; enBin = 126
bins2merge = 3
numBins = (enBin - stBin + 1) / bins2merge

sensorArea_m2 = 0.005 # m^2
sensorArea_mm2 = 0.005*1e6 #mm^2
delta_t       = 10 # Integration Time (secs)
rainThreshhold = 0.1 # mm/h

binBottoms = np.genfromtxt("C:\MATLAB\SkyTemp\ETHdiams.txt",delimiter=",")
#binBottoms = binBott[stBin:enBin]

binSz = np.subtract(binBottoms[1:],binBottoms[:-1])
binSizes = np.append(binSz,binSz[-1])
dropDiams = np.cbrt(np.divide((np.power(np.add(binBottoms,binSizes),4) - np.power(binBottoms,4)),np.multiply(4,binSizes)))
dropSpeeds = np.subtract(9.65,np.multiply(10.3,np.exp(np.divide(np.multiply(-6,dropDiams),10))))
binVols    = np.multiply((4/3),np.multiply(np.pi,np.power(np.divide(dropDiams,2),3)))   #    %Volume of each bin in cubic mm

dataArray = np.empty([1,133]); #tempDataArray = np.empty([1,133]);

#Loop over the daterange:
for currDate in dateRange:
    print (currDate.strftime("%d-%b-%Y"))   
    
    disdFileName = os.path.abspath('C:/MATLAB/Disdrometer/2003-9/') + '\\cfarr-disdrometer_chilbolton_' + currDate.strftime("%Y%m%d") + '.nc'
    metSenFileName = os.path.abspath('C:/MATLAB/MetSensors/2003-9/') + '\\cfarr-met-sensors_chilbolton_' + currDate.strftime("%Y%m%d") + '.nc'
    rGaugeFileName = os.path.abspath('C:/MATLAB/RainGauge/2003-9/') + '\\cfarr-raingauge_chilbolton_' + currDate.strftime("%Y%m%d") + '.nc'
    beaconFileName = os.path.abspath('C:/MATLAB/BeaconSatData/Chilbolton/') + '\\' + currDate.strftime("%Y%m%d") + '_C_DBM'
        
    if os.path.exists(disdFileName):
        print("File found for: "+ currDate.strftime("%d-%b-%Y"))
        # import dataset
        daysDisdData = Dataset(disdFileName)
        daysMetData = Dataset(metSenFileName)
        daysRGaugeData = Dataset(rGaugeFileName)
        daysBeaconData1sec = np.genfromtxt(beaconFileName,delimiter=",")
        
        disdTimeStamp10s = daysDisdData.variables['time'][:]
        windTimeStamp10s = daysMetData.variables['time'][:]
        rGaugeTimeStamp10s = daysRGaugeData.variables['time'][:]

        dropSize = daysDisdData.variables['drop_size'][:]
        dropCounts10s = daysDisdData.variables['drop_count'][:]
        windSpeed10s = daysMetData.variables['wind_speed'][:]
        windDirn10s = daysMetData.variables['wind_direction'][:]
        temperature10s = daysMetData.variables['temperature'][:]
        rainGauge10s = daysRGaugeData.variables['drop_count_b'][:]
        #dewPoint = daysMetData.variables['dew_point'][:]
                
        
        windSpeed1min = np.mean(np.reshape(windSpeed10s, (6, -1)), axis=0)
        windDirn1min = np.mean(np.reshape(windDirn10s, (6, -1)), axis=0)
        temperature1min = np.mean(np.reshape(temperature10s, (6, -1)), axis=0)
        rainGauge1min = np.sum(np.reshape(rainGauge10s, (6, -1)), axis=0)
        daysBeaconData1min = np.sum(np.reshape(rainGauge10s, (6, -1)), axis=0)
        dropCounts1min = np.mean(np.reshape(dropCounts10s, (6, -1, 127)),axis=0)
        #dropCount1minTrunc = dropCounts1min[:,stBin:enBin]
        dropDensities = np.divide(dropCounts1min,np.multiply(sensorArea_m2,np.multiply(delta_t,np.multiply(dropSpeeds,binSizes))))

        disdRainRates = dropCounts1min.dot(binVols)*(3600/delta_t)/sensorArea_mm2
        
        
        #%%****************** Steps through the each sampled minute
        minInDay = len(dropCounts1min)
        for i in range(minInDay):
            #
            timeDiv = 24*i/minInDay
            hours = math.floor(24*i/minInDay); mins = i%60
            
            print (currDate.strftime("%d-%b-%Y ") + "%02d" % (hours,) + ':' + "%02d" % (mins,))
            #print(disdRainRates[i])
            
            # build minute's data and append
            #dataAray = np.append(np.append(np.append(np.append(np.append(np.append(dropDensities[i,:],disdRainRates[i]),rainGauge1min[i]),windSpeed1min[i]),windDirn1min[i]),temperature1min[i]),daysBeaconData1min[i])
            dataAray = np.append(np.append(np.append(np.append(np.append(np.append(dropCounts1min[i,:],disdRainRates[i]),rainGauge1min[i]),windSpeed1min[i]),windDirn1min[i]),temperature1min[i]),daysBeaconData1min[i])
            # np.append(dataArray, tempDataArray, axis=0)

            result = neuralNet3Layer.neuralNet(dataArray)



            if ((disdRainRates[i] >= rainThreshhold)): # and (sum(dropCount1minTrunc[i,:])>=20)):
                disdRainRate = disdRainRates[i]
                dropCount = dropCounts1min[i,:]
                dropCount10sec = dropCounts10s[i,:]
                dropCount1min = dropCounts1min[i,:]
                dropDensity = dropDensities[i,:]
                numOfBins = len(dropDiams)
                
                biggestBin = functions.findBiggestBin(dropDensity) #% Find the biggest bin containing more than zero drops
                
                noOfModes = 3   #% find number of modes
                
                #%% fit the GMM

# gmm = GaussianMixture(n_components=3, covariance_type="full", tol=0.001)
# gmm = gmm.fit(X=np.expand_dims(samples, 1))
# 
# # Evaluate GMM
# gmm_x = np.linspace(-2, 1.5, 5000)
# gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

                dropDens = np.expand_dims(dropDensity, axis=1)
                xPos = np.arange(len(dropDiams))
                
                gmm = GaussianMixture(n_components=noOfModes)
                gmm.fit(dropDens)   #ity)

                # # Evaluate GMM
                gmm_x = dropDiams    #np.linspace(-2, 1.5, 5000)
                
                #gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
                gmm_y = np.exp(gmm.score_samples(dropDiams.reshape(-1, 1))) #dropDens

                # # Make regular histogram

                plt.figure(1)
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
                ax.hist(dropDensity, bins=numOfBins, normed=True, alpha=0.5, color="#0070FF")
                ax.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
                plt.show()


                
                plt.figure(2)
                #plt.subplot(211)
                #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
                #ax.hist(dropDens, bins=numOfBins, normed=True, alpha=0.5, color="#0070FF")
                plt.bar(xPos,dropDensity, align='center', alpha=0.5)
                #plt.plot(dropDiams, gmm_y, color="crimson", lw=1, label="GMM")
                # #
                # # Annotate diagram
                plt.ylabel("N(D), Probability density")
                plt.xlabel("Drop diams (mm)")
                plt.title('DROP DENSITIES')
                # 
                # # Make legend
                plt.legend()
                plt.show()
                # 
                
                #plt.subplot(212)
                #plt.bar(xPos, dropDensity, align='center', alpha=0.5)
                ##plt.xticks(xPos, dropDiams)
                #plt.xlabel('Drop diams (mm)')
                #plt.ylabel('N(D)')
                #plt.title('DROP DENSITIES')
                ##plt.xticks(xPos, dropDiams)  #, rotation=17)
                #plt.show()




# =============================================================================
#                 print(gmm.means_)
#                 print('\n Covariances=')
#                 print(gmm.covariances_)
# 
#                 #plt.close()
#                 plt.figure(1)
#                 #plt.subplot(224)
#                 xPos = np.arange(len(dropDiams))
#                 plt.bar(xPos, dropDensity, align='center', alpha=0.5)
#                 plt.xticks(xPos, dropDiams)
#                 plt.xlabel('Drop diams (mm)')
#                 plt.ylabel('N(D)')
#                 plt.title('DROP DENSITIES')
#                  
#                 fig = plt.gcf()
#                 plt.show()
#                 #time.sleep(.3)
# =============================================================================
# =============================================================================
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# 
# # Define simple gaussian
# def gauss_function(x, amp, x0, sigma):
#     return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
# 
# # Generate sample from three gaussian distributions
# samples = np.random.normal(-0.5, 0.2, 2000)
# samples = np.append(samples, np.random.normal(-0.1, 0.07, 5000))
# samples = np.append(samples, np.random.normal(0.2, 0.13, 10000))
# 
# # Fit GMM
# gmm = GaussianMixture(n_components=3, covariance_type="full", tol=0.001)
# gmm = gmm.fit(X=np.expand_dims(samples, 1))
# 
# # Evaluate GMM
# gmm_x = np.linspace(-2, 1.5, 5000)
# gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
# 
# # Construct function manually as sum of gaussians
# gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
# for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
#     gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
#     gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w
# 
# # Make regular histogram
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
# ax.hist(samples, bins=50, normed=True, alpha=0.5, color="#0070FF")
# ax.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
# ax.plot(gmm_x, gmm_y_sum, color="black", lw=4, label="Gauss_sum", linestyle="dashed")
# 
# # Annotate diagram
# ax.set_ylabel("Probability density")
# ax.set_xlabel("Arbitrary units")
# 
# # Make legend
# plt.legend()
# 
# plt.show()
# 
# =============================================================================


# =============================================================================
#           plt.close()
#         plt.figure()
#         plt.plot(x,y)
#         plt.draw()
#         time.sleep(2)# 
# =============================================================================
# =============================================================================
                  # %% fit a lognormal
#                 #scale = np.trapz(dropDensity,dropDiams) # scale = trapz(drop_diams,drop_densities);   %scale = sum(drop_densities.*delta_diam); %   scale = trapz(bin_sizes,drop_densities);   %scale = sum(drop_densities.*delta_diam);
#                 
#                 ##probDensity = dropDensity/scale
#                 #mu    = np.sum(np.multiply(probDensity,np.multiply(deltaDiam,np.log(dropDiams))))
#                 
#                 ##sigma = sqrt(sum(probDensity.*deltaDiam.*log(dropDiams).^2) - mu^2);
#                 #sigma = np.sqrt((np.multiply(probDensity,np.multiply(deltaDiam,np.log(np.power(dropDiams,2))))).sum(0)) - mu^2);
#             
#                 #%% fit a gamma using moments of drop densities
#                 #m2 = (np.multiply(np.power(drop_diams,2),np.multiply(drop_densities,delta_diam))).sum(0);
#                 #m3 = (np.multiply(np.power(drop_diams,3),np.multiply(drop_densities,delta_diam))).sum(0);
#                 #m4 = (np.multiply(np.power(drop_diams,4),np.multiply(drop_densities,delta_diam))).sum(0);
#                 #m6 = (np.multiply(np.power(drop_diams,6),np.multiply(drop_densities,delta_diam))).sum(0);
# 
#                 #meanDiam = np.divide(m4,m3)
#                 #Nw  = (256/6) * np.divide(np.power(m3,5),np.power(m4,4))
#                 #eta  = np.divide(np.power(m4,2),np.multiply(m2,m6))
#                 ##### stopped here!!!  muGamma   = (7 - 11*eta - np.sqrt(power((7 - 11*eta),2) -    np.multiply(4*(eta-1),(30*eta-12))    ))/(2*(eta-1));
# =============================================================================
                 
                #%% GMM
                roughSampleSize = 1000
                #dropDensitiesScaled = round(roughSampleSize * dropDensities/np.multiply(dropDensities.sum(0),dropDiams)
                