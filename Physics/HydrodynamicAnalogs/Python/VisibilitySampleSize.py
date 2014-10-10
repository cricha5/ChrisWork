
#import matplotlib.pyplot as plt
import numpy as np

def visibility(Imin, Imax):
    if Imin == 0.0 and Imax == 0.0:
        return 0.0
    else:
        return (Imax - Imin)/(Imax + Imin)

THEDATASIZE = 10000
CHUNKS = [10,100,200,400,500,1000,2000,5000]

    
fileName = "/Users/cricha5/Desktop/Physics/ASimpleModelOfQuandrops/Mathematica/HydroDoubleSlitAuto2M80.dat"    

with open(fileName, 'r') as theFile:
        theData = list(theFile)
        theFile.close()

theData =[float(e.strip()) for e in theData[0:THEDATASIZE]]

#plt.hist(theData,50)

bestStd = 100.0
bestChunkSize = 0

for chunkSize in CHUNKS:
    
    visibilityList = []
    nCount = 0
    std = 0.0

    while nCount * chunkSize < THEDATASIZE:
    
        nMin1 = 0.0
        nMin2 = 0.0
        nMax = 0.0
        vis1 = 0.0
        vis2 = 0.0    
        
        for poo in theData[nCount * chunkSize:(nCount+1) * chunkSize]:
            if poo < 0.05 and poo > -0.05:
                nMax += 1
            if poo < 0.27 and poo > 0.17:
                nMin1 += 1
            if poo < -0.17 and poo > -0.27:
                nMin2 += 1
    #    print(nMax)
    #    print(nMin1)
    #    print(nMin2)
        
        vis1 = visibility(nMin1, nMax)
        vis2 = visibility(nMin2, nMax)
        
        visibilityList.append(vis1)
        visibilityList.append(vis2)
        
        nCount+=1
        
        #print(visibilityList)
        #print "Memory  = ", float(memLength) * 10/4
        #print "numpy Mean = ", np.mean(visibilityList)
        std = np.std(visibilityList)
        
    print "Chunk Size = ", chunkSize
    print "std  = ", std
    
    if std < bestStd:
        bestStd = std
        bestChunkSize = chunkSize

print "Best Chunk Size = ", bestChunkSize
print "Best std  = ", bestStd


#print("cheese")

#theFile = open("/Users/cricha5/Desktop/Physics/ASimpleModelOfQuandrops/Mathematica/HydroDoubleSlitAuto2M80.dat")

#theData = theFile.readlines([0,100])

#print(theData)

#theHist, theBin_edges = np.histogram(theData)

#print(theHist, theBin_edges)