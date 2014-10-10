
import matplotlib.pyplot as plt
import numpy as np

def visibility(Imin, Imax):
    if Imin == 0.0 and Imax == 0.0:
        return 0.0
    else:
        return (Imax - Imin)/(Imax + Imin)

FILEMEMORYLENGTHS = ["1", "2.5", "5", "7.5", "10", "12.5", "15", "20", "30", "40", "50", "60", "70", "80"]
THEDATASIZE = 10000
THEDATACHUNK = 1000
mmArray = "{"
mmErrorArray = "{"

for memLength in FILEMEMORYLENGTHS:
    
    fileName = "/Users/cricha5/Desktop/Physics/ASimpleModelOfQuandrops/Mathematica/HydroDoubleSlitAuto2M" + memLength + ".dat"    
    
    with open(fileName, 'r') as theFile:
            theData = list(theFile)
            theFile.close()
    
    theData = [float(e.strip()) for e in theData[0:THEDATASIZE]]
    
    #theDataRange = [e for e in theData if (e<-1.0)]
    #print(theDataRange)

    #plt.figure()
    #plt.hist(theData,50)
    #plt.show()

    visibilityList = []
    nCount = 0
    
    while nCount * THEDATACHUNK < THEDATASIZE:       
        
        nMin1 = 0.0
        nMin2 = 0.0
        nMax = 0.0
        vis1 = 0.0
        vis2 = 0.0    
        
        for poo in theData[(nCount * THEDATACHUNK):((nCount+1) * THEDATACHUNK)]:
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
    print("Memory  = ", float(memLength) * 10.0/4.0)
    print("numpy Mean = ", np.mean(visibilityList))
    print("numpy std = ", np.std(visibilityList))
    
    mmArray += "{" + str(float(memLength) * 10.0/4.0) + "," + str(np.mean(visibilityList)) + "},"
    mmErrorArray += "{{" + str(float(memLength) * 10.0/4.0) + "," + str(np.mean(visibilityList)) + "},ErrorBar[" + str(np.std(visibilityList)) + "]},"

mmArray += "}\n"
mmErrorArray += "}"
print(mmArray)
print(mmErrorArray)
#{{{5*DistanceToMemory, 0.14725928770279353`}, 
#  ErrorBar[0.028661093263093823`]}, {{7.5*DistanceToMemory, 
#   0.30194977986356375`}, 
#  ErrorBar[0.02093537328913231`]}, {{10*DistanceToMemory, 
#   0.4364037101662228`}, 
#  ErrorBar[0.038283847087534184`]}, {{12.5*DistanceToMemory, 
#   0.555312618256738`}, 
#  ErrorBar[0.020859221713538267`]}, {{15*DistanceToMemory, 
#   0.6144829800899165`}, 
#  ErrorBar[0.02384184697194225`]}, {{20*DistanceToMemory, 
#   0.7028137115657748`}, 
#  ErrorBar[0.025872064217915547`]}, {{30*DistanceToMemory, 
#   0.7720332695536726`}, 
#  ErrorBar[0.0398349295895124`]}, {{40*DistanceToMemory, 
#   0.7779378802433558`}, 
#  ErrorBar[0.02]}, {{50*DistanceToMemory, 0.8014930853016766`}, 
#  ErrorBar[0.02]}, {{60*DistanceToMemory, 0.7901549991483563`}, 
#  ErrorBar[0.02]}, {{70*DistanceToMemory, 0.8126739806779104`}, 
#  ErrorBar[0.02]}, {{80*DistanceToMemory, 0.807422436385155`}, 
#  ErrorBar[0.02]}}
#print("cheese")

#theFile = open("/Users/cricha5/Desktop/Physics/ASimpleModelOfQuandrops/Mathematica/HydroDoubleSlitAuto2M80.dat")

#theData = theFile.readlines([0,100])

#print(theData)

#theHist, theBin_edges = np.histogram(theData)

#print(theHist, theBin_edges)