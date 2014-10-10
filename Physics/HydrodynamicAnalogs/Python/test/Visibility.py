
import matplotlib.pyplot as plt
import statistics as st

def visibility(Imin, Imax):
    if Imin == 0.0 and Imax == 0.0:
        return 0.0
    else:
        return (Imax - Imin)/(Imax + Imin)

FILEMEMORYLENGTHS = ["1", "2.5"]
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
    print("numpy Mean = ", st.mean(visibilityList))
    print("numpy std = ", st.stdev(visibilityList))
    
    mmArray += "{" + str(float(memLength) * 10.0/4.0) + "," + str(st.mean(visibilityList)) + "},"
    mmErrorArray += "{{" + str(float(memLength) * 10.0/4.0) + "," + str(st.mean(visibilityList)) + "},ErrorBar[" + str(st.stdev(visibilityList)) + "]},"

mmArray += "}\n"
mmErrorArray += "}"
print(mmArray)
print(mmErrorArray)
