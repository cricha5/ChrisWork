import BlackScholes
import numpy
import math
import time
import pyopencl as cl
import pyopencl.array as clarray
import random

from pyopencl.clrandom import RanluxGenerator
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel

def valueBlackScholes(S_init,Strike,Exp_Time,Int_Rate,Vol,PTYPE):
    
    return BlackScholes.valueEuropeanOption(S_init,Strike,Exp_Time,Int_Rate,Vol,PTYPE)

def valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE,nUnderlyings=1):
    
    finiteDifferenceNextStep_source = """
        __kernel void finiteDifferenceNextStep(__global float *fd1,__global float *fd2, const uint timeStep, const uint nAssetSteps, const float dS, const float dt, const float Int_Rate, const float Vol)
        {
        
        size_t global_id = get_global_id(0);
        
        float initSVal = global_id * dS;
        
        float Delta;
        float Gamma;
        float Theta;
        
        size_t index_previousStep = global_id-1;
        size_t index_nextStep = global_id+1;
        size_t index_current = global_id;
        
        if(global_id != 0 && global_id != (nAssetSteps - 1)){
        
        Delta = (fd1[index_nextStep] - fd1[index_previousStep]) / (2.f * dS);
        Gamma = (fd1[index_nextStep] - (2.f * fd1[index_current]) + fd1[index_previousStep]) / (dS * dS);
        Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * fd1[index_current];
        
        fd2[index_current] = fd1[index_current] + dt * Theta;
        
        }else if(global_id == 0){
        
        fd2[index_current] = fd1[index_current] * (1-Int_Rate * dt);
        
        }else{
        
        fd2[index_current] = 2 * fd2[index_previousStep] - fd2[index_previousStep-1];
        
        }
        
        }
        """
    
    
    #deduce time constants and grid spacing
    dt = 0.8 / Vol**2 / nAssetSteps**2
    nTimeSteps = math.ceil(Exp_Time / dt)+1
    dS = S_max / nAssetSteps
    
    #set up kernel
    mf = cl.mem_flags
    prg = cl.Program(ctx, finiteDifferenceNextStep_source).build()
    
    #the arrays
    fdResult = cl.array.zeros(queue,nAssetSteps,numpy.float32)
    fdBuffer = cl.array.zeros_like(fdResult)
    
    #Initial option values
    for i in range(nAssetSteps):
        fdResult[i] = max(PTYPE * (i*dS - Strike),0.0)
    #print(nTimeSteps,nAssetSteps)
    #the time loop
    for t in range(nTimeSteps):
        for underlyingStep in range(pow(nAssetSteps,(nUnderlyings-1))):
        #for underlyingStep in range(150):
            prg.finiteDifferenceNextStep(queue, fdResult.shape, None, fdResult.data, fdBuffer.data, numpy.uint32(t), numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(dt), numpy.float32(Int_Rate), numpy.float32(Vol))
        
        fdResult, fdBuffer = fdBuffer, fdResult

    result = float(fdResult[math.ceil(100/ dS)].get())
    return result,dt,dS


def valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops=1):
    
    nextStepPathKernel = ElementwiseKernel(ctx,"float *latestStep, float *ran, float Strike, float Int_Rate, float Exp_Time, float dt, float Vol","float rval = exp((Int_Rate - 0.5f * Vol*Vol)*dt + Vol * sqrt(dt) * ran[i]); latestStep[i] *= rval;","nextStepPathKernel")
    
    excersisePriceKernel = ElementwiseKernel(ctx,"float *latestStep, float Strike, float Int_Rate, float Exp_Time","float rval = (latestStep[i]-Strike); latestStep[i] = exp(-Int_Rate*Exp_Time)  * max(rval,0.0f);","excersisePriceKernel")
    
    
    sumKernel = ReductionKernel(ctx, numpy.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]", arguments="__global float *x")
    
    maxWorkItems = 1*2**9
    multiplier = 1
    
    if(nPaths > maxWorkItems):
        multiplier = math.ceil(nPaths/maxWorkItems)
        nPaths = multiplier * maxWorkItems
    else:
        maxWorkItems = nPaths
    #print(maxWorkItems, multiplier, nPaths)
    nTimeStepsMonte = math.ceil(Exp_Time/dtMonte)
    #print(nTimeStepsMonte,nMonteLoops)
    #set up random number generator
    gen = RanluxGenerator(queue, maxWorkItems, luxury=4, seed=time.time())

#the arrays
    ran = cl.array.zeros(queue, maxWorkItems, numpy.float32)
    latestStep = cl.array.zeros_like(ran)
    
    means = numpy.zeros(nMonteLoops)
    theMean = 0
    
    #the loop
    for loop in range(nMonteLoops):
        
        theSum = 0
        
        for mult in range(multiplier):
            
            latestStep.fill(S_init)
            
            
            
            for t in range(nTimeStepsMonte):
                gen.fill_normal(ran)
                gen.synchronize(queue)
                nextStepPathKernel(latestStep, ran, Strike, Int_Rate, Exp_Time, dtMonte, Vol)
            
            
            excersisePriceKernel(latestStep, Strike, Int_Rate, Exp_Time)
            #print(latestStep)
            
            #add to array
            
            theSum += sumKernel(latestStep, queue).get()
        means[loop] = theSum / nPaths
    
    monteAverage = numpy.mean(means)
    monteStdDeviation = numpy.std(means)
    
    return monteAverage,dtMonte, monteStdDeviation



