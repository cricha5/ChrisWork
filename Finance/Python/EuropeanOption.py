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

def valueFiniteDifferenceGPU(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE):
    
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

    #the time loop
    for t in range(0,nTimeSteps):
        prg.finiteDifferenceNextStep(queue, fdResult.shape, None, fdResult.data, fdBuffer.data, numpy.uint32(t), numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(dt), numpy.float32(Int_Rate), numpy.float32(Vol))
    
        fdResult, fdBuffer = fdBuffer, fdResult

    result = float(fdResult[math.ceil(100/ dS)].get())
    return result,dt,dS


def valueMonteCarloGPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops=1):

    nextStepPathKernel = ElementwiseKernel(ctx,"float *latestStep, float *ran, float Strike, float Int_Rate, float Exp_Time, float dt, float Vol","float rval = exp((Int_Rate - 0.5f * Vol*Vol)*dt + Vol * sqrt(dt) * ran[i]); latestStep[i] *= rval;","nextStepPathKernel")

    excersisePriceKernel = ElementwiseKernel(ctx,"float *latestStep, float Strike, float Int_Rate, float Exp_Time","float rval = (latestStep[i]-Strike); latestStep[i] = exp(-Int_Rate*Exp_Time)  * max(rval,0.0f);","excersisePriceKernel")
    
    
    sumKernel = ReductionKernel(ctx, numpy.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]", arguments="__global float *x")
    
    
    nTimeStepsMonte = math.ceil(Exp_Time/dtMonte)
    #print(nTimeStepsMonte,nMonteLoops)
    #set up random number generator
    gen = RanluxGenerator(queue, nPaths, luxury=4, seed=time.time())

    #the arrays
    ran = cl.array.zeros(queue, nPaths, numpy.float32)
    latestStep = cl.array.zeros_like(ran)
    
    means = numpy.zeros(nMonteLoops)
    theMean = 0

    #the loop
    for loop in range(nMonteLoops):
        
        latestStep.fill(S_init)
        
        for t in range(nTimeStepsMonte):
            gen.fill_normal(ran)
            gen.synchronize(queue)
            nextStepPathKernel(latestStep, ran, Strike, Int_Rate, Exp_Time, dtMonte, Vol)


        excersisePriceKernel(latestStep, Strike, Int_Rate, Exp_Time)
        #print(latestStep)
        theMean = sumKernel(latestStep, queue).get()
        means[loop] = theMean / nPaths

    monteAverage = numpy.mean(means)
    monteStdDeviation = numpy.std(means)
    
    return monteAverage,dtMonte, monteStdDeviation


def valueFiniteDifferenceGPUOptimized(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE):
    
    finiteDifferenceEvolve_source = """
        __kernel void finiteDifferenceEvolve(__global float *fd1, __global float *fd2,const uint nAssetSteps, const float dS, const float Exp_Time, const float Int_Rate, const float Vol, const uint multiplier)
        {
        
        size_t global_id = get_global_id(0);
        
        float initSVal;
        
        float Delta;
        float Gamma;
        float Theta;
        
        size_t index_previousStep;
        size_t index_nextStep;
        size_t index_current;
        
        float dt = 0.8f / (Vol*Vol*nAssetSteps*nAssetSteps);
        uint nTimeSteps = ceil(Exp_Time / dt) ;
        
        uint mult;
        
        for(uint t=0; t < nTimeSteps; t++){
        
            for(mult = 0; mult < multiplier; mult++){
            
                index_previousStep = multiplier * global_id + mult - 1;
                index_nextStep = multiplier * global_id + mult + 1;
                index_current = multiplier * global_id + mult;
                
                initSVal = index_current * dS;
            
                if(index_current > 0 && index_current < (nAssetSteps - 1)){
            
                    Delta = (fd1[index_nextStep] - fd1[index_previousStep]) / (2.0f * dS);
                    Gamma = (fd1[index_nextStep] - (2.0f * fd1[index_current]) + fd1[index_previousStep]) / (dS * dS);
                    Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * fd1[index_current];
                
                    fd2[index_current] = fd1[index_current] + dt * Theta;
                
                }else if(index_current == 0){
        
                    fd2[index_current] = fd1[index_current] * (1.0f-Int_Rate * dt);
        
                }else if(index_current == nAssetSteps-1){
        
                    fd2[index_current] = 2.0f * fd1[index_previousStep] - fd1[index_previousStep-1];
                }else{
                    fd2[index_current] = -1.0f;
                }
                //fd2[index_current] = index_current*1.0f;
                
        
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            for(mult = 0; mult < multiplier; mult++){
                index_current = multiplier * global_id + mult;
                fd1[index_current] = fd2[index_current];
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        }
        """

    maxWorkItems = 1*2**9
    
    
    
    if(nAssetSteps > maxWorkItems):
        multiplier = math.ceil(nAssetSteps/maxWorkItems)
        nAssetSteps = mulitiplier * maxWorkItems
    else:
        maxWorkItems = nAssetSteps
        multiplier = 1
    
    #print(MAXWORKITEMS,mulitiplier,nAssetSteps)

    #deduce time constants and grid spacing
    dt = 0.8 / Vol**2 / nAssetSteps**2
    #dt=0.000001
    #nTimeSteps = math.ceil(Exp_Time / dt)+1
    dS = S_max / nAssetSteps

    #set up kernel
    mf = cl.mem_flags
    prg = cl.Program(ctx, finiteDifferenceEvolve_source).build()
    
    #the arrays
    #fdResult = cl.array.zeros(queue,nAssetSteps,numpy.float32)
    #fdBuffer = cl.array.zeros_like(fdResult)
    fdResult = numpy.zeros(nAssetSteps*200,numpy.float32)
    fdBuffer = numpy.zeros_like(fdResult)


    #Initial option values
    for i in range(nAssetSteps):
        fdResult[i] = max(PTYPE * (i*dS - Strike),0.0)
    
    mf = cl.mem_flags
    fdResult_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdResult)
    fdBuffer_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdBuffer)

#prg.finiteDifferenceEvolve(queue, (int(maxWorkItems),), None, fdResult.data, fdBuffer.data, numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(Exp_Time), numpy.float32(Int_Rate), numpy.float32(Vol), numpy.uint32(mulitiplier))
    prg.finiteDifferenceEvolve(queue, (int(maxWorkItems),), None, fdResult_dev, fdBuffer_dev, numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(Exp_Time), numpy.float32(Int_Rate), numpy.float32(Vol), numpy.uint32(multiplier))

#    print((fdResult[math.ceil(100/ dS)].get()+fdResult[math.ceil(100/ dS)+1].get())/2)
#    print(fdResult[math.ceil(100/ dS)+1].get())

    cl.enqueue_copy(queue, fdResult, fdResult_dev)

    print(fdResult)
    print(maxWorkItems,multiplier,nAssetSteps)
    print(512**3)
#result = float(fdResult[math.ceil(100/ dS)].get())
    result = float(fdResult[math.ceil(100/ dS)])
    return result,dt,dS

def valueFiniteDifferenceNUnderlyingsGPUOptimized(ctx,queue,S_init,S_max,nAssetSteps,Strike,Exp_Time,Int_Rate,Vol,PTYPE,nUnderlyings=1):
    
    finiteDifferenceEvolve_source = """
        __kernel void finiteDifferenceEvolve(__global float *fd1, __global float *fd2,const uint nAssetSteps, const float dS, const float Exp_Time, const float Int_Rate, const float Vol, const uint multiplier, const uint nUnderlyings)
        {
        
        size_t global_id = get_global_id(0);
        
        float initSVal;
        
        float Delta;
        float Gamma;
        float Theta;
        
        size_t index_previousStep;
        size_t index_nextStep;
        size_t index_current;
        
        float dt = 0.8f / (Vol*Vol*nAssetSteps*nAssetSteps);
        uint nTimeSteps = ceil(Exp_Time / dt) ;
        
        uint mult;
        float underLoopCount = pow(nAssetSteps*1.0f, (nUnderlyings-1)*1.0f);
        //underLoopCount = 100.0f;
        //uint underLoopCount = 1;
        float underStep;
        
        for(uint t=0; t < nTimeSteps; t++){
        
            for(underStep = 0.0f; underStep < underLoopCount; underStep++){
            
                for(mult = 0; mult < multiplier; mult++){
        
                    index_previousStep = multiplier * global_id + mult - 1;
                    index_nextStep = multiplier * global_id + mult + 1;
                    index_current = multiplier * global_id + mult;
            
                    initSVal = index_current * dS;
            
                    if(index_current > 0 && index_current < (nAssetSteps - 1)){
            
                        Delta = (fd1[index_nextStep] - fd1[index_previousStep]) / (2.0f * dS);
                        Gamma = (fd1[index_nextStep] - (2.0f * fd1[index_current]) + fd1[index_previousStep]) / (dS * dS);
                        Theta = 0.5f * Vol * Vol * initSVal * initSVal * Gamma + Int_Rate * initSVal * Delta - Int_Rate * fd1[index_current];
            
                        fd2[index_current] = fd1[index_current] + dt * Theta;
            
                    }else if(index_current == 0){
            
                        fd2[index_current] = fd1[index_current] * (1.0f-Int_Rate * dt);
            
                    }else if(index_current == nAssetSteps-1){
            
                        fd2[index_current] = 2.0f * fd1[index_previousStep] - fd1[index_previousStep-1];
                    
                    }else{
                        fd2[index_current] = -1.0f;
                    }
                    //fd2[index_current] = index_current*1.0f;
            
            
                }
                
            }
        
                barrier(CLK_GLOBAL_MEM_FENCE);
            
                for(mult = 0; mult < multiplier; mult++){
                    index_current = multiplier * global_id + mult;
                    fd1[index_current] = fd2[index_current];
                    }
            
                barrier(CLK_GLOBAL_MEM_FENCE);
        }
        }
        """
    
    maxWorkItems = 1*2**9
    
    
    
    if(nAssetSteps > maxWorkItems):
        mulitiplier = math.ceil(nAssetSteps/maxWorkItems)
        nAssetSteps = mulitiplier * maxWorkItems
    else:
        maxWorkItems = nAssetSteps
        mulitiplier = 1
    
    #print(MAXWORKITEMS,mulitiplier,nAssetSteps)
    
    #deduce time constants and grid spacing
    dt = 0.8 / Vol**2 / nAssetSteps**2
    #dt=0.000001
    #nTimeSteps = math.ceil(Exp_Time / dt)+1
    dS = S_max / nAssetSteps
    
    #set up kernel
    mf = cl.mem_flags
    prg = cl.Program(ctx, finiteDifferenceEvolve_source).build()
    
    #the arrays
    #fdResult = cl.array.zeros(queue,nAssetSteps,numpy.float32)
    #fdBuffer = cl.array.zeros_like(fdResult)
    fdResult = numpy.zeros(nAssetSteps*200,numpy.float32)
    fdBuffer = numpy.zeros_like(fdResult)
    
    
    #Initial option values
    for i in range(nAssetSteps):
        fdResult[i] = max(PTYPE * (i*dS - Strike),0.0)
    
    mf = cl.mem_flags
    fdResult_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdResult)
    fdBuffer_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=fdBuffer)

#prg.finiteDifferenceEvolve(queue, (int(maxWorkItems),), None, fdResult.data, fdBuffer.data, numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(Exp_Time), numpy.float32(Int_Rate), numpy.float32(Vol), numpy.uint32(mulitiplier))
    prg.finiteDifferenceEvolve(queue, (int(maxWorkItems),), None, fdResult_dev, fdBuffer_dev, numpy.uint32(nAssetSteps), numpy.float32(dS), numpy.float32(Exp_Time), numpy.float32(Int_Rate), numpy.float32(Vol), numpy.uint32(mulitiplier), numpy.uint32(nUnderlyings))
    
    #    print((fdResult[math.ceil(100/ dS)].get()+fdResult[math.ceil(100/ dS)+1].get())/2)
    #    print(fdResult[math.ceil(100/ dS)+1].get())
    
    cl.enqueue_copy(queue, fdResult, fdResult_dev)
    
    print(fdResult)
    print(maxWorkItems,mulitiplier,nAssetSteps)
    print(512**3)
    #result = float(fdResult[math.ceil(100/ dS)].get())
    result = float(fdResult[math.ceil(100/ dS)])
    return result,dt,dS


def valueMonteCarlo2GPU(ctx,queue,S_init,nPaths,Exp_Time, dtMonte,Strike,Int_Rate,Vol,PTYPE, nMonteLoops=1):
    
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
    print(maxWorkItems, multiplier, nPaths)
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



