//
//  main.c
//  ChrisFiniteDifferenceTransitionOMP
//
//  Created by Christopher Richardson on 14/05/13.
//  Copyright (c) 2013 Christopher Richardson. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
//#include <omp.h>

#define XLENGTH 1000
#define XSTART -100
#define XSTEP 0.2
#define XCUTS 1000
//#define PROBCUTS 50
#define TLENGTH 10000000
#define TSTEP 0.000001
#define TCUTS 10
#define SIGMA 2.0
#define MASS 1.0
#define EP 0.0
#define DSLIT 20.0
#define FILENAME "probdata_collide_ep-0.dat"
#define FILENAME2 "probdata_collide_ep-0.dat"

const unsigned int _xLength = XLENGTH;
const double _xStart = XSTART;
const double _xStep = XSTEP;

//const unsigned int _cuts = XCUTS;

const unsigned int _tLength = TLENGTH;
const double _tStep = TSTEP;

const double _ep = EP;
const double _sigma = SIGMA;
const double _dslit = DSLIT;

_Complex double _wfunc[XLENGTH];
_Complex double _wfuncNext[XLENGTH];
double _returnMatrix[TCUTS][XCUTS];

_Complex double recurrenceRelation(_Complex double wfAtPrev,_Complex double wfAtCurrent, _Complex double wfAtNext);

int writeToFile(char * fileName);
int updateProgress(unsigned int progress);
double cnorm(_Complex double val);

_Complex double initialDistribution(double x);
_Complex double initialDistributionCollide(double x, double pnot);
_Complex double tStepDxStepSq = TSTEP/ (2 * XSTEP * XSTEP);

_Complex double recurrenceRelationBack(_Complex double wfAtPrev,_Complex double wfAtCurrent, _Complex double wfAtNext);

int appendToFile(char * fileName, double * outputMatrix);

int main(int argc, const char * argv[])
{
    //FILE * theFile;
    
    //theFile = fopen(FILENAME, "w");
    fclose(fopen(FILENAME, "w"));
    
    time_t startTime, endTime;
    
    startTime = time(NULL);
    
    #pragma omp parallel
    {
        _Complex double * _wfuncPtr = _wfunc;
        _Complex double * _wfuncNextPtr = _wfuncNext;
        _Complex double * wfuncTempPtr;
        
        unsigned int xIndex, xChunk = XLENGTH / XCUTS;
        unsigned int c = 1,cutCount = 0,tChunk = TLENGTH / TCUTS;
        
        int x=0,t=0;
        
        
        #pragma omp for
        for (x=0; x < XLENGTH; x++) {
            //_wfuncPtr[x] = initialDistribution(XSTART + x * XSTEP);
            _wfuncPtr[x] = initialDistributionCollide(XSTART + x * XSTEP, 4);
        }
        
        for (t = 0; t < _tLength; t++) {
            //printf("starting loop again\n");
            //Take care of recurrence
            
            #pragma omp for
            for (x=1; x < XLENGTH-1 ; x++) {
                
                _wfuncNextPtr[x] = recurrenceRelation(_wfuncPtr[x - 1], _wfuncPtr[x], _wfuncPtr[x + 1]);

            }
            
            //End take care of recurrence
            //printf("took care of recurrence\n");
            
            //Take care of boundries
            
            _wfuncNextPtr[0] = recurrenceRelation(0, _wfuncPtr[0], _wfuncPtr[1]);
            
            _wfuncNextPtr[_xLength - 1] = recurrenceRelation(_wfuncPtr[_xLength-2], _wfuncPtr[_xLength - 1], 0);
            
            //End take care of boundries
            //printf("took care of boundries\n");
            
            //Update return matrix
            
            if (c == tChunk) {
                #pragma omp for
                for (x=0; x < XCUTS; x++) {
                    xIndex = x * xChunk;
                    _returnMatrix[cutCount][x] = cnorm(_wfuncNextPtr[xIndex]);
                }
                c = 1;
                updateProgress(cutCount);
                #pragma omp single
                {
                    appendToFile(FILENAME, _returnMatrix[cutCount]);
                }
                cutCount++;
            }
            c++;
            
            //End update return matrix
            //printf("took care of update return matrix\n");
            
            //Swap pointers
            
            wfuncTempPtr = _wfuncNextPtr;
            _wfuncNextPtr = _wfuncPtr;
            _wfuncPtr = wfuncTempPtr;
            
            //End swap pointers
            //printf("took care of swap pointers\n");
        }
        
        //printf("out of loop\n");
        
        //Make sure we got the last time step.
        
        #pragma omp for
        for (x=0; x < XCUTS; x++) {
            xIndex = x * xChunk;
            _returnMatrix[TCUTS - 1][x] = cnorm(_wfuncNextPtr[xIndex]);
        }
        
        //End make sure we got the last time step.
    }
    
    endTime = time(NULL);
    
    printf("time for parrallel loop is %f\n", difftime(endTime,startTime)/60);
    
    //appendToFile(FILENAME, _returnMatrix[TCUTS - 1]);
    printf("write to file returns %d\n",writeToFile(FILENAME));
    //printf("write to file returns %d\n",writeToFile(strcat("probdata_ep-", "5.dat")));
    
    return 0;
}

_Complex double recurrenceRelation(_Complex double wfAtPrev,_Complex double wfAtCurrent, _Complex double wfAtNext)
{
    
    _Complex double epTerm, tStepDxStepSq = _tStep/ (_xStep * _xStep), returnValue;
    
    double wfReal = creal(wfAtCurrent), wfImag = cimag(wfAtCurrent);
    
    if(cabs(wfAtCurrent) == 0.0){
        epTerm = wfAtCurrent * cnorm(wfAtCurrent);
    }else{
        epTerm = wfAtCurrent * (cabs(wfAtNext) - 2.0 * cabs(wfAtCurrent) + cabs(wfAtPrev)) / cabs(wfAtCurrent);
    }
    
    if(isnan(epTerm) || isnan(creal(epTerm)) || isnan(cimag(epTerm)))
    {
        printf("\nHMMMMM absTerm value is nan");
    }
    
    returnValue = wfAtCurrent + tStepDxStepSq * MASS * (wfAtNext - 2.0 * wfAtCurrent + wfAtPrev - (1 - EP) * epTerm) * _Complex_I;
    
    if(isnan(returnValue))
    {
        printf("\nReturn value is nan! divProb real = %e imag = %e\n, wfReal = %e, wfImag = %e", creal(wfAtCurrent), cimag(wfAtCurrent), wfReal, wfImag);
        return 0;
    } else {
        return returnValue;
    }
}

_Complex double recurrenceRelationBack(_Complex double wfAtPrev,_Complex double wfAtCurrent, _Complex double wfAtNext)
{
    
    _Complex double tStepDxStepSq = _tStep/ (_xStep * _xStep), divProb, returnValue;
    
    double absTerm, wfReal = creal(wfAtCurrent), wfImag = cimag(wfAtCurrent);
    
    absTerm = cabs(wfAtNext) - 2.0 * cabs(wfAtCurrent) + cabs(wfAtPrev);
    
    divProb = wfAtCurrent / cabs(wfAtCurrent);
    
    if(isnan(divProb) || isnan(wfReal) || isnan(wfImag))
    {
        if ((wfReal == 0.0) && (wfImag == 0.0)) {
            if((creal(wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev) != 0.0)|| (cimag(wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev)) != 0.0){
                printf("\nHMMMMM Return value is nan.  Zeros! Term 1 R = %e I = %e, Term 2 R = %e, Difference = %e\n", creal(wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev), cimag(wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev), absTerm, creal(wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev) - absTerm);
            }
            divProb = 1;
            //divProb = 0;
            
            double wfPrevReal = creal(wfAtPrev), wfPrevImag = cimag(wfAtPrev);
            if(isnan(wfPrevReal) && isnan(wfPrevImag)){
                divProb = wfAtPrev / cabs(wfAtPrev);
            }
            if(isnan(divProb)){
                divProb = 1;
            }
            
        } else if(wfReal == wfImag) {
            divProb = (sqrt(2) / 2) * (1 + _Complex_I);
        } else if((wfReal == 0.0) && (wfImag != 0.0)){
            divProb = 1;
        } else if((wfReal != 0.0) && (wfImag == 0.0)) {
            divProb = -1;
        } else if(((wfReal == 0.0) && (wfImag == -0.0)) || ((wfReal == -0.0) && (wfImag == 0.0))) {
            divProb = 1;
        } else {
            double q = 2 * wfReal / ((wfReal / wfImag) + 1);
            _Complex double p = q / (wfReal + wfImag - q);
            p = p * _Complex_I;
            divProb = (wfReal - wfImag)/(wfReal + wfImag - q) + p;
            printf("\nHMMMMM Return value is nan! divProb real = %e imag = %e, wfReal = %e, wfImag = %e/n", creal(divProb), cimag(divProb), wfReal, wfImag);
        }
        
        if(isnan(divProb)){
            divProb = 1;
            printf("\nHMMMMM divProb value is nan");
        }
    }
    
    if(creal(divProb) > 1.0){
        divProb = 1;
    }
    
    returnValue = wfAtCurrent + tStepDxStepSq * MASS * (wfAtNext - 2.0 *  wfAtCurrent + wfAtPrev - (1 - EP) * divProb * absTerm) * _Complex_I;
    
    if(isnan(returnValue))
    {
        printf("\nReturn value is nan! divProb real = %e imag = %e\n, wfReal = %e, wfImag = %e", creal(divProb), cimag(divProb), wfReal, wfImag);
        return 0;
    } else {
        return returnValue;
    }
}

int writeToFile(char * fileName)
{
    FILE * theFile;
    int t,x;
    
    theFile = fopen(fileName, "w");
    
    for (t=0; t < TCUTS;t++) {
        for (x=0; x < XCUTS; x++) {
            fprintf(theFile,"%e ", _returnMatrix[t][x]);
        }
        fprintf(theFile,"\n");
    }
    
    return fclose(theFile);
    
}

int appendToFile(char * fileName, double * outputMatrix)
{
    FILE * theFile;
    int x;
    
    theFile = fopen(fileName, "a");
    
    for (x=0; x < XCUTS; x++) {
        fprintf(theFile,"%e ", outputMatrix[x]);
    }
    fprintf(theFile,"\n");
    
    return fclose(theFile);
    
}

int updateProgress(unsigned int progress)
{
    printf("%i\n",progress);
    return 1;
}

double cnorm(_Complex double val)
{
    return creal(val) * creal(val) + cimag(val) * cimag(val);
}

_Complex double initialDistribution(double x)
{
    _Complex double returnValue;
    
    returnValue = ((exp(-(x - _dslit)*(x - _dslit) / (4.0 * _sigma * _sigma)) + exp(-(x + _dslit)*(x + _dslit) / (4 * _sigma * _sigma))) / (pow(2.0, 3.0 / 4.0) * pow(M_PI, 1.0 / 4.0) * sqrt(_sigma * (1.0 + exp(-_dslit * _dslit / (2.0 * _sigma * _sigma))))));
    
    if(returnValue != returnValue)
    {
        printf("\nInitial return value is nan!\nx = %f\n",x);
    }
    
    return returnValue;
}

_Complex double initialDistributionCollide(double x, double pnot)
{
    _Complex double returnValue;
    
    returnValue = cexp(- x * pnot * _Complex_I - (x - _dslit)*(x - _dslit) / (4.0 * _sigma * _sigma)) / csqrt(_sigma * csqrt(2 * M_PI));
    
    //returnValue = ((cexp(- x * pnot * _Complex_I) * cexp(- (x - _dslit)*(x - _dslit) / (4.0 * _sigma * _sigma)) + cexp( x * pnot * _Complex_I) * cexp(- (x + _dslit)*(x + _dslit) / (4 * _sigma * _sigma))) / (cpow(2.0, 3.0 / 4.0) * cpow(M_PI, 1.0 / 4.0) * csqrt(_sigma * (1.0 + cexp(-(_dslit * _dslit + 4.0 * pnot * pnot * _sigma * _sigma * _sigma * _sigma) / (2.0 * _sigma * _sigma))))));
    
    //returnValue = ((cexp(- (x - _dslit)*(x - _dslit) / (4.0 * _sigma * _sigma)) + cexp( x * pnot * _Complex_I) * cexp(- (x + _dslit)*(x + _dslit) / (4 * _sigma * _sigma))) / (cpow(2.0, 3.0 / 4.0) * cpow(M_PI, 1.0 / 4.0) * csqrt(_sigma * (1.0 + cexp(-_dslit * _dslit / (2.0 * _sigma * _sigma))))));
    
    if(returnValue != returnValue)
    {
        printf("\nInitial return value is nan!\nx = %f\n",x);
    }
    
    return returnValue;
}

