#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

double gaussian( double x )
{
  return ( exp( - x * x ) );
}

double width( double hbar, double t, double sigma )
{
  double w1 = hbar * t / ( 2 * sigma );
  double w2 = w1 * w1 + sigma * sigma;
  return ( sqrt( w2 ) );
}

double density( double x, double d, double hbar, double t, double sigma )
{
  double sigmat = width( hbar, t, sigma );
  double incoh = gaussian( ( x - d ) / ( 2 * sigmat ) ) + 
    gaussian( ( x + d ) / ( 2 * sigmat ) );
  double interf = sin( hbar * t * x * d / 
		       ( 4 * sigma * sigma * sigmat * sigmat ) );
  double rho = incoh * incoh - 4 * interf * interf
    * exp( - ( x * x + d * d ) / ( 2 * sigmat * sigmat ) );
  rho /= sigmat * 2 * sqrt( 2 * M_PI ) * sigma * 
    ( gaussian( d / ( sqrt( 2 ) * sigma ) ) + 1 );
  return rho;
}

main()
{
  double sigma = 1;
  double d = 3;
  double epsilon = 1;
  double hbar = sqrt( epsilon );
  double t = 20;

  ofstream out( "prob.dat" );

  double xmin = -30;
  double xmax = 30;
  int Nx = 600;

  for ( int i = 0; i < Nx; i++ )
  {
    double x = xmin + ( i + 0.5 ) * ( xmax - xmin ) / Nx;
    double y = density( x, d, hbar, t, sigma );
    out << x << " " << y << endl;
  }
}
