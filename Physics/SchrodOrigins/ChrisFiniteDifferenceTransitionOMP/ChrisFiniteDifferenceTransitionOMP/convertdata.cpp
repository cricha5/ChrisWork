#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    ifstream inp;
    inp.open( "probdata_ep-10.dat" );

    /*string testes;
    getline( inp, testes);
    cout << testes;
    getline( inp, testes);
    cout << testes;*/

  int Nt = 10;
  int Nx = 1000;
  for ( int t = 0; t < Nt; t++ )
  {
      ofstream out;
      out.open( "quantum_e0.0.dat" );
      
    for ( int i = 0; i < Nx; i++ )
    {
      long double x = -100 + i * 0.2;
      long double prob;
      inp >> prob;
        cout << prob;
      out << x << " " << prob << endl;
    }
      
      out.close();
  }
    inp.close();

    return 0;
}
