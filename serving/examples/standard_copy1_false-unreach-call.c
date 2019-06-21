//Expected ranking:
// 2LS: UNKNOWN .090
//CBMC: UNKNOWN 86
//CMBC-Path: UNKNOWN 880
//CPAChecker: UNKNOWN 900
//DepthK: UNKNOWN 160
//DIVINE: UNKNOWN 900
//MAP2Check: TRUE 6.6
//PeSCo: UNKNOWN 900
//Pinaka: TRUE 480
//Symbotic: TRUE 6.6
//VeriAbs: TRUE 4.6
//VeriFuzz: True 4.8
//VIAP: True 7.4

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if(!(cond)) { ERROR: __VERIFIER_error(); } }
extern int __VERIFIER_nondet_int();
int main( ) {
  int a1[100000];
  int a2[100000];
  int a;
  for ( a = 0 ; a < 100000 ; a++ ) {
      a1[a] = __VERIFIER_nondet_int();
      a2[a] = __VERIFIER_nondet_int();
  }
  int i;
  for ( i = 0 ; i < 100000 ; i++ ) {
    a1[i] = a1[i];
  }
  int x;
  for ( x = 0 ; x < 100000 ; x++ ) {
    __VERIFIER_assert( a1[x] == a2[x] );
  }
  return 0;
}
