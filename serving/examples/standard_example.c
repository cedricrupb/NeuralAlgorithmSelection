int main() {
  fib(5);
  return 0;
}

int fib(int i){

  if(i <= 1){
    return 1;
  }

  int a[100];

  for(int k = 0; k < 100; k++){
    a[k] = 1;
  }

  return fib(i - 1) + fib(i - 2);
}
