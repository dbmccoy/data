object fibonacci {
  def fib(i: Int) : Int = {
    @annotation.tailrec
    def go(n: Int, f1: Int, f2: Int): Int = {
      if(n == i) (f1 + f2)
      else go(n+1,f2,(f1+f2))
    }
    go(i,0,1)
  }

  def main(args: Array[String]): Unit = {
    println(fib(9))
  }
}
/*
0
1
1
2
3
5
8
13
21
*/
