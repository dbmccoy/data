def isSorted[A](ar: Array[A],gt: (A,A) => Boolean): Boolean = {
  @annotation.tailrec
  def go(n1: Int, n2: Int): Boolean = {
    val lower = gt(ar(n1), ar(n2))
    if(lower == false) false
    else if(n2 == ar.length - 1) true
    else(go(n2, n2+1))
  }
  go(0,1)
}

def isLower(a: Int, b: Int): Boolean = {
  a < b
}

val a = (1 to 10).toArray
val a2 = Array(1,2,3,5,4)

println(isSorted(a,isLower))
println(isSorted(a2,isLower))
