

def partial1[A,B,C](a: A, f: (A,B) => C): B => C = {
  f(a, _:B)
}

//A: Int
//B: String
//C: Boolean

//val lenOverX = (a: Int, b: String) => b.length > a
def lenOverX(a: Int)(b: String): Boolean = b.length > a
val Over4 = lenOverX(4) _


println(Over4("hello"))
println(lenOverX(4)("no"))
