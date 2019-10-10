module MathLib
import LinearAlgebra

macro api(as,of)
    eval(Expr(:toplevel,:($as=$of),:(export $as)))
end

using ..MathLibSignatures
using ..Concepts



@api MathematicalObject MathLibSignatures.MathematicalObject,
@api Cone               MathLibSignatures.Cone
@api Interval           MathLibSignatures.Interval
@api SemidefiniteCone   MathLibSignatures.SemidefiniteCone
@api ClosedInterval     MathLibSignatures.ClosedInterval

@api project            MathLibSignatures.project
@api project!           MathLibSignatures.project!





function LinearAlgebra.rank(obj::SemidefiniteCone)
  return obj.rank
end


include("./Projections.jl")


end
