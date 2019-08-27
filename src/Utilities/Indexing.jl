module Indexing
using MatrixCompletion.Concepts 



abstract type AbstractTracker end

export DIST_FLAGS,
    AbstractTracker,IndexTracker,DistInfoTracker,
    construct_type_matrix,              # LEGACY CODE
    construct_index_tracker,            # LEGACY CODE 
    provide
    

## some type alias
const MatrixIndices = Array{CartesianIndex{2},1}

const DefaultNumberType = Union{Array{Float64,1},
                                Array{Int64,1},
                                Array{Union{Float64,Missing},1},
                                Array{Union{Int64,Missing},1}};


#using ...Losses
#import ...ExponentialFamily 

#using ...ExponentialFamily




## enum distributions flags



function _check_bernoulli(col::DefaultNumberType)
    actualColumn = filter(x -> !ismissing(x),col);
    distanceToBernoulli = mapreduce(x->min(abs(x-1),abs(x)),+,actualColumn);
    return distanceToBernoulli < 1e-5;
end


## Currently it is a rudimentary implementation that does not discern poisson
## and negative binomial. More intricate approach using Goodness or Fit test
## will be implemented in later versions.
function _check_poisson(col::DefaultNumberType)
    if _check_bernoulli(col)
        return false;
    end
    actualColumn = filter(x -> !ismissing(x),col);
    distanceToPoisson = mapreduce(x-> abs(x - round(x)),+,actualColumn);
    return distanceToPoisson < 1e-5
end



function _check_gaussian(col::DefaultNumberType)
    if !_check_poisson(col) && !_check_bernoulli(col)
        return true;
    end
    return false;
end


function _check_gamma(col::DefaultNumberType)
    return false;
end


function _check_negativebinomial(col::DefaultNumberType)
    return false;
end




struct DistInfoTracker <: AbstractTracker end



mutable struct IndexTracker  <: AbstractTracker
    Gaussian::MatrixIndices
    Bernoulli::MatrixIndices
    Poisson::MatrixIndices
    NegativeBinomial::MatrixIndices
    Gamma::MatrixIndices
    Missing::MatrixIndices
    Observed::MatrixIndices

    function IndexTracker()
    end
    
    function IndexTracker(input_type_matrix::Array{Union{DIST_FLAGS,Missing},2})
        tracker = new();
        tracker.Missing = findall(x -> ismissing(x), input_type_matrix);
        tracker.Gaussian = findall(x -> !ismissing(x) && x == Gaussian, input_type_matrix);
        tracker.Bernoulli = findall(x -> !ismissing(x) && x == Bernoulli, input_type_matrix);
        tracker.Poisson = findall(x -> !ismissing(x) && x == Poisson, input_type_matrix);
        tracker.Gamma = findall(x -> !ismissing(x) && x == Gamma, input_type_matrix);
        tracker.NegativeBinomial = findall(x -> !ismissing(x) && x == NegativeBinomial, input_type_matrix);
        tracker.Observed = findall(x-> !ismissing(x),input_type_matrix);
        return tracker;
    end
end



# function provide(object::IndexTracker;input_matrix::Array{Union{DIST_FLAGS,Missing},2})
#         tracker = IndexTracker()
#         tracker.Missing = findall(x -> ismissing(x), input_type_matrix);
#         tracker.Gaussian = findall(x -> !ismissing(x) && x == Gaussian, input_type_matrix);
#         tracker.Bernoulli = findall(x -> !ismissing(x) && x == Bernoulli, input_type_matrix);
#         tracker.Poisson = findall(x -> !ismissing(x) && x == Poisson, input_type_matrix);
#         tracker.Gamma = findall(x -> !ismissing(x) && x == Gamma, input_type_matrix);
#         tracker.NegativeBinomial = findall(x -> !ismissing(x) && x == NegativeBinomial, input_type_matrix);
#         tracker.Observed = findall(x-> !ismissing(x),input_type_matrix);
#      return tracker;
# end





# function provide(object::DistInfoTracker; input_matrix::Array{T,2}) where T<:Union{S,Missing} where S<:Real
#     n,m = size(input_matrix);
#     typemat = Array{Union{DIST_FLAGS,Missing}}(undef,m,n);
#     for col in 1:m
#         if _check_poisson(input_matrix[:,col])
#             typemat[:,col] .= Ref(Poisson);
#         elseif _check_bernoulli(input_matrix[:,col])
#             typemat[:,col] .= Ref(Bernoulli);
#         elseif _check_gaussian(input_matrix[:,col])
#             typemat[:,col] .= Ref(Gaussian);
#         elseif _check_gamma(input_matrix[:,col])
#             typemat[:,col] .= Ref(Gamma);
#         elseif _check_negativebinomial(input_matrix[:,col])
#             typemat[:,col] .= Ref(NegativeBinomial);
#         end
#     end
#     for missing_idx in findall(x->ismissing(x),input_matrix)
#         typemat[missing_idx] = missing;
#     end
#     return typemat;
# end
                 

# function assign!(object::DistInfoTracker;input_dist_info_matrix::Array{Union{DIST_FLAGS,Missing}},
#                  new_type::Array{Pair{UnitRange{Int64},ExponentialFamily}})
#     for (idx,the_type) in new_type
#         input_dist_info_matrix[idx] = convert(the_type)
#   end
# end

# function assign(object::DistInfoTracker;input_dist_info_matrix::Array{Union{DIST_FLAGS,Missing}},
#                 new_type::Array{Pair{UnitRange{Int64},ExponentialFamily}})
#     copyof_input_dist_info_matrix = deepcopy(input_dist_info_matrix)
#     for (idx,the_type) in new_type
#         copyof_input_dist_info_matrix[idx] = convert(the_type)
#     end
#     return copyof_input_dist_info_matrix
# end



function construct_type_matrix(input_matrix::Array{T,2}) where T<:Union{S,Missing} where S<:Real
    n,m = size(input_matrix);
    typemat = Array{Union{DIST_FLAGS,Missing}}(undef,m,n);
    for col in 1:m
        if _check_poisson(input_matrix[:,col])
            typemat[:,col] .= Ref(Poisson);
        elseif _check_bernoulli(input_matrix[:,col])
            typemat[:,col] .= Ref(Bernoulli);
        elseif _check_gaussian(input_matrix[:,col])
            typemat[:,col] .= Ref(Gaussian);
        elseif _check_gamma(input_matrix[:,col])
            typemat[:,col] .= Ref(Gamma);
        elseif _check_negativebinomial(input_matrix[:,col])
            typemat[:,col] .= Ref(NegativeBinomial);
        end
    end
    for missing_idx in findall(x->ismissing(x),input_matrix)
        typemat[missing_idx] = missing;
    end
    return typemat;
end


function construct_index_tracker(;input_type_matrix::Array{Union{DIST_FLAGS,Missing},2})
    return IndexTracker(input_type_matrix);
end




end
