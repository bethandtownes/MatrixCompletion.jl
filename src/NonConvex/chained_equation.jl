module ChainedEquations
# TODO: Port the R module

import ..Concepts
import ..Concepts: MaybeMissing, type_conversion
import ..Utilities
import ..ModelFitting.ObservedOrMissing
import DataFrames

using RCall

struct MICE end

export MICE

function initialize_trackers(A::Array{MaybeMissing{Float64}}, type_assignment)
    type_tracker = Utilities.IndexTracker{Symbol}()
    if type_assignment == nothing
        Concepts.disjoint_join(type_tracker, Concepts.fit(DistributionDeduction(), A))
    else
        Concepts.disjoint_join(type_tracker, type_assignment)
    end
    Concepts.disjoint_join(type_tracker, Concepts.fit(ObservedOrMissing(), A))
    result = Concepts.groupby(type_tracker, [:Observed, :Missing])
    result[:Observed][:Total] = Base.findall(!ismissing, A)
    result[:Missing][:Total] = Base.findall(ismissing, A)
    result[:SingleView] = type_tracker
    return result
end


function Concepts.complete(model::MICE;
                           A::Array{MaybeMissing{Float64}}, 
                           impute_rounds = 10,
                           impute_method = "norm",
                           type_assignment)
    tracker = initialize_trackers(A, type_assignment)
    r_data = convert(DataFrames.DataFrame, A)
    r_missing = hcat(Base.map(x -> collect(x.I), tracker[:Missing][:Total])...)' 
    display(r_data)
    display(r_missing)
    @rput r_data;@rput r_missing; @rput impute_rounds; @rput impute_method;
    R"""
        # print(r_missing)
        print(sessionInfo())
        library(mice)
        library(Matrix)
        data <- airquality
        data[4:10,3] <- rep(NA,7)
        data[1:5,4] <- NA
        imp.train_raw <- mice(data, m = 5, method='cart')    
        # For(i in seq(1:dim(r_missing)[1]))
        # {
        #     r_data[r_missing[i,1],r_missing[i,2]] = NA
        # }
        # print(r_data)
        # imp.train_raw <- mice(r_data, m = 5, method='norm', printFlag=FALSE)
        # rImputed = as.matrix(complete(imp.train_raw))
        """
    # imputed = @rget rImputed
    # display(imputed)
    
    # rData = convert(Data
    # rMissing = asMatrix(findMissingEntry(size(data.sourceMatrix,1),
    #                                      size(data.sourceMatrix,2),
    #                                      data.observedEntry));
    # rImputeTimes = model.imputeTimes; rImputeMethod = model.imputeMethod;
    # # rImputeTimes = 5; rImputeMethod = "norm"
    # @rput rData;@rput rMissing;@rput rImputeTimes;@rput rImputeMethod;
    # R"""
    #     library(mice)
    #     for(i in seq(1:dim(rMissing)[1]))
    #     {
    #         rData[rMissing[i,1],rMissing[i,2]] = NA
    #     }
    #     imp.train_raw <- mice(rData, m=5, method='norm', printFlag=FALSE)
    #     rImputed = as.matrix(complete(imp.train_raw))
    #     """
    # imputed = @rget rImputed
    # ret = ImputedModel(original = data.sourceMatrix,
    #                    imputed = imputed,
    #                    missing = findMissingEntry(size(data.sourceMatrix,1),
    #                                               size(data.sourceMatrix,2),
    #                                               data.observedEntry));
    # logImputedModel(logConfig; model = ret);
    # return ret
end

end
