










function analyzeType(matrix::Array{Float64,2})
    #
    # function isBinary(column::Array{Float64,1})
    #     projectedColumn = map(x -> abs(1-x)/abs(x)>1 ? 0:1,column);
    #     return vecnorm(projectedColumn - column) < 1e-6;
    # end

    function isBinary(column::Array{Float64,1})
        projectedColumn = map(x -> abs(1-x)/abs(x)>1 ? 0 : 1,filter(x->x!=MISSING,column));
        return norm(projectedColumn - filter(x->x!=MISSING,column)) < 1e-6;
    end

    function isCount(column::Array{Float64,1})
        projectedColumn = map(round,filter(x->x!=MISSING,column));
        return norm(projectedColumn - filter(x->x!=MISSING,column)) < 1e-6;
    end


    R,C  = size(matrix);
    typeMatrix = UNDEF .* ones(R,C);

    for c = 1:C
        if isBinary(matrix[:,c])
            typeMatrix[:,c] .= BINARY;
        elseif isCount(matrix[:,c])
            typeMatrix[:,c] .= COUNT;
        else
            typeMatrix[:,c] .= CONTINOUS;
        end
    end
    return typeMatrix;
end
