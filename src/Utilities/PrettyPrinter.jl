module PrettyPrinter

using Printf

export append_both_ends,
  toprule,
  bottomrule,
  table_header,
  add_row

function append_both_ends(str::String, token::String)
  return token * str * token
end

function Base.similar(str::String, token::Char)
  return token^length(str)
end

function toprule(col_name::Array{String}; io::IO = Base.stdout)
  new_rule = map(x -> Base.similar(append_both_ends(x, ""), '-'), col_name)  
  push!(new_rule, "") 
  @printf(io, "%s\n", foldl((x, y) -> x * "+" *y, new_rule, init=""))
  # @printf("%s\n", new_rule2)
end

function bottomrule(col_name::Array{String}; io::IO = Base.stdout)
  new_rule = map(x -> Base.similar(append_both_ends(x, ""), '-'), col_name)
  push!(new_rule, "") 
  @printf(io, "%s\n", foldl((x, y) -> x * "+" *y, new_rule, init=""))
  # @printf("%s\n", new_rule2)
end

function header_column(names::Array{String}; io::IO = Base.stdout)
  transformed_names = Base.similar(names)
  map!(x -> append_both_ends(x, ""), transformed_names, names)
  push!(transformed_names, "")
  @printf(io, "%s\n", foldl((x, y) -> x * "|" * y, transformed_names, init=""))
end


function ensure_width(w::Int64, str::String)
  if length(str) >= w
    return str
  end
  return append_both_ends(str, Base.repeat(" ", trunc(Int64, (w - length(str)/2)) + 1))
end


function table_header(col_names::Array{String}; io::IO = Base.stdout)
  # copy_col_names = map(x -> ensure_width(7, x), col_names)
  copy_col_names = col_names
  toprule(copy_col_names, io = io)
  header_column(copy_col_names, io = io)
  bottomrule(copy_col_names, io = io)
end

function add_row(header::Array{String}; data::Array{String}, io::IO = Base.stdout)
  # transformed_data = collect(zip(header, data))
  transformed_data = map(x -> rpad(append_both_ends(x[2], ""), length(append_both_ends(x[1], "")), " "), collect(zip(header, data)))
  push!(transformed_data, "")
  @printf(io, "%s\n", foldl((x, y) -> x * "|" * y, transformed_data, init=""))
end

end
