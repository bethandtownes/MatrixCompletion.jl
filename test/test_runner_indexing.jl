using Test
import Random
import Distributions
import StatsBase


import MatrixCompletion.Utilities.Indexing:_check_bernoulli, _check_gaussian,_check_poisson

bernoulliMatrixTest1 = Random.rand(Distributions.Bernoulli(0.5),100);
nonBernoulliMatrixTest1 = Random.rand(Distributions.Uniform(1,10),100);
poissonMatrixTest1 = Random.rand(Distributions.Poisson(3),100);
nonPoissonMatrixTest1 = Random.rand(Distributions.Uniform(1,10),100);
poissonBernoulliDifferenceTest1 = Random.rand(Distributions.Poisson(3),100);
poissonBernoulliDifferenceTest2 = Random.rand(Distributions.Bernoulli(0.4),100);



@test _check_bernoulli(bernoulliMatrixTest1) == true
@test _check_bernoulli(nonBernoulliMatrixTest1) == false
@test _check_poisson(poissonMatrixTest1) == true
@test _check_poisson(nonBernoulliMatrixTest1) == false
@test _check_bernoulli(poissonBernoulliDifferenceTest1) == false
@test _check_poisson(poissonBernoulliDifferenceTest1) == true
@test _check_poisson(poissonBernoulliDifferenceTest2) == false
@test _check_bernoulli(poissonBernoulliDifferenceTest2) == true




import MatrixCompletion.Utilities.Indexing:construct_type_matrix,construct_index_tracker,DIST_FLAGS
import MatrixCompletion.Utilities.Indexing:Bernoulli,
                                           Poisson,
                                           Gaussian,
                                           Gamma,
                                           NegativeBinomial



# test 1: bernoulli only
type_matrix_test_input1 = Random.rand(Distributions.Bernoulli(0.5),5,5)
type_matrix_test_expected1 = Array{Union{DIST_FLAGS,Missing}}(undef,5,5)
fill!(type_matrix_test_expected1,Bernoulli)
@test construct_type_matrix(type_matrix_test_input1) ==  type_matrix_test_expected1

construct_type_matrix(type_matrix_test_input1)
# test 2: poisson only
type_matrix_test_input2 = Random.rand(Distributions.Poisson(5),5,5)
type_matrix_test_expected2 = Array{Union{DIST_FLAGS,Missing}}(undef,5,5)
fill!(type_matrix_test_expected2,Poisson)
@test construct_type_matrix(type_matrix_test_input2) == type_matrix_test_expected2


# test 3: gaussian only
type_matrix_test_input3 = Random.rand(Distributions.Gaussian(0,1),5,5)
type_matrix_test_expected3 = Array{Union{DIST_FLAGS,Missing}}(undef,5,5)
fill!(type_matrix_test_expected3,Gaussian)
@test construct_type_matrix(type_matrix_test_input3) == type_matrix_test_expected3



import MatrixCompletion.Utilities.RandomMatrices.rand
# test4
type_matrix_test_input4 = rand([(Distributions.Bernoulli(0.5),50=>25,10),(Distributions.Gaussian(5,10),50=>25,10)])
type_matrix_test_expected4_part1 = Array{Union{DIST_FLAGS,Missing}}(undef,50,25)
fill!(type_matrix_test_expected4_part1,Bernoulli)
type_matrix_test_expected4_part2 = Array{Union{DIST_FLAGS,Missing}}(undef,50,25)
fill!(type_matrix_test_expected4_part2,Gaussian)
type_matrix_test_expected4 = hcat(type_matrix_test_expected4_part1,type_matrix_test_expected4_part2)
@test construct_type_matrix(type_matrix_test_input4) == type_matrix_test_expected4


import MatrixCompletion.Utilities.Indexing:IndexTracker
import MatrixCompletion.Utilities.Indexing:construct_index_tracker
# INDEX TRACKER TEST 1
id_tracker_test_input1 = construct_type_matrix(Random.rand(Distributions.Gaussian(0,1),10,10))
id_tracker_test_output1 = construct_index_tracker(input_type_matrix = id_tracker_test_input1)
id_tracker_test_expect1_gaussian = [CartesianIndex(i,j) for i in 1:10 for j in 1:10]
@test sort(id_tracker_test_expect1_gaussian) == sort(id_tracker_test_output1.Gaussian)
@test isempty(id_tracker_test_output1.Gamma)
@test isempty(id_tracker_test_output1.Bernoulli)
@test isempty(id_tracker_test_output1.Poisson)
@test isempty(id_tracker_test_output1.NegativeBinomial)
@test isempty(id_tracker_test_output1.Missing)

# INDEX TRACKER TEST 2
id_tracker_test_input2 = construct_type_matrix(rand([(Distributions.Bernoulli(0.5), 100=>25, 10),
                                                     (Distributions.Gaussian(5,10), 100=>25, 10),
                                                     (Distributions.Poisson(10),    100=>25, 10),
                                                     (Distributions.Gaussian(0,1),  100=>25, 10)]))
id_tracker_test_output2 = construct_index_tracker(input_type_matrix = id_tracker_test_input2)
id_tracker_test_expect2_gaussian = [CartesianIndex(i,j) for i in 1:100 for j in [26:50;76:100]]
id_tracker_test_expect2_bernoulli = [CartesianIndex(i,j) for i in 1:100 for j in 1:25]
id_tracker_test_expect2_poisson = [CartesianIndex(i,j) for i in 1:100 for j in 51:75]
@test sort(id_tracker_test_output2.Gaussian) == sort(id_tracker_test_expect2_gaussian)
@test sort(id_tracker_test_output2.Bernoulli) == sort(id_tracker_test_expect2_bernoulli)
@test sort(id_tracker_test_output2.Poisson) == sort(id_tracker_test_expect2_poisson)
@test isempty(id_tracker_test_output2.Missing)
@test isempty(id_tracker_test_output2.Gamma)
@test isempty(id_tracker_test_output2.NegativeBinomial)
