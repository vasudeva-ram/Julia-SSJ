using Test
using SSJ

@testset "SSJ.jl tests" begin
    @test 1 == 1

    @testset "all run" begin
        # not a real test, but breaks if breaks.
        main();
        mainKS();
        mainSSJ();
    end
end