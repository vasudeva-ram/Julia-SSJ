@testitem "get_yso" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    
    steadystate = SSJ.solve_SteadyState(a,guess = (0.011,0.05)); # find the steady state

    yso_r = SSJ.get_yso(a, steadystate, 'r')
    @test maximum(abs,yso_r[end] - yso_r[end-1]) > 1e-4


    # Get the policies (yso) and associated transition matrices (Λso)
    yso_r = SSJ.get_yso(a, steadystate, 'r', shock_T_minus = 5)
    @test yso_r[end] ≈ yso_r[end-1] atol = 1e-6
    @test yso_r[end] ≈ yso_r[end-3] atol = 1e-6
    @test yso_r[end] ≈ yso_r[end-4] atol = 1e-6
    @test maximum(abs,yso_r[end-5] - yso_r[end-4]) > 1e-4
    @test maximum(abs,yso_r[end-6] - yso_r[end-5]) > 1e-6
    # @test yso_r[end-1] == yso_r[end-2]
    # @test yso_r[end-1] == yso_r[1]
end

