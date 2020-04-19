#
# Case study 1 - Example 1 from Leyffer et al 2005 - ralph2 from the MacMPEC collection
# Interior Methods for Mathematical Programs with Complementarity Constraints
#
# Caroline S. M. Nakama
# April 2020
#

using LaTeXStrings, PyPlot;

include("mpcc.jl")

model = Model(Ipopt.Optimizer);
@variable(model, x[1] >= 0, start = 1);
@variable(model, y[1] >= 0, start = 1);

# define original objective function as an expression
@expression(model, obj, x[1]^2 + y[1]^2 - 4 * x[1] * y[1]);      # first example
# @expression(model, obj, (100 * x[1] - 1)^2 + (y[1] - 1)^2)      # second example

ans = solve_mpcc!(model, [(:x, :y)], true);

# plotting the results
n = length(ans)
nd = length(ans[1])
data = Array{Float64,2}(undef, n, nd)
for i in 1:n
    data[i,:] = ans[i]
end
fig = figure();
ax = PyPlot.subplot();
PyPlot.xlabel("Outer iterations");
PyPlot.ylabel(L"log_{10}");
for i in 1:nd
    PyPlot.plot(1:n, log.(data[:,i]), marker = i+6);
end
PyPlot.grid("on")
PyPlot.legend(["Objective", L"μ_k", L"π_k", "max(complementarity)"]);

