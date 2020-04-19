#
# Case study 2 - Suggested problem from Nonlinear programming: concepts, algorithms...
#
# Caroline S. M. Nakama
# April 2020
#
using LaTeXStrings, PyPlot;

include("mpcc.jl")

ans = Array{Any,1}(undef,4)
for j in 1:4
    N = 10^j;

    model = Model(Ipopt.Optimizer);
    @variable(model, x[1:N], start = 1);
    @variable(model, -1 <= u[1:N] <= 1, start = 1);
    @variable(model, s1[1:N] >= 0, start = 1);
    @variable(model, s2[1:N] >= 0, start = 1);
    @variable(model, x0); 
    fix(x0, -2.0);
    @variable(model, h); 
    fix(h, 2.0 / N);

    @constraint(model, x[1] == x0 + h * (-u[1] + 2));
    @constraint(model, [i = 2:N], x[i] == x[i-1] + h * (-u[i] + 2));
    @constraint(model, [i = 1:N], x[i] == s1[i] - s2[i]);
    @expression(model, c1[i = 1:N], 1 - u[i]);
    @expression(model, c2[i = 1:N], 1 + u[i]);
    # define original objective function as an expression
    @expression(model, obj, (x[N] - 5.0 / 3)^2 + sum(2 / N * x[i]^2 for i in 1:N));

    comp = [(:s1, :c1), (:s2, :c2)];
    ans[j] = solve_mpcc!(model, comp, true);
end

# plotting the results
fig = figure();
for j in 1:4
    n = length(ans[j])
    nd = length(ans[j][1])
    data = Array{Float64,2}(undef, n, nd)
    for i in 1:n
        data[i,:] = ans[j][i]
    end
    ax = PyPlot.subplot(2, 2, j);
    # PyPlot.title("N = $(10^j)", fontsize = 10, horizontalalignment = "left")
    PyPlot.xlabel("Outer iterations");
    PyPlot.xticks(1:2:n)
    PyPlot.ylabel(L"log_{10}");
    for i in 1:nd
        PyPlot.plot(1:n, log.(data[:,i]), marker = i+6);
    end
    PyPlot.text(n-4, -5, "N = $(10^j)")
    PyPlot.grid("on")
    PyPlot.legend(["Objective", L"μ_k", L"π_k", "max(complementarity)"], fontsize = 8);
end