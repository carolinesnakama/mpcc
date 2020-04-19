#
# Implementation of Algorithm I from Leyffer et al 2005 
# Interior Methods for Mathematical Programs with Complementarity Constraints
#
# Caroline S. M. Nakama
# April 2020
#

# needs JuMP v0.21.1
using JuMP, Ipopt, LinearAlgebra, Printf;

function solve_mpcc!(model, comp, plot_data = false, objective = :obj)
    # defining parameters
    gamma = 0.4;
    kappa = 0.2;
    sigma = 10;
    theta = 10;
    mu = 0.1;
    pi = 1;
    tol = 1e-8;
    k_max = 100;
    
    # getting model variables
    nc = length(comp);
    cvar = Array{Tuple{Any,Any},1}(undef, nc);
    cval = copy(cvar);
    nvc = Array{Int64,1}(undef, nc + 1);
    nvc[1] = 1;
    for i in 1:nc
        cvar[i] = (getindex(model, comp[i][1]), getindex(model, comp[i][2]));
        nvc[i+1] = length(cvar[i][1]) + nvc[i];
    end
    obj = getindex(model, objective);

    # initial values
    k = 1;
    last_obj = 0;
    min_val = zeros(sum(nvc),1);
    ans = [];
    time = 0;
    if plot_data
        pl = [];
    end

    # setting objective function with penalty term
    penalty = @expression(model, pen, 
              sum(cvar[i][1][j] * cvar[i][2][j] for i in 1:nc for j in 1:nvc[i+1]-nvc[i]));
    @objective(model, Min, obj + pi * pen);

    # printing header for output
    @printf("\nIter   Termination_status    Objective    MPCC_compl lg(mu)   pi    CPUs \n")

    # outer loop
    while k <= k_max
        # reset variables for new iteration
        tol_c = mu^gamma;
        tol_p = theta*mu;
        set_optimizer_attributes(model, "mu_target" => mu, "mu_init" => mu,
                                        "dual_inf_tol" => tol_p, "constr_viol_tol" => tol_p,
                                        "compl_inf_tol" => tol_p, "print_level" => 0);
                                        # "compl_inf_tol" => tol_p);
        
        # solving the optimization problem                                
        optimize!(model);
        obj_val = objective_value(model);
        ans = value.(all_variables(model));
        time += solve_time(model);
        for i in 1:nc
            cval[i] = (Array(value.(cvar[i][1])), Array(value.(cvar[i][2])));
            min_val[nvc[i]:nvc[i+1]-1] = min.(cval[i][1], cval[i][2]);
        end
        compl = norm(min_val, Inf);

        # checking complementarity condition
        if compl <= tol_c
            @printf("%4.g %21s %.8e %.4e %2.2f %.1e, %.4f\n", k, termination_status(model), 
                    obj_val, compl, log10(mu), pi, time);
            if plot_data
                append!(pl, [[obj_val, mu, pi, compl]]);
            end
            # checking termination criteria
            if abs(obj_val - last_obj) < tol             
                println("\nIterations = ", k);
                println("Objective value = ", obj_val);
                println("MPCC complementarity = ", compl);
                println("pi = ", pi);
                println("tol_p = ", tol_p);
                println("tol_c = ", tol_c);
                println("Time = ", time);
                break 
            else
                last_obj = obj_val;
            end
            k += 1;
            mu = mu*kappa;
            set_start_value.(all_variables(model), ans);
        else
            pi *= sigma;
            @objective(model, Min, obj + pi * pen);
        end
    end
    if k > k_max
        println("Maximum number of outer iterations reached.");
        return -1
    end
    if plot_data
        return pl
    else
        return 0
    end
end







