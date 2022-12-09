using LaTeXStrings

#@time begin

	begin
			using LinearAlgebra
			using ToeplitzMatrices
			using DSP
			using LaTeXStrings
			using Plots
			using FileIO
			using LightGraphs
			using Distributions
			using CSV
			using DataFrames
	end


	function communication_adj(exp_case, N, hr, graph)
		#graph = random_regular_graph(iseven(3N) ? N : (N-1), 3)
		vc23 = independent_set(graph, DegreeIndependentSet()) # ilc_nodes
		ilc_nodes23 = zeros(N)
		for i in 1:length(ilc_nodes23)
			if i ∈ vc23
				ilc_nodes23[i] = 1.
			end
		end
		if exp_case == 1
			adj = diagm(0=>ones(N*hr))
			print("Exp ", exp_case, ": Adjacency matrix: ", adj)
		elseif exp_case == 2
			cover2 = Dict([v => neighbors(graph, v) for v in vc23])
			adj = diagm(0=>ilc_nodes23)
			for i in keys(cover2)
				adj[i,cover2[i]] .= 1
			end
			adj *= 0.25 # needed?
			print("Exp ", exp_case, ": Adjacency matrix: ", adj)
		elseif exp_case == 3
			adj = diagm(0=>ilc_nodes23)
			print("Exp ", exp_case, ": Adjacency matrix: ", adj)
		elseif exp_case == 4
			ilc_ratio = Int(N/2)
			vc4 = sample(1:N,ilc_ratio,replace = false) # ilc_nodes
			adj =  diagm(0 => zeros(N))
			cover4 = [] # ilc_covers
			for i in 1:ilc_ratio
				adj[vc4[i],vc4[i]] = 1
				#kappa[few[1], vc[i]] = 1 # here multiple entries
				#kappa[few[2], vc[i]] = 1
				#kappa[few[3], vc[i]] = 1
				a = 1:vc4[i]-1
				b = vc4[i]+1:N
				c = [collect(a); collect(b)]
				few = sample(c,3,replace = false)
				adj[vc4[i], few[1]] = 1
				adj[vc4[i], few[2]] = 1
				adj[vc4[i], few[3]] = 1
				push!(cover4, Dict([vc4[i] => few]))
			end
			adj *= 0.25 # needed?
			print("Exp ", exp_case, ": Adjacency matrix: ", adj)
		else
			error("exp needs to be an integer between 1 and 4")
		end
		adj
	end


	nsteps = 25
	#N_range = 24:24
	#N = 24
	N_range = 4:4:52
	df_AS = zeros(nsteps,length(N_range)+1)
	df_MC = zeros(nsteps,length(N_range)+1)
	simu = false

	for (N_index, N) in enumerate(N_range)
			println("N = ", N)
			graph = random_regular_graph(iseven(3N) ? N : (N-1), 3) # change last "3" to 1 for N=2

				## Independent parameters
				#syms k
				Yfactor = 1
				l = 435#435*Int(Yfactor)#*10000 # If M is scaled with C, l needs to be scaled with C^2
				h = 1/l;#h 0.0043
				Day = l*24
				#N = 24

				# Substitute all the parameters

				M = repeat([5],inner = N) #[5. 4.8 4.1 4.8] #repeat([5],inner = N) #./ 100;# 5
				T =  repeat([0.05],inner = N) #[0.04 0.045 0.047 0.043] #repeat([0.05],inner = N) #./ 10;# 0.05
				V =  repeat([1],inner = N);
				kp = repeat([525],inner = N)#[400 110 100 200] #repeat([525],inner = N);# 525
				ki = repeat([0.005],inner = N)#[0.05 0.004 0.05 0.001] # repeat([0.005],inner = N);# 0.005
				Yadj = adjacency_matrix(graph)
				Y  = Yfactor .* 6. .* Yadj#(ones(N,N)-I);# Y full matrix
				FF = zeros(N)
				A = zeros(3*N,3*N)
				Aa = zeros(3,3)
				AA = zeros(3,3)
				# # # Y square matrix
				#Y = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0];
				#
				# # Y line matrix
				#Y = [0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0];

				## _____-- A Matrix--____##


				for i = 1: N
				    i
				    for j = 1: N
				        j
				        if (i == j)#if it's on the diagonal
				            sum = 0
				            for c = 1:N #coupling part in the diagonal of matrix
				                sum = sum +V[j]*V[c]*Y[j,c] ;
				            end
				            FF[j] = -sum + V[j]*V[j]*Y[j,j];#sign

				            AA = [0 1 0;1/M[j]*FF[j] -kp[j]/M[j] 1/M[j]; 0 -1/T[j] -ki[j]/T[j]];
				            A[(i-1)*3+1:i*3,(j-1)*3+1:j*3] = AA;
				        else
				            Aa = [0 0 0;V[i]*V[j]*Y[i,j]*1/M[j] 0 0;0 0 0];
				            A[(i-1)*3+1:i*3,(j-1)*3+1:j*3] = Aa;
				        end

				    end
				end



				## _____-- B Matrix--____##

				B = zeros(3*N,N);
				BB = zeros(3,1);
				for i = 1:N
				    i
				    BB=[0;1/M[i];0];
				    B[(i-1)*3+1:i*3,i] = BB;
				end




				## _____-- C Matrix--____##

				C  =zeros(N,3*N);
				CC = zeros(N,3);

				for i = 1:N
				    i

				    CC  =[0,-kp[i],1];
				    C[i,(i-1)*3+1:i*3]= CC;

				end

				eigvals(A)
				## _____-- Discretization--____##



				Ad = exp(A*h)#symsum(1/factorial(k)*(A*h)^k,k,0,5);#Ad=e^(h*A)
				#Ad = double(Ad);
				ord =10
				sumexp = zeros(ord+1,3*N,3*N)
				for i = 0:ord
					sumexp[i+1,:,:] = 1/factorial(i+1)*(A*h)^i
				end

				B_3 = reshape(h*sum(sumexp,dims=1),3*N,3*N)'#h*symsum(1/factorial(k+1)*(A*h)^k,k,0,5);
				# (I - expm(A*h)) * inv(-A) * B
				Bd = B_3*B
				#Bd = double(Bd);
				## P matrix
				#Ad = double(Ad);
				#Bd = double(Bd);

				## A_pr is symsum(Ad^(235-k),k,1,235)
				A_pr = 0
				A_pr_s = zeros(l,3*N,3*N)
				for i = 1:l
					A_pr_s[i,:,:] = Ad^(l-i)
				end
				A_pr = reshape(sum(A_pr_s,dims = 1),3*N,3*N)

				## p(h) = [p1 p2 ... p24]
				p = zeros(N,24*N)

				for h = 1:24 # something goes terribly wrong in this loop!!!!
					    h
					    if h == 1
					        sum1 = zeros(3*N,3*N);
					        A_h1 = Ad^0;
					        for j = 1:l
					            j
					                sum1 = sum1 + (l+1-j)*A_h1;
					                A_h1 = A_h1 *Ad;
					        end
					        p[1:N,1:N]= C*sum1*Bd;

					    else
					        #h>2
					        sum2 = zeros(3*N,3*N);
					        A_h2 = Ad^(l*(h-1)+1-l);
					        for i = 1:l
					            i
					            sum2 = sum2 + A_h2;
					            A_h2 = A_h2*Ad;
					        end
					        p[1:N,(h-1)*N+1:h*N]= C*sum2*A_pr*Bd;
					    end
				end

				# for i=1:2
				# 	m = 0
				# 	println("i ", i)
				# 	println("m ", m)
				# 	b = i
				# 	println("b ", b)
				# 	for f=1:10
				# 		println("f ", f)
				# 		m = m+b
				# 		println("m ", m)
				# 		b = b*5
				# 		println("b ", b)
				# 	end
				# end

				## Toeplitz(p)
				PP = zeros(N,24*N)
				for i = 1:24
				    i
				    PP[1:N,(i-1)*N+1:i*N]= p[1:N,(24-i)*N+1:(25-i)*N];
				end

				P = zeros(N*24,N*24);
				for j  = 1:24
				    j
				     P[((j-1)*N+1):N*j,1:N*j] = PP[1:N,((24-j)*N+1):24*N];
				end

				## Comparison with MATLAB solutions
				# using MAT
				# dir = @__DIR__
				# file = matopen("$dir/../Matlab_matrices/data/ABCP_N4_Yfull.mat")
				# Pmat = read(file, "P")  #note that this does NOT introduce a variable ``varname`` into scope
				# Amat = read(file, "A")
				# Bmat = read(file, "B")
				# Mmat = read(file, "M")
				# Cmat = read(file, "C")
				# Admat = read(file, "Ad")
				# Bdmat = read(file, "Bd")
				# A_prmat = read(file, "A_pr")
				# PPmat = read(file,"PP")
				# close(file)




				## Qfilter
				u = [zeros(1000,1);1;zeros(1000,1)];
				fc = 1/6;
				a = digitalfilter(Lowpass(fc),Butterworth(2));
				Q1 = filtfilt(a,u);#Markov Parameter
				Qq = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);
				Q = kron(Qq,diagm(0=>ones(N)))
				# Q = diagm(0=>ones(N*24))

				## Plotting max(eig(Q*(I-PL))) with L in different kappa
				#exp_case = 1
		for exp_case in 1:1
				i = 0;
				# nsteps= 25;
				kappa = zeros(nsteps)
				hr = 24
				d1_N1 = zeros(nsteps)
				d1_N2 = zeros(nsteps)
				mo = zeros(nsteps)
				adja = communication_adj(exp_case, N, hr, graph)
				if exp_case != 1
					adjac = zeros(N*hr,N*hr)
					for i in 1:hr
						adjac[(i-1)*N+1:i*N,(i-1)*N+1:i*N] = adja
					end
				else
					adjac = adja
				end
				#P = Pmat
				for a = 1:nsteps
				    println(a)
				    kappa[a] = -(a-1)*2/(nsteps*l) ;
				   # KK= repeat([kappa[a]],inner =hr*N);
				    K1 = kappa[a] .* adjac # # diagm(0=>KK) # diagonal kappa
				    check1_N1 = Q*(I-K1*P);
				    d1_N1[a] = maximum(abs.((eigvals(check1_N1))));
				    mo[a] = maximum(svdvals(P*Q/P*(I-P*K1)));
					#print("\n P = ", P)
				end

			df_AS[:,1] = -kappa*l
			df_AS[:,N_index+1] = d1_N1
			df_MC[:,1] = - kappa*l
			df_MC[:,N_index+1] = mo

			if length(N_range) == 1
				## Plot AS and Mono
				x = -kappa*l
				y1 = d1_N1;
				y2=1*ones(size(x));
				y3 = mo;
				# MATLAB: p=find(y1==min(y1));
				# MATLAB: q=find(y3==min(y3));
				using LaTeXStrings
				plot(x,y2, linewidth=3,linestyle = :dot, label = "Upper bound", legend=:bottomright, margin=3Plots.mm, xaxis=(L"\kappa [h^{-1}]", font(14)), xtickfontsize=18,legendfontsize=11, yaxis=("Max. eigenvalue or singular value", font(14)),ytickfontsize = 18) #  xticks = (0:0.25:2, string.(0:0.25:2))
				plot!(x,y1, linewidth=3, label = L"$\rho \, (Q(I-LP))$") # : max(|eig(Q(I-PL))|)
				plot!(x,y3,linestyle=:dash, linewidth=3, label = L"$\bar \sigma \, (PQP^{-1}(I-PL))$");# max(sv(PQ/P(I-PL))
				# MATLAB:text(x(p),y1(p),[num2str(x(p))],'color','k');
				# MATLAB:text(x(q),y3(q),[num2str(x(q))],'color','k');
				ylims!(0,1.5)
				xlims!(0,2.)
				title!("Asymptotic stability & monotonic convergence\n (N = $N, scenario $(exp_case)) \n") #sampling per hour: $l)
				dir = @__DIR__
				using Dates
				date = Dates.now()
				savefig("$dir/plots/$(date)_ewplot_$(Yfactor)_$(l)_exp_$(exp_case)_N$(N)_Q.png")
			end


						if simu
							num_days = 10

							struct demand_amp_var
								demand
							end

							function (dav::demand_amp_var)(t)
								index = Int(floor(t / (24*3600)))
								dav.demand[index + 1,:]
							end
							using Interpolations

							demand_amp1 = demand_amp_var(60 .+ rand(num_days+1,Int(N/4)).* 40.)
							demand_amp2 = demand_amp_var(70 .+ rand(num_days+1,Int(N/4)).* 30.)
							demand_amp3 = demand_amp_var(80 .+ rand(num_days+1,Int(N/4)).* 20.)
							demand_amp4 = demand_amp_var(90 .+ rand(num_days+1,Int(N/4)).* 10.)
							demand_amp = t->vcat(demand_amp1(t), demand_amp2(t),demand_amp3(t),demand_amp4(t))


							# demand_amp1 = demand_amp_var(90 .+ rand(num_days+1,Int(N/4)).* 10.)
							# demand_amp = t->vcat(demand_amp1(t),demand_amp1(t),demand_amp1(t),demand_amp1(t))

							periodic_demand =  t-> demand_amp(t)./100 .* sin(t*pi/(24*3600))^2
							samples = 24*4



							inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
							#inter = interpolate([.05 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
							residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range

							using Plots
							using LaTeXStrings
							dd = t->((periodic_demand(t) .+ residual_demand(t)))
							# plot(0:num_days*l_day, t -> (dd(t)[1] .+ dd(t)[2] .+ dd(t)[3] .+ dd(t)[4]), alpha=0.2, label = latexstring("\$P^d_j\$"),linewidth=3, linestyle=:dot)


							D = zeros(N*24, num_days)
							for i = 1:num_days
								for j = 1:24
										D[(j-1)*N+1:j*N,i]  = dd(((i-1)*24 + j)*3600)'
								end
							end
							#plot(sum(reshape(D,N,24*num_days)',dims=2))

							u = zeros(24*N, num_days+2)
							y = zeros(24*N, num_days)
							kap = 0.75 /l
							#Kvec = repeat([kap],inner =24*N)
							K = kap .* adjac#diagm(0=>Kvec)
							for i = 1:num_days
								y[:,i] = P * (u[:,i] - D[:,i])
								#println(y[1,i])
								u[:,i+1] = Q *  (u[:,i] + K*y[:,i])
								#println(u[1,i+1])
							end
							ym = reshape(y,N,24*num_days)'
							um = reshape(u,N,24*(num_days+2))'
							aa = 0
							bb = 10
							plot(sum(reshape(D,N,24*num_days)'[:,1],dims=2), label= "demand", lw = 3)
							plot!(aa*24+1:bb*24,ym[aa*24+1:bb*24,1]./435, label = "y",
								xaxis = ("hours", font(14)),
								yaxis = ("y, u, demand",font(14)),
								xtickfontsize=14,
								ytickfontsize=14, #legend = false,
								 lw = 3)
							plot!(aa*24+1:bb*24,um[aa*24+1:bb*24,1], label = "u", lw = 3)
							dir = @__DIR__
							using Dates
							date = Dates.now()
							savefig("$dir/plots/$(date)_N$(N)_learning_one_node_exp_$(exp_case)_Q.png")


							if N == 4
								plot(aa*24+1:bb*24, sum(reshape(D,N,24*num_days)'[aa*24+1:bb*24,1],dims=2) + sum(reshape(D,N,24*num_days)'[aa*24+1:bb*24,2],dims=2) + sum(reshape(D,N,24*num_days)'[aa*24+1:bb*24,3],dims=2) + sum(reshape(D,N,24*num_days)'[aa*24+1:bb*24,4],dims=2), label= "demand", lw = 3)
								plot!(aa*24+1:bb*24,(ym[aa*24+1:bb*24,1] + ym[aa*24+1:bb*24,2] + ym[aa*24+1:bb*24,3] + ym[aa*24+1:bb*24,4])./435, label = "y",
								xaxis = ("hours", font(14)),
								yaxis = ("y, u, demand",font(14)),
									xtickfontsize=14,
									ytickfontsize=14, #legend = false,
									 lw = 3)
								plot!(aa*24+1:bb*24,um[aa*24+1:bb*24,1] + um[aa*24+1:bb*24,2] + um[aa*24+1:bb*24,3] +um[aa*24+1:bb*24,4], label = "u", lw = 3)
								savefig("$dir/plots/$(date)_N$(N)_learning_sum_Q.png")
							end


							# only for homogeneous zero, otherwise sum is necessary
							#plot(aa*24+1:bb*24,um[aa*24+1:bb*24,1]+ym[aa*24+1:bb*24,1]./435-sum(reshape(D,N,24*num_days)'[:,1],dims=2), label= "balance sum", lw = 3)

						end

		end # exp_case

	end # N_range

	if length(N_range) > 1
		dir = @__DIR__
		using Dates
		using LaTeXStrings
		using Plots
		using ColorSchemes
		date = Dates.now()
		CSV.write("$dir/data/AS_N_kappa_Q.csv", DataFrame(df_AS))
		CSV.write("$dir/data/MC_N_kappa_Q.csv", DataFrame(df_MC))

		AS_N_kappa = CSV.read("$dir/data/AS_N_kappa_Q.csv")
		MC_N_kappa = CSV.read("$dir/data/MC_N_kappa_Q.csv")


		AS_z = convert(Matrix,AS_N_kappa[:,2:end])'

		heatmap(
			AS_N_kappa.x1,
			collect(N_range),
			AS_z,
			xaxis = (L"\kappa\, [h^{-1}]", font(14)),
			yaxis = ("number of nodes N", font(14)),
			colorbar = :right,
			c = :matter,
			title = L"$\rho \, (Q(I-LP))$",
			legend=:bottomleft,
			clims = (0, 2),
			xlims = (0, 2),
			size = (600, 400),
			dpi=300,
			margin=3Plots.mm,
			ytickfontsize=14,
			xtickfontsize=14
		)
		plot!(AS_N_kappa.x1,
			collect(N_range),
			AS_z,
			levels=[0.25, 0.5, 0.75, 1.],
			color=:black,
			contour_labels=true,
			)
		savefig("$dir/plots/$(date)_AS_kappa_N_Q.png")

		MC_z = convert(Matrix,MC_N_kappa[:,2:end])'
		heatmap(MC_N_kappa.x1,
				N_range,
				MC_z,
				xlabel = L"\kappa\, [h^{-1}]",
				ylabel = "number of nodes N",
				colorbar = :right,
				c = :matter,
				title = L"$\bar \sigma \, (PQP^{-1}(I-PL))$",
				legend=:bottomleft,
				clims = (0, 2),
				xlims = (0, 2),
				size = (600, 400),
				dpi=300,
				margin=3Plots.mm,
				ytickfontsize=14,
				xtickfontsize=14
			)
		plot!(MC_N_kappa.x1,
			collect(N_range),
			MC_z,
			levels=[0.25, 0.5, 0.75, 1.],
			color=:black,
			contour_labels=true
			)
		savefig("$dir/plots/$(date)_MC_kappa_N_Q.png")
	end

#end # @time begin
