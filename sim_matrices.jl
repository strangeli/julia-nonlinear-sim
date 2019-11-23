@time begin

	begin
			using LinearAlgebra
			using ToeplitzMatrices
			using DSP
			using LaTeXStrings
			using Plots
			using FileIO
	end


	## Independent parameters
	#syms k
	Yfactor = 1.
	l = 435*Int(Yfactor)#*10000 # If M is scaled with C, l needs to be scaled with C^2
	h = 1/l;#h 0.0043
	Day = l*24
	N = 4

	# Substitute all the parameters

	M = [5. 4.8 4.1 4.8] #repeat([5],inner = N) #./ 100;# 5
	T = [0.04 0.045 0.047 0.043] #repeat([0.05],inner = N) #./ 10;# 0.05
	V = repeat([1],inner = N);# 1
	kp = [400 110 100 200] #repeat([525],inner = N);# 525
	ki = [0.05 0.004 0.05 0.001] # repeat([0.005],inner = N);# 0.005
	Y  = Yfactor .* 6. .* (ones(N,N)-I);# Y full matrix
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

	## Plotting max(eig(Q*(I-PL))) with L in different kappa
	i = 0;
	nsteps= 800;
	kappa = zeros(nsteps)
	hour = 24
	d1_N1 = zeros(nsteps)
	d1_N2 = zeros(nsteps)
	mo = zeros(nsteps)
	#P = Pmat
	for a = 1:nsteps
	    println(a)
	    kappa[a] = -(a-1)*2/(nsteps*l) ;
	    KK= repeat([kappa[a]],inner =hour*N);
	    K1 = diagm(0=>KK);#diagnol kappa
	    check1_N1 = Q*(I-P*K1);
	    d1_N1[a] = maximum(abs.((eigvals(check1_N1))));
	    mo[a] = maximum(svdvals(P*Q/P*(I-P*K1)));
	end

#begin
	using LaTeXStrings
	## Plot AS and Mono
	x = -kappa*l
	y1 = d1_N1;
	y2=1*ones(size(x));
	y3 = mo;
	# MATLAB: p=find(y1==min(y1));
	# MATLAB: q=find(y3==min(y3));
	plot(x,y2, linewidth=3,linestyle = :dot, label = "Upper bound", legend=:topleft, margin=3Plots.mm, xaxis=(L"\kappa [h^{-1}]", font(14)), xtickfontsize=18,legendfontsize=11, yaxis=("Max. eigenvalue or singular value", font(14)),ytickfontsize = 18) #  xticks = (0:0.25:2, string.(0:0.25:2))
	plot!(x,y1, linewidth=3, label = "AS") # : max(|eig(Q(I-PL))|)
	plot!(x,y3,linestyle=:dash, linewidth=3, label = "MC");# max(sv(PQ/P(I-PL))
	# MATLAB:text(x(p),y1(p),[num2str(x(p))],'color','k');
	# MATLAB:text(x(q),y3(q),[num2str(x(q))],'color','k');
	ylims!(0,1.5)
	#title!("Asymptotic stability & monotonic convergence\n (N = 4, sampling per hour: $l)\n")
	dir = @__DIR__
	using Dates
	date = Dates.now()
	savefig("$dir/plots/$(date)_ewplot_julia_$(Yfactor)_$(l)_hetero.png")
end
