using LaTeXStrings

@time begin

	begin
			using LinearAlgebra
            using Interpolations
			using ToeplitzMatrices
			using DSP
			using LaTeXStrings
			using Plots
			plotlyjs()
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

end

	## Qfilter
	u = [zeros(1000,1);1;zeros(1000,1)];
	fc = 1/6;
	a = digitalfilter(Lowpass(fc),Butterworth(2));
	Q1 = filtfilt(a,u);#Markov Parameter
	Qq = Toeplitz(Q1[1001:1001+24-1],Q1[1001:1001+24-1]);
	Q = kron(Qq,diagm(0=>ones(N)))
	# Q = diagm(0=>ones(N*24))
   

#################################################################

num_days = 15


struct demand_amp_var
	demand
end

function (dav::demand_amp_var)(t)
	index = Int(floor(t / (24*3600)))
	dav.demand[index + 1,:]
end



# slowly increasing and decreasing amplitude - only working for <= 10 days and N = 4 now
demand_amp1 = demand_amp_var(repeat([80 80 80 10 10 10 40 40 40 40 10 10 10 10 10 10], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp2 = demand_amp_var(repeat([10 10 10 80 80 80 40 40 40 40 10 10 10 10 10 10], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp3 = demand_amp_var(repeat([60 60 60 60 10 10 10 40 40 40 10 10 10 10 10 10], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp4 = demand_amp_var(repeat([30 30 30 30 10 10 10 80 80 80 10 10 10 10 10 10], outer=Int(N/4))') # random positive amp over days by 10%
demand_amp = t->vcat(demand_amp1(t), demand_amp2(t), demand_amp3(t), demand_amp4(t))

periodic_demand =  t-> demand_amp(t)./100 .* sin(t*pi/(24*3600))^2
samples = 24*4

inter = interpolate([.2 * randn(N) for i in 1:(num_days * samples + 1)], BSpline(Linear()))
residual_demand = t -> inter(1. + t / (24*3600) * samples) # 1. + is needed to avoid trying to access out of range


##################################################################################################################

d = zeros(N*24, num_days) # N*24 x num_days

for hours = 0:(24*num_days-1)

  d[Int(mod(hours,num_days))*N+1:(Int(mod(hours,num_days))+1)*N, Int(floor(hours/24)+1)] =  periodic_demand((hours+1)*3600) .+ residual_demand((hours+1)*3600)

end

y = zeros(N*24, num_days) # N*24 x num_days
u = zeros(N*24, num_days) # N*24 x num_days


u[:,1] = zeros(24*N)
y[:,1] = P*(-d[:,1] + u[:,1]) ./435
kappa = 1.0

for days = 2:num_days
    u[:,days] = Q*(u[:, days-1] + kappa * y[:,days-1])
    y[:,days] = P*(-d[:,days] + u[:,days]) ./435
end

d1 = d[1:4:end, :]
d1v = reshape(d1, 24*num_days,1)
d2 = d[2:4:end, :]
d2v = reshape(d2, 24*num_days,1)
d3 = d[3:4:end, :]
d3v = reshape(d3, 24*num_days,1)
d4 = d[4:4:end, :]
d4v = reshape(d4, 24*num_days,1)
dsum = d1v .+ d2v .+ d3v .+ d4v

u1 = u[1:4:end, :]
u1v = reshape(u1, 24*num_days,1)
u2 = u[2:4:end, :]
u2v = reshape(u2, 24*num_days,1)
u3 = u[3:4:end, :]
u3v = reshape(u3, 24*num_days,1)
u4 = u[4:4:end, :]
u4v = reshape(u4, 24*num_days,1)
usum = u1v .+ u2v .+ u3v .+ u4v

y1 = y[1:4:end, :]
y1v = reshape(y1, 24*num_days,1)
y2 = y[2:4:end, :]
y2v = reshape(y2, 24*num_days,1)
y3 = y[3:4:end, :]
y3v = reshape(y3, 24*num_days,1)
y4 = y[4:4:end, :]
y4v = reshape(y4, 24*num_days,1)
ysum = y1v .+ y2v .+ y3v .+ y4v

using Plots
plot(1:num_days*24,ysum)
plot!(1:num_days*24,dsum)
plot!(1:num_days*24,usum)
# working now -> but 435 steps inbetween should be added for the demand, otherwise it is too broad.

#plot!(1:num_days*24,y2v)
#plot!(1:num_days*24,y3v)
#plot!(1:num_days*24,y4v)