#=
L: Magnetic shell
c_0: Speed of light in unit of m/s
E_0: Rest mass energy in units of MeV
Re: Earth radius

Δt: Time step in units of s
trecord: Time points to record the PSD

boundaryTα and boundaryTξ: Boundary conditions along the α and ξ=ln(p) directions respectively; first element represents lower boundary and second element represents upper boundary; 0 represents fixed boundary condition and -1 represents equivalent extrapolation boundary condition

dcfile: File containing diffusion coefficients and grids
nα: Number of grid points along the pitch angle direction
np: Number of grid points along the momentum direction
α: Pitch angles in units of rad
p: Momentums dimensionlessed by mₑc
Dαα, Dαp, and Dpp: Pitch-angle, cross, momentum diffusion coefficients
Dαα, Dαp/p, and Dpp/p^2 in unites of 1/s

αₗ: Loss cone angle in units of rad
tau: A quarter of the bounce period in units of s
rtau: Reciprocal of tau inside the loss cone and zero outside the loss cone.
=#

using LinearAlgebra
using ShiftedArrays # lazy view of a shifted Array
using JLD2
using Markdown
using CairoMakie


#Read diffusion coefficients and grids
function readbd(file) 
    f = jldopen(file, "r")
	nα = f["nα"]
    np = f["np"]
	α = f["pitchangle"]
	p = f["momentum"]  # p has been normalized to mₑc
	Dαα = f["Dαα"]
    Dαp = f["Dαp"]
    Dpp = f["Dpp"]
	close(f)
    return nα, np, α, p, Dαα, Dpp, Dαp
end

#Central difference approximation for the first order derivatives
function cda(w2d, Δx, dim::Int64)
	if dim == 1
		w2dt = (ShiftedArray(w2d, (-1, 0), default = NaN64) - ShiftedArray(w2d, (1, 0), default = NaN64)) ./(2.0Δx)
 		w2dt[begin, :]=@view w2dt[begin+1, :]
		w2dt[end, :]=@view w2dt[end-1, :] 
	elseif dim == 2
		w2dt = (ShiftedArray(w2d, (0, -1), default = NaN64) - ShiftedArray(w2d, (0, 1), default = NaN64)) ./(2.0Δx)
 		w2dt[:, begin]=@view w2dt[:, begin+1]
		w2dt[:, end]=@view w2dt[:, end-1] 
	end
	return w2dt
end

#Central difference approximation for the second order mixed derivative
function cdam(w2d, Δx, Δy)
	w2dt = (ShiftedArray(w2d, (-1, -1), default = NaN64) - ShiftedArray(w2d, (1, -1), default = NaN64)- ShiftedArray(w2d, (-1, 1), default = NaN64) + ShiftedArray(w2d, (1, 1), default = NaN64)) ./(4.0*Δx*Δy)
 	w2dt[begin, :]=@view w2dt[begin+1, :]
	w2dt[end, :]=@view w2dt[end-1, :]
	w2dt[:, begin]=@view w2dt[:, begin+1]
	w2dt[:, end]=@view w2dt[:, end-1]
	return w2dt
end

@doc raw"""
    solve1d(a, b, c, d, w, nx, Δx, Δt, boundaryT)

	One-dimensional solver
	```math
		\frac{\partial w}{\partial t} = a \frac{\partial^2 w}{\partial x^2} + b \frac{\partial w}{\partial x} + c \left(\frac{\partial w}{\partial x}\right)^2 + d
	```	
"""
function solve1d(a, b, c, d, w, Δx, Δt, boundaryT)
	dd = @. -2.0a/Δx^2 - 1.0/Δt
	dl = similar(dd)
	du = similar(dd)
	bt = b .+ c .* (ShiftedArray(w, -1, default = NaN64) .- ShiftedArray(w, 1, default = NaN64)) ./(2.0Δx)

	for i in eachindex(bt)
		if bt[i]<0.0
			dd[i] = dd[i] + bt[i]/Δx
			dl[i] = a[i]/Δx^2 - bt[i]/Δx
			du[i] = a[i]/Δx^2
		else 
			dd[i] = dd[i] - bt[i]/Δx
			dl[i] = a[i]/Δx^2 
			du[i] = a[i]/Δx^2 + bt[i]/Δx
		end
	end
	B = @. (-w/Δt - d)

	#lower boundary
	if boundaryT[begin] == 0 #constant boundary condition
		dd[begin] = 1.0
		du[begin] = 0.0
		B[begin] = w[begin]
	elseif boundaryT[begin] == -1 #equivalence extrapolation boundary condition
		dd[begin] = 1.0
		du[begin] = -1.0
		B[begin] = 0.0
	end

	#upper boundary
	if boundaryT[end] == 0 #constant boundary condition
		dd[end] = 1.0
		dl[end] = 0.0
		B[end] = w[end]
	elseif boundaryT[end] == -1 #equivalence extrapolation boundary condition
		dd[end] = 1.0
		dl[end] = -1.0
		B[end] = 0.0
	end

	A = Tridiagonal(dl[begin+1:end], dd, du[begin:end-1])
	return A \ B
end

"""
    solve2d(αa2d, αb2d, αc2d, ξa2d, ξb2d, ξc2d, psd, Dαp, p2d, rtau, nα, np, Δα, Δξ, Δt)

	Determine the required inputs to solve1d, with the equivalence extrapolation boundary condition at α=0
"""
function solve2d(αa2d, αb2d, αc2d, ξa2d, ξb2d, ξc2d, d2d, psd, rtau, Δα, Δξ)

	#α-direction
	αd2d = d2d .* ( cda(psd, Δα, 1) .* cda(psd, Δξ, 2) .+  cdam(psd, Δα, Δξ)) .- rtau
	psdt = stack([solve1dα(a, b, c, d, w) for (a, b, c, d, w) in zip(eachcol(αa2d), eachcol(αb2d), eachcol(αc2d), eachcol(αd2d), eachcol(psd))])
	psdt[:, begin]=@view psd[:, begin]
	psdt[:, end]=@view psdt[:, end-1]
	
	#ξ-direction
	ξd2d = d2d .* ( cda(psdt, Δα, 1) .* cda(psdt, Δξ, 2) .+  cdam(psdt, Δα, Δξ))
	psdt = stack([solve1dξ(a, b, c, d, w) for (a, b, c, d, w) in zip(eachrow(ξa2d), eachrow(ξb2d), eachrow(ξc2d), eachrow(ξd2d), eachrow(psdt))])'
	psdt[begin, :]=@view psdt[begin+1, :]
	psdt[end, :]=@view psdt[end-1, :]
	
	#ξ-direction
	ξd2d = d2d .* ( cda(psdt, Δα, 1) .* cda(psdt, Δξ, 2) .+  cdam(psdt, Δα, Δξ))
	psdt = stack([solve1dξ(a, b, c, d, w) for (a, b, c, d, w) in zip(eachrow(ξa2d), eachrow(ξb2d), eachrow(ξc2d), eachrow(ξd2d), eachrow(psdt))])'
	psdt[begin, :]=@view psdt[begin+1, :]
	psdt[end, :]=@view psdt[end-1, :]

	#α-direction
	αd2d = d2d .* ( cda(psdt, Δα, 1) .* cda(psdt, Δξ, 2) .+  cdam(psdt, Δα, Δξ)) .- rtau
	psdt = stack([solve1dα(a, b, c, d, w) for (a, b, c, d, w) in zip(eachcol(αa2d), eachcol(αb2d), eachcol(αc2d), eachcol(αd2d), eachcol(psdt))])
	psdt[:, begin]=@view psd[:, begin]
	psdt[:, end]=@view psdt[:, end-1]
	return psdt
end



const L=4.5
const c_0 = 3e8 
const E_0 = 0.511
const Re = 6376e3 

const Δt = 20.0
const trecord = collect(0:10).*8640.0

const boundaryTα = [-1,-1]
const boundaryTξ = [0,-1]

const dcfile = "DiffusionCoefficients.jld2"
const nα, np, α, p, Dαα, Dpp, Dαp=readbd(dcfile)
const ξ = @. log(p)
const Δα = α[2]-α[1]
const Δξ = ξ[2]-ξ[1]

const αd = rad2deg.(α)
const γ = @. sqrt(1.0 + p^2)
const Ek = @. (γ - 1.0)*E_0
const Tα = (1.30.-0.56sin.(α))
const G =(Tα.*sin.(α).*cos.(α)) * (p.^2)' .+ 1e-20 
const αₗ = asin(L^(-3.0/2.0)*(4.0-3.0/L)^(-1.0/4.0))
const tau = Tα * (L*Re ./ (p./γ.*c_0))' 
const rtau = [ α[i]<αₗ ? 1.0/tau[i, j] : 0.0  for i in eachindex(α), j in eachindex(ξ)]

#Derived coefficients for the 2D diffusion equation
const p2d = repeat(p', nα, 1)
const αa2d = copy(Dαα)
const αb2d = cda(G .* Dαα, Δα, 1) ./ G  .+ cda(G .* Dαp, Δξ, 2) ./ G ./ p2d
const αc2d = copy(Dαα) 
const ξa2d = Dpp ./ p2d.^2
const ξb2d = cda(G .* Dpp ./ p2d, Δξ, 2) ./ G ./ p2d  .+ cda(G .* Dαp, Δα, 1) ./ G ./ p2d
const ξc2d = Dpp ./ p2d.^2
const d2d = Dαp ./ p2d

solve1dα(a, b, c, d, w) = solve1d(a, b, c, d, w, Δα, Δt, boundaryTα)
solve1dξ(a, b, c, d, w) = solve1d(a, b, c, d, w, Δξ, Δt, boundaryTξ)
solve2dαξ(psd) = solve2d(αa2d, αb2d, αc2d, ξa2d, ξb2d, ξc2d, d2d, psd, rtau, Δα, Δξ)
function fokkerplanck2d!(t, trecord, logpsdrecord)
	tind = 2
	tend = maximum(trecord)
	logpsd = logpsdrecord[1,:,:]

	while round(t; digits=1) < tend
		logpsd = solve2dαξ(logpsd)
		t = t + 2.0Δt
		if abs(round(t; digits=1)-trecord[tind])<2.0Δt
			logpsdrecord[tind,:,:] .= logpsd
			tind = tind+1
			@show t::Float64, maximum(logpsd)::Float64
		end	
	end
	return nothing
end

function plotflux(trecord, psdrecord, ptime)
	size_inches = (9, 12)
	size_pt = 72 .* size_inches
	fig=Figure(size = size_pt, fontsize = 15, figure_padding = (3,-30, 3, 10))
	gl = fig[1, 1] = GridLayout()
	gr = fig[1, 2] = GridLayout()

	axs = [Axis(gl[i, 1]
	,xlabel=rich(rich("α", font = :italic), " (Deg)")
	,ylabel=rich(rich("E",font = :italic), subscript("k"), " (MeV)")
	,xticklabelsvisible=false, yscale=log10, xticks=range(0, 90, step=30), yticks=[0.2,1,5]
	,xminorticksvisible = true, xminorticks = IntervalsBetween(3)
	,yminorticksvisible = true, yminorticks = IntervalsBetween(4)
	,limits=(0,90,0.2,5)
	) for i in eachindex(ptime)]

	colorrange=(1e-6,1e0)
	txt = ["(a)", "(b)", "(c)", "(d)"]
	for (i, pt) in enumerate(ptime)
		flux = psdrecord[pt, :, :] .* p2d.^2
		hm = heatmap!(axs[i], αd, Ek, flux, colorrange=colorrange, colorscale=log10, lowclip = :white)
		text!(axs[i],  0.0, 5.0, text=rich(txt[i], rich(" ∂f/∂α|",subscript("α=",rich("0",font = :regular),font = :italic),font = :italic),"=0 ",rich("t=",font=:italic), "$(trecord[pt]) s"," 90×80 grid ", rich("∆",rich("t",font = :italic),"=$Δt s" )),align=(:left,:top),offset=(4,-2))
	end
	
	linkxaxes!(axs[1:4]...)
	hidexdecorations!.(axs[1:3])

	Colorbar(gr[1:20,1], colorrange=colorrange, scale=log10, valign=:top,halign=:left,ticksize=5.0
	,label=rich("Differential Flux ",rich("j", font = :italic)," (arbitrary units)")
	,ticks=10.0 .^ range(-6, 0)
	,tickformat = values -> [rich("10",superscript("$(round(Int64, log10(value)))")) for value in values])
		

	elem_a=0.5
	elem_b=0.2
	elem_c=1.1
	elem_d=0.8
	elem_1= [PolyElement(color = :white, strokecolor = :black, strokewidth = 1.0, points = Point2f[(elem_a,elem_b), (elem_c, elem_b), (elem_c,elem_d),(elem_a,elem_d)])]

	Legend(gr[21,1],[elem_1],[rich(" <10",superscript("-6"))], framevisible = false,halign=:left,labelsize=15.0,padding=(-10,0,0,0), patchlabelgap = -1)
	fig
end

t=0.0
#initial condition
psd0 = sin.(α) * (exp.(-(Ek.-0.2) ./0.1) ./(p.^2))'#
maxpsd0 = maximum(psd0)
psd0 = psd0 ./ maxpsd0
psd0[begin,:] = psd0[begin+1, :]

logpsdrecord = zeros(length(trecord), length(α), length(ξ))
logpsdrecord[1,:,:] = log.(psd0)

fokkerplanck2d!(t, trecord, logpsdrecord)

psdrecord = exp.(logpsdrecord) .* maxpsd0
ptime = [0, 1, 5, 10] .+ 1

fig = plotflux(trecord, psdrecord, ptime)
save("FPdemo.pdf", fig)




#=
#Efforts to optimise Julia code
using BenchmarkTools
@btime fokkerplanck2d!(t, trecord, logpsdrecord)
@benchmark fokkerplanck2d!(t, trecord, logpsdrecord)
@code_warntype fokkerplanck2d!(t, trecord, logpsdrecord)

using JET
@report_opt fokkerplanck2d!(t, trecord, logpsdrecord)
=#
















 

