using Distributed
if length(ARGS) > 0
    machinefile = ARGS[1]
    mlist = collect(eachline(machinefile))
    addprocs([(mlist[1], 35), (mlist[2], 36)])
end

using Printf
@everywhere using Serialization
function known_value(L::Int64)
# ラテン方陣の正確な個数．
    Latin_squares = [1, 2, 12, 576, 161280, 812851200,  
        61479419904000, 
        108776032459082956800, 
        5524751496156892842531225600, 
        9982437658213039871725064756920320000, 
        776966836171770144107444346734230682311065600000];
    # Zhang, Maの推定値
    ZhangMa = log.([9.988, 7.773, 3.102, 7.500, 1.266, 1.728, 2.161, 2.804, 4.256, 8.354, 2.365, 5.67, 1.55]) + 
    [36, 47, 60, 74, 91, 109, 129, 151, 175, 201, 230, 2250, 11710]log(10);

    if (size(Latin_squares, 1) >= L)
        est_val = Float64( log(Latin_squares[L]) )
    elseif (10 <= L) && (L <= 20) 
        est_val = ZhangMa[L-9]
    elseif (L == 50)
        est_val = ZhangMa[end-1]
    elseif (L == 100)
        est_val = ZhangMa[end]
    else
        est_val = 0
    end
    return est_val
end

function compensation_value(L::Int64)
    # 全換置換の正確な個数．
    recontres = [0, 1, 2, 9, 44, 265, 1854,  
        14833, 133496, 1334961, 14684570, 176214841];
    if (length(recontres) >= L)
        est_val = log(div(recontres[L], L-1))
    else
        est_val = sum( log.(1:L) ) - 1 - log(L-1)
    end
    return est_val
end

function show_setting(settings, fp)
    #settings = [K, Beta_max, unit_MCS]
    K, Beta_max, unit_MCS = settings
    println(fp)
    println(fp, "Squares dim    : ", L)
    println(fp, "Process number : ", nprocs())
    println(fp, "Replica number : ", K)
    println(fp, "unit    MCS    : ", unit_MCS)
    println(fp, "Known   value  : ", known_value(L))
    println(fp)
end

@everywhere function make_cl(x)
    cl = reshape(zeros(Int64, L^2), L, L)
    for i in 1:(L-1)
        for j in 1:L
            cl[j,x[i,j]] += 1
        end
    end
    return cl
end

@everywhere function make_initial(dK::Int64, Beta_unit::Float64, pn::Int64)
    Beta_a   = (pn-1)*dK*Beta_unit .+ [dk*Beta_unit for dk in 0:dK-1]
    expn2b_a = exp.( -2*Beta_a )
    expn4b_a = exp.( -4*Beta_a )

    return expn2b_a, expn4b_a
end

function make_x_a(dK::Int64, proc_num::Int64)
    # 初期化．初期状態はエネルギー0の状態を選ぶ．
    x = reshape(zeros(Int64, (L-1)*L), L-1, L)#L^2->(L-1)*L, L->L-1
    for i in 1:(L-1)
        for j in 1:L
            x[i,j] = ((i+j-2) % L) + 1
        end
    end
    x_a = [deepcopy(x) for _ in 1:proc_num*dK]
    x_a = reshape(x_a, proc_num, dK)
    return x_a
end

function replica_exchange(x_a, e_a, dK::Int64, Beta_unit::Float64, proc_num::Int64)
    for pn in 1:proc_num
        for k in 1:dK-1
            ratio = exp( -Beta_unit*(e_a[pn,k] - e_a[pn,k+1]) )
            if (ratio > rand())
                 x_a[pn,k], x_a[pn,k+1] = x_a[pn,k+1], x_a[pn,k]
                 e_a[pn,k], e_a[pn,k+1] = e_a[pn,k+1], e_a[pn,k]
            end
        end
        if (pn == proc_num) break end
        ratio = exp( -Beta_unit*(e_a[pn,dK] - e_a[pn+1,1]) )
        if (ratio > rand())
             x_a[pn,dK], x_a[pn+1,1] = x_a[pn+1,1], x_a[pn,dK]
             e_a[pn,dK], e_a[pn+1,1] = e_a[pn+1,1], e_a[pn,dK]
        end
    end
end

@everywhere function calc_k(j::Int64, u::Int64, r::Int64)
    if (u==1)
        k = rand(2:L-1)
        if (k >= j) k += 1 end
    else
        k = r
        if (j > u) j,u = u,j end
        if (k >= j) k += 1 end
        if (k >= u) k += 1 end
    end
    return k
end

@everywhere function run_unit_MCS(x_a, dK::Int64, Beta_unit::Float64, 
        unit_MCS::Int64, p::Int64)

    expn2b_a, expn4b_a = make_initial(dK, Beta_unit, p)
    # 各列の各数の個数の配列
    cl_a = [make_cl(x_a[k]) for k in 1:dK]
    # エネルギーの配列
    e_a = [sum(abs.(cl_a[k] .- 1))-L for k in 1:dK]#add -L
    
    # 期待値計算のための配列．
    sum_a = zeros(Float64, dK)
    e_sum_a = zeros(Float64, dK)
    e2_sum_a = zeros(Float64, dK)
    # 最後のレプリカの制約violation数．
    violation = zero(Int64)
    r_a = reshape(zeros(Int64, (L-2)*2), L-2, 2)
    rmax = (L-1)*(L-3)-1
    
    for _ in 1:unit_MCS
        
        for l in 1:L-2
            dr = rand(0:rmax)
            k,j = divrem(dr,L-1)
            r_a[l,1] = j+2#rand(2:L)   #j+2# 0:L-2 -> 2:L
            r_a[l,2] = k+2#rand(2:L-2) #k+2# 0:L-4 -> 2:L-2
        end
        # 1番目からdK番目までのレプリカについてMCSを実行．
        for k in 1:dK
            e_t, sum_t, e_sum_t, e2_sum_t = run_single_MCS(x_a[k], cl_a[k], expn2b_a[k], expn4b_a[k], 
                r_a, e_a[k], Beta_unit)
            e_a[k] = e_t
            sum_a[k] += sum_t
            e_sum_a[k] += e_sum_t
            e2_sum_a[k] += e2_sum_t
        end
        if (e_a[dK] != 0) violation += 1 end

    end
    e_w_a = sum_a/(unit_MCS)
    e_sum_a /= unit_MCS
    e2_sum_a /= unit_MCS
    return x_a, e_a, e_w_a, e_sum_a, e2_sum_a, violation
end

# 状態変化のエネルギー変化を計算．
@everywhere sub_diff(x::Int64) = signbit(x-1) ? -1 : 1

# 全状態について状態変化試行を実行．
# 各行の成分二つをランダムに交換する．
@everywhere function run_single_MCS(x::Array{Int64,2}, cl::Array{Int64,2}, 
        en2b::Float64, en4b::Float64, 
        r_a::Array{Int64,2}, e::Int64, Beta_unit)
    sum = zero(Int64)
    e_sum = zero(Int64)
    e2_sum = zero(Int64)
    for i in 2:L-1
        j = r_a[i-1,1]#rand(2:L)#
        u = x[i,j]
        k = calc_k(j,u,r_a[i-1,2])#rand(2:L-2))#
        v = x[i,k]
        if (v == j)
            sum += exp(-Beta_unit*e)
            e_sum += e
            e2_sum += e*e
            continue
        end#全換置換のみ実行
        e_diff  = (cl[j,v]>=1) - (cl[j,u]>=2) + (cl[k,u]>=1) - (cl[k,v]>=2)
        e_diff *= 2
        if ( e_diff > zero(Int64) ) 
            r = (e_diff==4) ? en4b : en2b
            if (r <= rand()) 
                sum += exp(-Beta_unit*e)
                e_sum += e
                e2_sum += e*e
                continue
            end
        end
        e += e_diff
        sum += exp(-Beta_unit*e)
        e_sum += e
        e2_sum += e*e
        x[i,j], x[i,k] = x[i,k], x[i,j]
        cl[j,u] -= 1
        cl[j,v] += 1
        cl[k,v] -= 1
        cl[k,u] += 1
    end
    sum /= (L-2)
    e_sum /= (L-2)
    e2_sum /= (L-2)
    return e, sum, e_sum, e2_sum
end

function run_MCS(res, x_a, ew_a, e_sum_a, e2_sum_a, dK::Int64, Beta_unit::Float64, unit_MCS::Int64,
    exchange::String)
    proc_num = nprocs()
    violation_a = zeros(Int64, proc_num)
    e_a = reshape(zeros(Int64, proc_num*dK), proc_num, dK)
    for pn in 1:proc_num
        res[pn] = @spawn collect(run_unit_MCS(x_a[pn,:], dK, Beta_unit, unit_MCS, pn))
    end
    for pn in 1:proc_num
        x_a[pn,:], e_a[pn,:], ew_a[pn,:], e_sum_a[pn,:], e2_sum_a[pn,:], violation_a[pn] = fetch(res[pn])
    end
#     for pn in 1:proc_num
#         res[pn] = collect(run_unit_MCS(x_a[pn,:], dK, Beta_unit, unit_MCS, pn))
#     end
#     for pn in 1:proc_num
#         x_a[pn,:], e_a[pn,:], ew_a[pn,:], e_sum_a[pn,:], e2_sum_a[pn,:], violation_a[pn] = res[pn]
#     end
    skip_flag = false
    if (violation_a[end] > 0) 
        # @printf "skip violation! violation %d / %d MCS\n" violation_a[end] unit_MCS
        skip_flag = true
    end
    replica_exchange(x_a, e_a, dK, Beta_unit, proc_num)
    return skip_flag
end

function write_files(fname, ew_a, x_a, sum_ew_a, sum_e_sum_a, sum_e2_sum_a, n, unit_MCS, K)
    if (fname=="") return 0 end
    open(io -> serialize(io, x_a), "data/"*fname*"xa.dat", "w")
    open(io -> serialize(io, sum_ew_a), "data/"*fname*"sumewa.dat", "w")
    # open(io -> serialize(io, sum_e_sum_a), "data/"*fname*"sumesum.dat", "w")
    # open(io -> serialize(io, sum_e2_sum_a), "data/"*fname*"sume2sum.dat", "w")
    open(io -> serialize(io, n), "data/"*fname*"xa_num.dat", "w")
    # fp = open("data/"*fname*"burnin.txt", "a")
    # show_single_result(ew_a, (n)*unit_MCS, fp)
    # close(fp)
    #sum_ew_aをcumsum回数n(単位unit_MCS)で割る．K次ベクトルなので，-K*log(n)
    fp = open("data/"*fname*"result.txt", "a")
    show_cumsum_result(sum_ew_a, K, n, unit_MCS, fp)
    close(fp)
end

function read_files(fname, div_K, Beta_max, unit_MCS, exchange)
    proc_num = nprocs()
    K = div_K*proc_num
    x_a = make_x_a(div_K, proc_num)
    recover_flag = false
    if (fname != "" && isfile("data/"*fname*"xa.dat"))
        x_a_recover = open(deserialize, "data/"*fname*"xa.dat")
        if (size(x_a) == size(x_a_recover) && 
                size(x_a[1]) == size(x_a_recover[1]))
            x_a = x_a_recover
            # println("Read "*fname*"xa.dat")
            recover_flag = true
        else
            println("Default xa")
        end
    else
        println("Default xa")
    end
    settings = [K, Beta_max, unit_MCS, proc_num]
    prev_num = zero(Int64)
    sum_ew_a = reshape(zeros(Float64, K), proc_num, div_K)
    sum_e_sum_a = reshape(zeros(Float64, K), proc_num, div_K)
    sum_e2_sum_a = reshape(zeros(Float64, K), proc_num, div_K)
    if (recover_flag && fname != "" && isfile("data/"*fname*"settings.dat"))
        settings_recover = open(deserialize, "data/"*fname*"settings.dat")
        if (settings_recover == settings)
            prev_num = open(deserialize, "data/"*fname*"xa_num.dat")
            sum_ew_a = open(deserialize, "data/"*fname*"sumewa.dat")
            # sum_e_sum_a = open(deserialize, "data/"*fname*"sumesum.dat")
            # sum_e2_sum_a = open(deserialize, "data/"*fname*"sume2sum.dat")
            # println("We recover previous data")
        else
            recover_flag = false
        end
    end
    make_files(fname, settings, recover_flag)
    return x_a, sum_ew_a, sum_e_sum_a, sum_e2_sum_a, prev_num
end

function make_files(fname, settings, recover_flag)
    # if (fname != "" && recover_flag)
        # fp = open("data/"*fname*"result.txt", "a")
        # close(fp)
        # fp = open("data/"*fname*"burnin.txt", "a")
        # close(fp)
    # else
        # fp = open("data/"*fname*"result.txt", "w")
        # show_setting(settings, fp)
        # close(fp)
        # fp = open("data/"*fname*"burnin.txt", "w")
        # show_setting(settings, fp)
        # close(fp)
    # end
    # show_setting(settings, stdout)
    open(io -> serialize(io, settings), "data/"*fname*"settings.dat", "w")
end

function show_single_result(ew_a, unit_MCS::Int64, fp)
    knw_val = known_value(L)
    comp_val = compensation_value(L)
    # 全状態数のlog
    log_Z0  = sum( log.(1:L-1) ) + sum( log.(1:L) )
    log_Z0 += (L-2)*comp_val
    # ラテン方陣の個数のlogの推定値
    sum_log_we = sum( log.(ew_a) )
    log_ZK = log_Z0 + sum_log_we
    diff = log_ZK - knw_val
    @printf fp "Single  : %.5f  Log ratio: %.5f  %d MCS\n" log_ZK diff unit_MCS
end


function main(K::Int64, Beta_max::Float64, unit_MCS::Int64, 
        burnin_num::Number, cumsum_num::Number, fname::String, exchange)
    
    proc_num = nprocs()
    div_K = ceil(Int64, K/proc_num)
    K = div_K*proc_num
    Beta_unit = Beta_max/(K-1)
    
    res = Array{Any}(undef, proc_num)
    x_a, sum_ew_a, sum_e_sum_a, sum_e2_sum_a, prev_num = read_files(fname, div_K, Beta_max, unit_MCS, exchange)
    ew_a = reshape(zeros(Float64, K), proc_num, div_K)
    e_sum_a = reshape(zeros(Float64, K), proc_num, div_K)
    e2_sum_a = reshape(zeros(Float64, K), proc_num, div_K)
    # Beta_a   = reshape(zeros(Float64, K), proc_num, div_K)
    # for pn in 1:proc_num
        # Beta_a[pn,:] = (pn-1)*div_K*Beta_unit .+ [dk*Beta_unit for dk in 0:div_K-1]
    # end
    # open(io -> serialize(io, Beta_a), "data/"*fname*"Beta_a.dat", "w")

    n = 1
    while n <= burnin_num
        skip_flag = run_MCS(res, x_a, ew_a, e_sum_a, e2_sum_a, div_K, Beta_unit, unit_MCS, exchange)
        if (skip_flag) continue end
        if (n % 10 == 0) show_single_result(ew_a, unit_MCS, stdout) end
        n += 1
    end

    # cumulation sum
    n = 1
    while n <= cumsum_num
        skip_flag = run_MCS(res, x_a, ew_a, e_sum_a, e2_sum_a, div_K, Beta_unit, unit_MCS, exchange)
        if (skip_flag) continue end
        sum_ew_a += ew_a 
        sum_e_sum_a += e_sum_a
        sum_e2_sum_a += e2_sum_a
        if (n % 10 == 0) 
            write_files(fname, ew_a, x_a, 
                sum_ew_a, sum_e_sum_a, sum_e2_sum_a, 
                n+prev_num, unit_MCS, K) 
            # Cv = ((sum_e2_sum_a/(n+prev_num) - (sum_e_sum_a/(n+prev_num)).^2)).*(Beta_a.^2)
            # open(io -> serialize(io, Cv), "data/"*fname*"Cv.dat", "w")
        end
        n += 1
    end

end

function show_cumsum_result(sum_ew_a, K, n, unit_MCS::Int64, fp)
    total_MCS = n*unit_MCS
    knw_val = known_value(L)
    comp_val = compensation_value(L)
    # 全状態数のlog
    log_Z0  = sum( log.(1:L-1) ) + sum( log.(1:L) )
    log_Z0 += (L-2)*comp_val
    # ラテン方陣の個数のlogの推定値
    sum_log_we = sum( log.(sum_ew_a) ) - K*log(n)
    log_ZK = log_Z0 + sum_log_we
    diff = log_ZK - knw_val
    Rn = exp(sum_log_we + (L-2)*comp_val)
    
    @printf "CumuSum : %.5f  Log ratio: %.5f  %d MCS  Rn: %.5e\n" log_ZK diff total_MCS Rn
    @printf fp "CumuSum : %.5f  Log ratio: %.5f  %d MCS  Rn: %.5e\n" log_ZK diff total_MCS Rn
end

# ラテン方陣の次数．
@everywhere const L = 11

K = 144#L#div(L*(L+1),2) # レプリカの個数．
Beta_max = 6.0 # betaの最大値
unit_MCS = 1000
burnin_num = 10
cumsum_num = 10
@time main(K, Beta_max, unit_MCS, burnin_num, cumsum_num, "", "exchange");

unit_MCS = 10_000
burnin_num = 1000
cumsum_num = 10
for i in 1:1
    @time main(K, Beta_max, unit_MCS, burnin_num, cumsum_num, "sq"*string(L)*"K"*string(K)*"_", "exchange");
end
burnin_num = 0
cumsum_num = 1000
while 1 > 0
    @time main(K, Beta_max, unit_MCS, burnin_num, cumsum_num, "sq"*string(L)*"K"*string(K)*"_", "exchange");
end
