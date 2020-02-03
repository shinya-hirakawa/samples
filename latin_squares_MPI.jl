import MPI
using Printf
function known_value()
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

function compensation_value()
    # 全換置換の正確な個数．
    recontres = [0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570, 176214841]
    est_val = (length(recontres) >= L) ? log(div(recontres[L], L-1)) : sum( log.(1:L) ) - 1 - log(L-1)
    return est_val
end

function make_cl(x)
    cl = reshape(zeros(Int64, L^2), L, L)
    for i in 1:(L-1)
        for j in 1:L
            cl[j,x[i,j]] += 1
        end
    end
    return cl
end

function make_x()
    # 初期化．初期状態はエネルギー0の状態を選ぶ．
    x = reshape(zeros(Int64, (L-1)*L), L-1, L)#L^2->(L-1)*L, L->L-1
    for i in 1:(L-1)
        for j in 1:L
            x[i,j] = ((i+j-2) % L) + 1
        end
    end
    return x
end

function calc_k(j::Int64, u::Int64, r::Int64)
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

function run_single_MCS(x::Array{Int64,2}, cl::Array{Int64,2}, e::Int64, en2b::Float64, en4b::Float64, K, Beta_unit)
    sumew = zero(Float64)
    for i in 2:L-1
        j = rand(2:L)
        u = x[i,j]
        k = calc_k(j,u,rand(2:L-2))
        v = x[i,k]
        if (v == j)#全換置換のみ実行
            sumew += exp(-Beta_unit*e)
            continue
        end
        e_diff  = (cl[j,v]>=1) - (cl[j,u]>=2) + (cl[k,u]>=1) - (cl[k,v]>=2)
        e_diff *= 2
        if ( e_diff > zero(Int64) ) 
            r = (e_diff==4) ? en4b : en2b
            if (r <= rand()) 
                sumew += exp(-Beta_unit*e)
                continue
            end
        end
        e += e_diff
        sumew += exp(-Beta_unit*e)
        x[i,j], x[i,k] = x[i,k], x[i,j]
        cl[j,u] -= 1
        cl[j,v] += 1
        cl[k,v] -= 1
        cl[k,u] += 1
    end
    sumew /= (L-2)
    return e, sumew
end

function run_unit_MCS(x, Beta, unit_MCS::Int64, K, Beta_unit)
    cl = make_cl(x)
    e = sum(abs.(cl .- 1)) - L 
    expn2b = exp( -2*Beta )
    expn4b = exp( -4*Beta )
    sumew = zero(Float64)
    for _ in 1:unit_MCS
        e, sumew_t = run_single_MCS(x, cl, e, expn2b, expn4b, K, Beta_unit)
        sumew += sumew_t
    end
    ew = sumew/unit_MCS
    return x, e, ew
end

function main(unit_MCS::Int64, burnin_num::Number, cumsum_num::Number, K, Beta_unit, class_num)

    comm = MPI.COMM_WORLD
    root = 0
    x = make_x()
    rank = MPI.Comm_rank(comm)
    cK = div(K,class_num)
    class = div(rank,cK) + 1
    ord = collect(1:K)
    ordinv = copy(ord)
    e_a = zeros(Int64,K)
    ew_a = zeros(Float64,K)
    ew_at = copy(ew_a)
    sumew_a = copy(ew_a)
    r_ew_a = copy(ew_a)
    t_ew_a = copy(ew_a)
    novio_a = [false for c in 1:class_num]
    for n in 1:burnin_num+cumsum_num
        while sum(novio_a)!=class_num
            if !novio_a[class]
                x, e, ew = run_unit_MCS(x, (ordinv[rank+1]-1)*Beta_unit, unit_MCS, K, Beta_unit)
            end
            e_a = MPI.Gather([e], root, comm)
            ew_a = MPI.Gather([ew], root, comm)
            if (rank==root)
                for c in 1:class_num
                    if novio_a[c] continue end
                    if (e_a[ord[c*cK]] == 0)
                        novio_a[c] = true
                        if (n > burnin_num)
                            for k in 1:K ew_at[k] = ew_a[ord[k]] end
                            # sumew_a += ew_at
                            r_ew_a += ew_at
                            t_ew_a  = sumew_a
                            sumew_a += r_ew_a
                            t_ew_a  = sumew_a - t_ew_a
                            r_ew_a -= t_ew_a
                            if (n % 10 == 0) m = n - burnin_num; show_result(sumew_a/m, m*unit_MCS, "CumuSum", stdout) end
                        else
                            if (n % 10 == 0) show_result(ew_a, n*unit_MCS, "Single ", stdout) end
                        end
                    end
                    ord, ordinv = replica_exchange(e_a, ord, ordinv, K, Beta_unit)
                end
            end
            MPI.Bcast!(novio_a, class_num, root, comm)
            MPI.Bcast!(ordinv, K, root, comm)
        end
        novio_a = [false for c in 1:class_num]
    end

end

function replica_exchange(e_a, ord, ordinv, K, Beta_unit)
    # for k in 1:K-1
        # ratio = exp( -Beta_unit*(e_a[ord[k]] - e_a[ord[k+1]]) )
        # if (ratio > rand())
            # ord[k], ord[k+1] = ord[k+1], ord[k]
            # ordinv[ord[k]], ordinv[ord[k+1]] = ordinv[ord[k+1]], ordinv[ord[k]]
        # end
    # end
    #20190905 偶奇レプリカ交換
    for k in 1:2:(K-1)
        ratio = exp( -Beta_unit*(e_a[ord[k]] - e_a[ord[k+1]]) )
        if (ratio > rand())
            ord[k], ord[k+1] = ord[k+1], ord[k]
            ordinv[ord[k]], ordinv[ord[k+1]] = ordinv[ord[k+1]], ordinv[ord[k]]
        end
    end
    for k in 2:2:(K-2)
        ratio = exp( -Beta_unit*(e_a[ord[k]] - e_a[ord[k+1]]) )
        if (ratio > rand())
            ord[k], ord[k+1] = ord[k+1], ord[k]
            ordinv[ord[k]], ordinv[ord[k+1]] = ordinv[ord[k+1]], ordinv[ord[k]]
        end
    end
    return ord, ordinv
end

function show_result(ew_a, total_MCS::Int64, str, fp)
    log_Z0 = sum( log.(1:L-1) ) + sum( log.(1:L) ) + (L-2)*compensation_value()
    log_ZK = log_Z0 + sum(log.(ew_a))
    diff = log_ZK - known_value()
    Rn = exp(sum(log.(ew_a)) + (L-2)*compensation_value())
    if str == "CumuSum"
        @printf fp "CumuSum : %.5f  Log ratio: %.5f  %d MCS  Rn: %.5e\n" log_ZK diff total_MCS Rn
    else
        @printf fp "Single  : %.5f  Log ratio: %.5f  %d MCS  Rn: %.5e\n" log_ZK diff total_MCS Rn
    end
end

# ラテン方陣の次数．
const L = 5
const Beta_max = 6.0 # betaの最大値
function universe()
    MPI.Init()
    comm = MPI.COMM_WORLD
    K = MPI.Comm_size(comm)
    Beta_unit = Beta_max/(K-1)

    class_num = 1
    unit_MCS = 1000
    burnin_num = 10
    cumsum_num = 10
    main(unit_MCS, burnin_num, cumsum_num, K, Beta_unit, class_num)
    unit_MCS = 10000
    burnin_num = 1000
    cumsum_num = 10000
    main(unit_MCS, burnin_num, cumsum_num, K, Beta_unit, class_num)

    MPI.Finalize()
end

universe()