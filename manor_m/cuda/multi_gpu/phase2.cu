// Reuse the complete original ABI and unchanged production CUDA kernels in
// this separate translation unit/library. No single-GPU source is modified.
#include "../core/manor_m_phase2_port.cu"
#include "phase2.h"
#include "../core/manor_m_index_validation.h"
#include <thread>
#include <exception>
#include <mutex>
#if MANOR_M_MULTI_CACHED_GRAM
#include "multi_cached_readout.cuh"
#endif

namespace {
thread_local std::string multi_error;
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
void require_multi(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void check_multi(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
size_t multi_fixed_bytes(int N,int D,int MM,int MY,int M) {
    return 4*(size_t(N)*((3+2*MM)*D+MY+1)+size_t(M)*(D*std::max(MY,MM+1)+D+4))
        +size_t(N)*5+size_t(M)*5+4096;
}
size_t multi_readout_bytes(int N,int D,int MM,int MY) {
    size_t scratch=128*1024*1024;
#if MANOR_M_MULTI_PERSISTENT_READOUT
    scratch=std::max(scratch,size_t(65536)*4*(D+MY+MM*D+9*(MY+4)+36)+16*1024*1024);
#endif
    return 4*size_t(N)*(MM+1)+scratch;
}
template<class F> int multi_call(F&& f) {
    try { f(); multi_error.clear(); return 0; }
    catch (const std::exception& e) { multi_error=e.what(); return 1; }
}
void multi_finish_readout(const float* points,int Q,int S,int MY,
    int* zero,int* one,float* sums,float* coords,int initial,
    float* mse,float* coordinate,float* tangent) {
    manor_eval_fixed_order_sum_kernel<<<S*(MY+4),manor_eval_sum_threads>>>(points,Q,MY,sums,coords,zero);
    if(!initial){
        owner_scale_eval_history_kernel<<<(S*4+255)/256,256>>>(sums,S*4,1,Q,one);
        owner_scale_eval_history_kernel<<<(S*MY+255)/256,256>>>(coords,S*MY,1,Q,one);
    }
    check_multi(cudaGetLastError());std::vector<float> raw(size_t(S)*4);
    check_multi(cudaMemcpy(raw.data(),sums,raw.size()*4,cudaMemcpyDeviceToHost));
    if(coordinate)check_multi(cudaMemcpy(coordinate,coords,size_t(S)*MY*4,cudaMemcpyDeviceToHost));
    const float divisor=float(Q);
    for(int i=0;i<S;++i){
        if(mse)mse[i]=initial?raw[size_t(i)*4]/divisor:raw[size_t(i)*4];
        if(tangent)tangent[i]=initial?raw[size_t(i)*4+3]/divisor:raw[size_t(i)*4+3];
    }
    if(initial&&coordinate)for(size_t i=0;i<size_t(S)*MY;++i)coordinate[i]/=divisor;
}

__global__ void pack_multi_state(const int* owners, int first, int count,
    int D, int MM, const float* tan, const float* cen, float* packed) {
    const int q=first+blockIdx.x;
    if (blockIdx.x>=count) return;
    const int i=owners[q], width=(MM+1)*D;
    for (int e=threadIdx.x;e<width;e+=blockDim.x)
        packed[size_t(q)*width+e]=e<MM*D
            ? tan[size_t(i)*MM*D+e] : cen[size_t(i)*D+e-MM*D];
}
__global__ void unpack_multi_state(const int* owners, int count, int D,
    int MM, const float* packed, float* tan, float* cen) {
    const int q=blockIdx.x;
    if (q>=count) return;
    const int i=owners[q], width=(MM+1)*D;
    for (int e=threadIdx.x;e<width;e+=blockDim.x) {
        const float v=packed[size_t(q)*width+e];
        if (e<MM*D) tan[size_t(i)*MM*D+e]=v;
        else cen[size_t(i)*D+e-MM*D]=v;
    }
}

struct MultiDevice {
    int device=0, first=0, count=0, tile=0, agg_tile=0;
    bool cached=false;
    std::unique_ptr<OwnerIndexTiles> layout;
    std::unique_ptr<OwnerIndexStream> stream;
    OwnerCudaBuffer<float> x,y,ta,tb,ca,cb,exchange,pre,weight,cacc,cov;
    OwnerCudaBuffer<float> readout_gram;
    size_t gram_budget=0, gram_bytes=0;
    int gram_iteration=-1;
    struct Readout {
        OwnerCudaBuffer<float> qx,qy,qt,hs,rs,xnorm,tdot,points,sums,coords,minq;
        OwnerCudaBuffer<int> pairs,pair_count,zero,one;
        int stats_iteration=-1;
        void allocate(int N,int D,int MM,int MY) {
            const size_t Q=65536,S=9;
            qx.allocate(Q*D,"multi query X");qy.allocate(Q*MY,"multi query Y");
            qt.allocate(Q*MM*D,"multi query tangent");hs.allocate(9,"multi readout h");rs.allocate(9,"multi readout rho");
            xnorm.allocate(N,"multi train norms");tdot.allocate(size_t(N)*MM,"multi tangent dot X");
            points.allocate(Q*S*(MY+4),"multi query contributions");sums.allocate(S*4,"multi metric sums");
            coords.allocate(S*MY,"multi coordinate sums");minq.allocate(Q*S*2,"multi sparse minq");
            pairs.allocate(std::max(Q*S*2,Q*(S+2)+1),"multi sparse pairs");pair_count.allocate(1,"multi sparse count");
            zero.allocate(1,"multi readout active");one.allocate(1,"multi readout valid");
            check_multi(cudaMemset(zero.get(),0,4));const int valid=1;
            check_multi(cudaMemcpy(one.get(),&valid,4,cudaMemcpyHostToDevice));
        }
    } readout;
    OwnerCudaBuffer<int> owners,lookup,index,agg_count,heavy_count,keep_count;
    OwnerCudaBuffer<int> stopped,valid;
    OwnerCudaBuffer<int32_t> counts,last;
    OwnerCudaBuffer<uint8_t> owner_mask,heavy,heavy_last,keep,keep_last,failed;
    OwnerCudaBuffer<uint32_t> mask;
    OwnerCudaBuffer<unsigned long long> sum,positive_sum;
    OwnerCudaBuffer<int> positive_count;
    float *tan=nullptr,*cen=nullptr,*tan_next=nullptr,*cen_next=nullptr;
    uint64_t index_upload=0, pinned=0;
    ~MultiDevice() {
        cudaSetDevice(device);
        // Members are released while this device remains current.
    }
};
}

struct ManorMMultiGpu {
    Phase2Shape shape;
    ManorMCudaPhase2Options options;
    ManorMMultiMemory memory;
    const int32_t* host_index=nullptr;
    std::vector<int> owner_host;
    std::vector<std::unique_ptr<MultiDevice>> devices;
    std::vector<std::vector<bool>> peer;
    void* relay=nullptr;
    static constexpr size_t relay_capacity=16*1024*1024;
    ManorMMultiStats stats{};
    bool poisoned=false;
    ~ManorMMultiGpu() {
        devices.clear();
        if (relay) cudaFreeHost(relay);
    }

    template<class F> void parallel(F f) {
        std::exception_ptr error;
        std::mutex mutex;
        std::vector<std::thread> threads;
        for (auto& ptr:devices) {
            MultiDevice* d=ptr.get();
            threads.emplace_back([&,d] {
                try { check_multi(cudaSetDevice(d->device)); f(*d); }
                catch (...) { std::lock_guard<std::mutex> lock(mutex);
                    if (!error) error=std::current_exception(); }
            });
        }
        for (auto& t:threads) t.join();
        if (error) { poisoned=true; std::rethrow_exception(error); }
    }

    template<class Getter> void gather(Getter get, size_t row_bytes) {
        for (size_t s=0;s<devices.size();++s) {
            auto& src=*devices[s];
            const size_t offset=size_t(src.first)*row_bytes;
            const size_t bytes=size_t(src.count)*row_bytes;
            for (size_t t=0;t<devices.size();++t) {
                if (s==t) continue;
                auto& dst=*devices[t];
                char* from=reinterpret_cast<char*>(get(src))+offset;
                char* to=reinterpret_cast<char*>(get(dst))+offset;
                if (peer[t][s]) {
                    check_multi(cudaSetDevice(dst.device));
                    check_multi(cudaMemcpyPeerAsync(to,dst.device,from,
                        src.device,bytes,nullptr));
                    stats.peer_bytes+=bytes;
                } else {
                    for (size_t k=0;k<bytes;k+=relay_capacity) {
                        const size_t n=std::min(relay_capacity,bytes-k);
                        check_multi(cudaSetDevice(src.device));
                        check_multi(cudaMemcpy(relay,from+k,n,cudaMemcpyDeviceToHost));
                        check_multi(cudaSetDevice(dst.device));
                        check_multi(cudaMemcpy(to+k,relay,n,cudaMemcpyHostToDevice));
                    }
                    stats.relayed_bytes+=bytes;
                }
            }
        }
        parallel([](MultiDevice& d){check_multi(cudaDeviceSynchronize());});
    }

    void masks(MultiDevice& d) {
        const int M=int(owner_host.size()), N=shape.N, T=256;
        check_multi(cudaMemset(d.heavy.get(),0,N));
        check_multi(cudaMemset(d.keep.get(),0,N));
        check_multi(cudaMemset(d.heavy_count.get(),0,4));
        check_multi(cudaMemset(d.keep_count.get(),0,4));
        check_multi(cudaMemset(d.sum.get(),0,8));
        owner_count_sum_kernel<<<(M+T-1)/T,T>>>(d.counts.get(),M,d.sum.get(),d.stopped.get());
        owner_heavy_mask_kernel<<<(M+T-1)/T,T>>>(d.owners.get(),d.counts.get(),M,
            2*shape.D,options.heavy_factor,d.sum.get(),d.heavy.get(),d.heavy_count.get(),d.stopped.get());
        owner_stop_if_empty_kernel<<<1,1>>>(d.heavy_count.get(),d.stopped.get());
        owner_record_valid_kernel<<<(M+T-1)/T,T>>>(d.counts.get(),d.last.get(),M,d.valid.get(),d.stopped.get());
        owner_copy_mask_kernel<<<(N+T-1)/T,T>>>(d.heavy.get(),d.heavy_last.get(),N,d.stopped.get());
        check_multi(cudaMemset(d.positive_sum.get(),0,8));
        check_multi(cudaMemset(d.positive_count.get(),0,4));
        owner_keep_stats_kernel<<<(M+T-1)/T,T>>>(d.owners.get(),d.counts.get(),M,
            d.heavy.get(),d.positive_sum.get(),d.positive_count.get(),d.stopped.get());
        owner_keep_mask_kernel<<<(M+T-1)/T,T>>>(d.owners.get(),d.counts.get(),M,
            2*shape.D,options.temporary_heavy_factor,d.heavy.get(),d.positive_sum.get(),
            d.positive_count.get(),d.keep.get(),d.keep_count.get(),d.stopped.get());
        owner_copy_mask_kernel<<<(N+T-1)/T,T>>>(d.keep.get(),d.keep_last.get(),N,d.stopped.get());
        check_multi(cudaGetLastError()); check_multi(cudaDeviceSynchronize());
    }

    void step(float h,float rho) {
        require_multi(!poisoned,"A failed multi-GPU model must be closed and restored.");
        require_multi(finite_positive(h)&&finite_positive(rho),"Invalid iteration bandwidth.");
        if (stats.stopped_no_heavy) return;
        const int M=int(owner_host.size()), D=shape.D, MM=shape.MM, MY=shape.MY, k=shape.k;
        size_t shared_bytes=sizeof(float)*(size_t(2)*D+size_t(D)*D+size_t(D)*MY);
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_SHARED_SOLVE
        shared_bytes=(shared_bytes+alignof(double)-1)&~(size_t(alignof(double))-1);
        shared_bytes+=sizeof(double)*(size_t(D+1)*(D+1)+size_t(D+1)*MY);
#endif
        const auto reg_time=Clock::now();
        parallel([&](MultiDevice& d) {
            check_multi(cudaMemset(d.counts.get(),0,size_t(M)*4));
            check_multi(cudaMemset(d.exchange.get(),0,size_t(M)*MY*D*4));
            owner_dispatch_maxd(D,[&](auto tier) {
                constexpr int MAXD=decltype(tier)::value;
                owner_copy_state_kernel<MAXD><<<d.count,MAXD>>>(d.owners.get()+d.first,
                    d.count,D,MM,d.tan,d.cen,d.tan_next,d.cen_next);
                for (int local=0;local<d.count;local+=d.tile) {
                    const int B=std::min(d.tile,d.count-local), first=d.first+local;
                    const int* idx=d.cached ? d.index.get()+size_t(local)*k :
                        d.stream->prepare(host_index,local,owner_host.data()+d.first);
#if MANOR_M_REGRESSION_WARP_ENTRIES
                    constexpr int threads=(MAXD==12||MAXD==20)?32:MAXD;
#else
                    constexpr int threads=MAXD;
#endif
                    owner_regression_kernel<MAXD><<<B,threads,shared_bytes>>>(d.x.get(),d.y.get(),
                        idx,nullptr,d.tan,d.cen,d.owners.get(),first,first,B,D,MM,MY,k,false,
                        h,rho,options.lambda,options.jitter,options.regression_mode,nullptr,
                        d.exchange.get(),d.pre.get(),d.weight.get(),d.counts.get(),d.failed.get(),d.stopped.get());
                    check_multi(cudaGetLastError());
                    if (!d.cached) d.stream->consumed();
                }
            });
            check_multi(cudaDeviceSynchronize());
            if (d.stream) d.stream->finish();
        });
        stats.regression_ms+=elapsed_ms(reg_time);
        auto comm=Clock::now();
        gather([](MultiDevice& d){return d.exchange.get();},size_t(MY)*D*4);
        gather([](MultiDevice& d){return d.pre.get();},size_t(D)*4);
        gather([](MultiDevice& d){return d.counts.get();},4);
        gather([](MultiDevice& d){return d.failed.get();},1);
        stats.communication_ms+=elapsed_ms(comm);
        const auto mask_time=Clock::now();
        parallel([&](MultiDevice& d){masks(d);});
        auto& first=*devices.front(); check_multi(cudaSetDevice(first.device));
        check_multi(cudaMemcpy(&stats.stopped_no_heavy,first.stopped.get(),4,cudaMemcpyDeviceToHost));
        check_multi(cudaMemcpy(&stats.valid_iterations,first.valid.get(),4,cudaMemcpyDeviceToHost));
        stats.masks_ms+=elapsed_ms(mask_time);
        if (stats.stopped_no_heavy) return;
        const auto agg_time=Clock::now();
        parallel([&](MultiDevice& d) {
            for (int local=0;local<d.count;local+=d.tile) {
                const int B=std::min(d.tile,d.count-local), first=d.first+local;
                const int* idx=d.cached ? d.index.get()+size_t(local)*k :
                    d.stream->prepare(host_index,local,owner_host.data()+d.first);
                for (int offset=0;offset<B;offset+=d.agg_tile) {
                    const int n=std::min(d.agg_tile,B-offset), q=first+offset;
                    const int* rows=idx+size_t(offset)*k;
                    owner_dispatch_maxd(D,[&](auto tier) {
                        constexpr int MAXD=decltype(tier)::value;
#if MANOR_M_REGRESSION_WARP_ENTRIES
                        constexpr int threads=(MAXD==12||MAXD==20)?32:MAXD;
#else
                        constexpr int threads=MAXD;
#endif
                        owner_neighbor_mask_rebuild_kernel<MAXD><<<n,threads>>>(d.x.get(),rows,nullptr,
                            d.tan,d.owners.get(),q,q,n,D,MM,k,false,h,rho,d.mask.get(),d.stopped.get());
                    });
#if MANOR_M_AGGREGATION_WARP_ENTRIES
                    if (D==10||D==20) {
                        auto launch=[&](auto dim){constexpr int DD=decltype(dim)::value;
                            owner_aggregation_warp_accumulate_kernel<DD,4><<<(n+3)/4,128>>>(
                                d.exchange.get(),d.pre.get(),d.owners.get()+q,d.lookup.get(),d.heavy.get(),
                                d.mask.get(),rows,nullptr,n,D,MY,k,(k+31)/32,false,d.agg_count.get(),
                                d.cacc.get(),d.cov.get(),d.stopped.get());};
                        if (D==10) launch(std::integral_constant<int,10>{});
                        else launch(std::integral_constant<int,20>{});
                    } else
#endif
                    owner_dispatch_maxd(D,[&](auto tier){constexpr int MAXD=decltype(tier)::value;
                        owner_aggregation_accumulate_kernel<MAXD><<<n,MAXD>>>(d.exchange.get(),d.pre.get(),
                            d.owners.get()+q,d.lookup.get(),d.heavy.get(),d.mask.get(),rows,nullptr,n,D,MY,k,
                            (k+31)/32,false,d.agg_count.get(),d.cacc.get(),d.cov.get(),d.stopped.get());});
                    owner_dispatch_maxd(D,[&](auto tier){constexpr int MAXD=decltype(tier)::value;
                        owner_aggregation_finalize_kernel<MAXD><<<n,32>>>(d.x.get(),d.tan,d.owners.get()+q,n,D,MM,
                            d.agg_count.get(),d.cacc.get(),d.cov.get(),d.tan_next,d.cen_next,d.stopped.get());});
                    check_multi(cudaGetLastError());
                }
                if (!d.cached) d.stream->consumed();
            }
            check_multi(cudaDeviceSynchronize()); if (d.stream) d.stream->finish();
            pack_multi_state<<<d.count,128>>>(d.owners.get(),d.first,d.count,D,MM,
                d.tan_next,d.cen_next,d.exchange.get());
            check_multi(cudaGetLastError()); check_multi(cudaDeviceSynchronize());
        });
        stats.aggregation_ms+=elapsed_ms(agg_time);
        comm=Clock::now();
        gather([](MultiDevice& d){return d.exchange.get();},size_t(MM+1)*D*4);
        parallel([&](MultiDevice& d) {
            unpack_multi_state<<<M,128>>>(d.owners.get(),M,D,MM,d.exchange.get(),d.tan_next,d.cen_next);
            check_multi(cudaGetLastError());check_multi(cudaDeviceSynchronize());
            std::swap(d.tan,d.tan_next);std::swap(d.cen,d.cen_next);
        });
        stats.communication_ms+=elapsed_ms(comm);
    }
};

extern "C" MANOR_M_CUDA_API const char* manor_m_multi_error() {return multi_error.c_str();}

extern "C" MANOR_M_CUDA_API int manor_m_multi_minimum_bytes(
    int32_t n,int32_t d,int32_t mm,int32_t my,int32_t owners,int32_t k,uint64_t* required) {
    return multi_call([&] {
        require_multi(required&&n>0&&d>=2&&d<=MAX_D_MANOR&&mm>0&&mm<d&&mm<=MAX_MM_MANOR&&
            my>0&&my<=MAX_MY_MANOR&&owners>0&&owners<=n&&k>0&&k<=n,"Invalid distributed capacity shape.");
        const size_t row=4*(size_t(d)*d+d+1+(size_t(k)+31)/32);
        *required=multi_fixed_bytes(n,d,mm,my,owners)+multi_readout_bytes(n,d,mm,my)
            +std::max(4*row,row+size_t(k)*8);
    });
}

extern "C" MANOR_M_CUDA_API int manor_m_multi_create(
    ManorMCudaContext* const* contexts,int32_t count,const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,const ManorMMultiMemory* memory,ManorMMultiGpu** result) {
    return multi_call([&] {
        require_multi(result&&contexts&&memory&&count>0&&count<=32,"Invalid multi-GPU descriptor.");
        *result=nullptr;
        require_multi(memory->struct_size==sizeof(*memory)&&memory->version==1&&
            memory->pinned_bytes_per_device>=4096&&memory->maximum_tile_owners>0&&
            (memory->flags&~3)==0,"Invalid memory plan.");
        ManorMCudaPhase2Output out{};out.struct_size=sizeof(out);out.abi_version=MANOR_M_CUDA_ABI_VERSION;
        require_multi(input&&options,"Missing input/options.");
        out.tangent_point_major=const_cast<float*>(input->tangent_point_major);
        out.center_point_major=const_cast<float*>(input->center_point_major);
        std::unique_ptr<ManorMMultiGpu> m(new ManorMMultiGpu);
        m->shape=phase2_validate(input,options,nullptr,&out,false);
        require_multi(!options->index_is_shared_slice&&!options->reserved,"Multi-GPU expects ordinary owner rows and updating options.");
        m->options=*options;m->memory=*memory;m->host_index=input->index_zero_based;
        const int N=m->shape.N,D=m->shape.D,MM=m->shape.MM,MY=m->shape.MY,k=m->shape.k;
        std::vector<int> lookup(N,-1);
        for (int i=0;i<N;++i) {
            require_multi(input->owner_mask[i]<=1,"Invalid owner mask.");
            if (input->owner_mask[i]) {lookup[i]=int(m->owner_host.size());m->owner_host.push_back(i);}
        }
        const int M=int(m->owner_host.size());require_multi(M>=count,"Fewer owners than GPUs.");
        for (auto pair:{std::make_pair(input->x_point_major,size_t(N)*D),
                       std::make_pair(input->response_point_major,size_t(N)*MY),
                       std::make_pair(input->tangent_point_major,size_t(N)*MM*D),
                       std::make_pair(input->center_point_major,size_t(N)*D)})
            for (size_t i=0;i<pair.second;++i) require_multi(std::isfinite(pair.first[i]),"Nonfinite training/model input.");
        for (int owner:m->owner_host)
            require_multi(manor_m_index_range_valid(input->index_zero_based+size_t(owner)*k,k,N),"Invalid exact index.");
        m->stats.struct_size=sizeof(m->stats);m->stats.version=1;m->stats.device_count=count;
        m->stats.owner_count=M;m->stats.logical_index_bytes=size_t(N)*k*4;
        for (int i=0;i<count;++i) {
            std::unique_ptr<MultiDevice> p(new MultiDevice);auto& d=*p;
            require_multi(manor_m_cuda_internal_context_device(contexts[i],&d.device)==0,"Invalid CUDA context.");
            for (auto& other:m->devices) require_multi(other->device!=d.device,"Duplicate GPU.");
            check_multi(cudaSetDevice(d.device));
            ManorMResidentCallScope scope(contexts[i]);
            d.first=int(int64_t(M)*i/count);d.count=int(int64_t(M)*(i+1)/count)-d.first;
            size_t free=0,total=0;check_multi(cudaMemGetInfo(&free,&total));
            const size_t limit=std::min(memory->working_bytes?size_t(memory->working_bytes):free,
                free>memory->reserve_bytes?free-size_t(memory->reserve_bytes):size_t(0));
            const size_t fixed=multi_fixed_bytes(N,D,MM,MY,M);
            const size_t evaluation=multi_readout_bytes(N,D,MM,MY);
            require_multi(limit>fixed+evaluation+4096,"Insufficient GPU capacity for distributed model state/evaluation.");
            const size_t scratch_limit=limit-evaluation;
            const size_t row_scratch=4*(size_t(D)*D+D+1+(size_t(k)+31)/32);
            d.agg_tile=int(std::min<size_t>({size_t(d.count),size_t(memory->maximum_tile_owners),
                (scratch_limit-fixed)/std::max<size_t>(1,4*row_scratch)}));
            require_multi(d.agg_tile>0,"No aggregation owner row fits.");
            const size_t available=scratch_limit-fixed-size_t(d.agg_tile)*row_scratch;
            d.cached=!(memory->flags&2)&&size_t(d.count)*k*4<=available;
            if (d.cached) d.tile=int(std::min<size_t>({size_t(d.count),size_t(memory->maximum_tile_owners),
                size_t(memory->pinned_bytes_per_device)/(size_t(k)*4)}));
            else d.tile=int(std::min<size_t>({size_t(d.count),size_t(memory->maximum_tile_owners),
                std::min(available,size_t(memory->pinned_bytes_per_device))/(size_t(k)*8)}));
            require_multi(d.tile>0,"No complete index row fits the tile budget.");
            d.x.allocate_input(size_t(N)*D,"multi X",input->x_point_major);
            d.y.allocate_input(size_t(N)*MY,"multi Y",input->response_point_major);
            check_multi(owner_upload_input(d.x.get(),input->x_point_major,size_t(N)*D*4));
            check_multi(owner_upload_input(d.y.get(),input->response_point_major,size_t(N)*MY*4));
            d.ta.allocate(size_t(N)*MM*D,"multi tangent");d.tb.allocate(size_t(N)*MM*D,"multi next tangent");
            d.ca.allocate(size_t(N)*D,"multi center");d.cb.allocate(size_t(N)*D,"multi next center");
            check_multi(cudaMemcpy(d.ta.get(),input->tangent_point_major,size_t(N)*MM*D*4,cudaMemcpyHostToDevice));
            check_multi(cudaMemcpy(d.ca.get(),input->center_point_major,size_t(N)*D*4,cudaMemcpyHostToDevice));
            check_multi(cudaMemset(d.tb.get(),0,size_t(N)*MM*D*4));check_multi(cudaMemset(d.cb.get(),0,size_t(N)*D*4));
            d.tan=d.ta.get();d.cen=d.ca.get();d.tan_next=d.tb.get();d.cen_next=d.cb.get();
            d.exchange.allocate(size_t(M)*D*std::max(MY,MM+1),"multi exchange");
            d.pre.allocate(size_t(M)*D,"multi regression centers");d.weight.allocate(M,"multi weights");
            d.owners.allocate(M,"multi owners");d.lookup.allocate(N,"multi owner lookup");
            check_multi(cudaMemcpy(d.owners.get(),m->owner_host.data(),size_t(M)*4,cudaMemcpyHostToDevice));
            check_multi(cudaMemcpy(d.lookup.get(),lookup.data(),size_t(N)*4,cudaMemcpyHostToDevice));
            d.counts.allocate(M,"multi counts");d.last.allocate(M,"multi last counts");
            d.failed.allocate(M,"multi solver status");d.heavy.allocate(N,"multi heavy");d.heavy_last.allocate(N,"multi last heavy");
            d.owner_mask.allocate(N,"multi immutable owner mask");
            check_multi(cudaMemcpy(d.owner_mask.get(),input->owner_mask,N,cudaMemcpyHostToDevice));
            d.keep.allocate(N,"multi keep");d.keep_last.allocate(N,"multi last keep");
            check_multi(cudaMemset(d.last.get(),0,size_t(M)*4));check_multi(cudaMemset(d.failed.get(),0,M));
            check_multi(cudaMemset(d.heavy_last.get(),0,N));check_multi(cudaMemset(d.keep_last.get(),0,N));
            d.heavy_count.allocate(1,"multi heavy count");d.keep_count.allocate(1,"multi keep count");
            d.stopped.allocate(1,"multi stopped");d.valid.allocate(1,"multi valid");
            check_multi(cudaMemset(d.stopped.get(),0,4));check_multi(cudaMemset(d.valid.get(),0,4));
            d.sum.allocate(1,"multi count sum");d.positive_sum.allocate(1,"multi positive sum");
            d.positive_count.allocate(1,"multi positive count");
            d.mask.allocate(size_t(d.agg_tile)*((size_t(k)+31)/32),"multi neighbor mask");
            d.agg_count.allocate(d.agg_tile,"multi aggregation count");d.cacc.allocate(size_t(d.agg_tile)*D,"multi aggregation center");
            d.cov.allocate(size_t(d.agg_tile)*D*D,"multi aggregation covariance");
            if (d.cached) {
                d.index.allocate(size_t(d.count)*k,"multi resident owner rows");
                std::vector<int> tile(size_t(d.tile)*k);
                for (int local=0;local<d.count;local+=d.tile) {
                    const int n=std::min(d.tile,d.count-local);
                    for(int j=0;j<n;++j) std::memcpy(tile.data()+size_t(j)*k,
                        input->index_zero_based+size_t(m->owner_host[d.first+local+j])*k,size_t(k)*4);
                    check_multi(cudaMemcpy(d.index.get()+size_t(local)*k,tile.data(),size_t(n)*k*4,cudaMemcpyHostToDevice));
                    d.index_upload+=size_t(n)*k*4;
                }
            } else {
                d.layout.reset(new OwnerIndexTiles(d.count,k,d.tile,false,1));
                d.stream.reset(new OwnerIndexStream(*d.layout));d.stream->allocate();
                d.pinned=d.layout->device_bytes;m->stats.pinned_bytes+=d.pinned;
            }
            const size_t used_index=d.cached?size_t(d.count)*k*4:d.layout->device_bytes;
            d.gram_budget=available>used_index?available-used_index:0;
#if MANOR_M_MULTI_PERSISTENT_READOUT
            d.readout.allocate(N,D,MM,MY);
#if MANOR_M_MULTI_CACHED_GRAM
            const size_t gram_count=size_t(N)*D*(D+1)/2,gram_bytes=gram_count*4;
            size_t free_now=0,total_now=0;check_multi(cudaMemGetInfo(&free_now,&total_now));
            if(gram_bytes<=d.gram_budget&&free_now>=gram_bytes+memory->reserve_bytes+64*1024*1024) {
                d.readout_gram.allocate(gram_count,"multi tangent Gram cache");d.gram_bytes=gram_bytes;
            }
#endif
#endif
            m->devices.push_back(std::move(p));
        }
        check_multi(cudaHostAlloc(&m->relay,ManorMMultiGpu::relay_capacity,cudaHostAllocDefault));
        m->stats.pinned_bytes+=ManorMMultiGpu::relay_capacity;
        m->peer.assign(count,std::vector<bool>(count,false));
        for (int i=0;i<count;++i) for(int j=0;j<count;++j) if(i!=j&&!(memory->flags&1)) {
            int can=0;check_multi(cudaDeviceCanAccessPeer(&can,m->devices[i]->device,m->devices[j]->device));
            if (can) {check_multi(cudaSetDevice(m->devices[i]->device));
                const auto e=cudaDeviceEnablePeerAccess(m->devices[j]->device,0);
                if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled) check_multi(e);
                if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                m->peer[i][j]=true;}
        }
        *result=m.release();
    });
}

extern "C" MANOR_M_CUDA_API int manor_m_multi_stats(ManorMMultiGpu* m,ManorMMultiStats* stats) {
    return multi_call([&]{require_multi(m&&stats,"Missing model/stats.");
        m->stats.uploaded_index_bytes=0;
        for(auto& d:m->devices) m->stats.uploaded_index_bytes+=d->index_upload+(d->stream?d->stream->host_to_device_bytes:0);
        *stats=m->stats;});
}
extern "C" MANOR_M_CUDA_API int manor_m_multi_readout_cache(ManorMMultiGpu* m,int32_t slot,
    uint64_t* bytes,int32_t* iteration) {
    return multi_call([&]{require_multi(m&&slot>=0&&size_t(slot)<m->devices.size()&&bytes&&iteration,
        "Invalid readout-cache descriptor.");auto& d=*m->devices[slot];
        *bytes=d.gram_bytes;*iteration=d.gram_iteration;});
}
extern "C" MANOR_M_CUDA_API int manor_m_multi_step(ManorMMultiGpu* m,float h,float rho,ManorMMultiStats* stats) {
    const int status=multi_call([&]{require_multi(m,"Missing model.");m->step(h,rho);});
    if(status) {if(m)m->poisoned=true;return status;}
    return manor_m_multi_stats(m,stats);
}
extern "C" MANOR_M_CUDA_API int manor_m_multi_snapshot(ManorMMultiGpu* m,float* tan,float* cen,
    int32_t* counts,uint8_t* heavy,uint8_t* keep,uint8_t* failed) {
    return multi_call([&]{require_multi(m&&tan&&cen,"Missing snapshot output.");auto& d=*m->devices.front();
        check_multi(cudaSetDevice(d.device));const int N=m->shape.N,D=m->shape.D,MM=m->shape.MM,M=int(m->owner_host.size());
        check_multi(cudaMemcpy(tan,d.tan,size_t(N)*D*MM*4,cudaMemcpyDeviceToHost));
        check_multi(cudaMemcpy(cen,d.cen,size_t(N)*D*4,cudaMemcpyDeviceToHost));
        std::vector<int32_t> num(counts?M:0);std::vector<uint8_t> fail(failed?M:0);
        if(counts){check_multi(cudaMemcpy(num.data(),d.last.get(),size_t(M)*4,cudaMemcpyDeviceToHost));std::fill(counts,counts+N,0);}
        if(failed){check_multi(cudaMemcpy(fail.data(),d.failed.get(),M,cudaMemcpyDeviceToHost));std::fill(failed,failed+N,0);}
        if(heavy)check_multi(cudaMemcpy(heavy,d.heavy_last.get(),N,cudaMemcpyDeviceToHost));
        if(keep)check_multi(cudaMemcpy(keep,d.keep_last.get(),N,cudaMemcpyDeviceToHost));
        for(int q=0;q<M;++q){if(counts)counts[m->owner_host[q]]=num[q];if(failed)failed[m->owner_host[q]]=fail[q];}
        int q=0;
        for(int i=0;i<N;++i) {
            if(q<M&&m->owner_host[q]==i) {++q;continue;}
            std::fill(tan+size_t(i)*D*MM,tan+size_t(i+1)*D*MM,0.0f);
            std::fill(cen+size_t(i)*D,cen+size_t(i+1)*D,0.0f);
        }
    });
}
extern "C" MANOR_M_CUDA_API int manor_m_multi_destroy(ManorMMultiGpu* m) {
    return multi_call([&]{delete m;});
}

// Read an already synchronized model directly. The prediction kernels and
// fixed-order reducer are unchanged; K0 retains the original host division,
// while K>0 uses the original device history-normalization kernel.
extern "C" MANOR_M_CUDA_API int manor_m_multi_evaluate(ManorMMultiGpu* m,int32_t slot,
    const ManorMCudaPhase2Evaluation* e,float h,float rho,int32_t initial,
    float* mse,float* coordinate,float* tangent,float* contributions) {
    return multi_call([&] {
        require_multi(m&&!m->poisoned&&e&&slot>=0&&size_t(slot)<m->devices.size(),"Invalid model/readout device.");
        require_multi(e->struct_size==sizeof(*e)&&e->abi_version==MANOR_M_CUDA_ABI_VERSION,"Readout ABI mismatch.");
        require_multi(finite_positive(h)&&finite_positive(rho)&&(initial==0||initial==1),"Invalid readout schedule.");
        require_multi(e->test_x_point_major&&e->test_point_count>0&&e->test_point_count<=65536,"Invalid readout queries.");
        const bool response=e->test_y_point_major!=nullptr, geometry=e->test_tangent_point_major!=nullptr;
        require_multi(response!=geometry&&!e->test_center_point_major,"Choose one response/tangent metric view.");
        require_multi(std::isfinite(e->sum_sparse_threshold)&&e->sum_sparse_threshold>=0&&
            e->count_sparse_threshold>=0&&finite_positive(e->fallback_exponent_scale)&&
            finite_positive(e->epsilon_weight),"Invalid sparse readout options.");
        const int N=m->shape.N,D=m->shape.D,MM=m->shape.MM,MY=m->shape.MY,Q=int(e->test_point_count);
        // The original entry resolves zero to 2*D before launching either
        // ordinary or sparse readout. Preserve that default in this entry too.
        const int sparse_count=e->count_sparse_threshold>0?e->count_sparse_threshold:std::max(1,2*D);
        const int nh=e->h_scale_count,nr=e->rho_scale_count;
        require_multi(nh>0&&nr>0&&int64_t(nh)*nr<=9&&e->h_scales&&e->rho_scales,"Invalid fixed readout grid.");
        const int S=nh*nr,channels=S*(MY+4);
        require_multi(contributions||((!response||(mse&&coordinate))&&(!geometry||tangent)),"Missing metric output.");
        for(int j=0;j<nh;++j)require_multi(finite_positive(e->h_scales[j]),"Invalid h scale.");
        for(int j=0;j<nr;++j)require_multi(finite_positive(e->rho_scales[j]),"Invalid rho scale.");
        for(int64_t j=0;j<int64_t(Q)*D;++j)require_multi(std::isfinite(e->test_x_point_major[j]),"Nonfinite query X.");
        const float* truth=response?e->test_y_point_major:e->test_tangent_point_major;
        for(int64_t j=0;j<int64_t(Q)*(response?MY:MM*D);++j)require_multi(std::isfinite(truth[j]),"Nonfinite query truth.");
        auto& d=*m->devices[slot];check_multi(cudaSetDevice(d.device));
#if MANOR_M_MULTI_PERSISTENT_READOUT
        auto &qx=d.readout.qx,&qy=d.readout.qy,&qt=d.readout.qt,&hs=d.readout.hs,&rs=d.readout.rs,
            &xnorm=d.readout.xnorm,&tdot=d.readout.tdot,&points=d.readout.points,&sums=d.readout.sums,
            &coords=d.readout.coords,&minq=d.readout.minq;
        auto &pairs=d.readout.pairs,&pair_count=d.readout.pair_count,&zero=d.readout.zero,&one=d.readout.one;
#else
        OwnerCudaBuffer<float> qx,qy,qt,hs,rs,xnorm,tdot,points,sums,coords,minq;
        OwnerCudaBuffer<int> pairs,pair_count,zero,one;
        qx.allocate(size_t(Q)*D,"multi query X");check_multi(cudaMemcpy(qx.get(),e->test_x_point_major,size_t(Q)*D*4,cudaMemcpyHostToDevice));
        if(response){qy.allocate(size_t(Q)*MY,"multi query Y");check_multi(cudaMemcpy(qy.get(),e->test_y_point_major,size_t(Q)*MY*4,cudaMemcpyHostToDevice));}
        if(geometry){qt.allocate(size_t(Q)*MM*D,"multi query tangent");check_multi(cudaMemcpy(qt.get(),e->test_tangent_point_major,size_t(Q)*MM*D*4,cudaMemcpyHostToDevice));}
        hs.allocate(nh,"multi readout h");rs.allocate(nr,"multi readout rho");
        check_multi(cudaMemcpy(hs.get(),e->h_scales,size_t(nh)*4,cudaMemcpyHostToDevice));
        check_multi(cudaMemcpy(rs.get(),e->rho_scales,size_t(nr)*4,cudaMemcpyHostToDevice));
        xnorm.allocate(N,"multi train norms");tdot.allocate(size_t(N)*MM,"multi tangent dot X");
        points.allocate(size_t(Q)*channels,"multi query contributions");sums.allocate(size_t(S)*4,"multi metric sums");
        coords.allocate(size_t(S)*MY,"multi coordinate sums");
        pairs.allocate(std::max(size_t(Q)*S*2,size_t(Q)*(S+2)+1),"multi sparse pairs");
        minq.allocate(size_t(Q)*S*2,"multi sparse minq");pair_count.allocate(1,"multi sparse count");
        zero.allocate(1,"multi readout active");one.allocate(1,"multi readout valid");
        check_multi(cudaMemset(zero.get(),0,4));const int valid=1;check_multi(cudaMemcpy(one.get(),&valid,4,cudaMemcpyHostToDevice));
#endif
#if MANOR_M_MULTI_PERSISTENT_READOUT
        check_multi(cudaMemcpy(qx.get(),e->test_x_point_major,size_t(Q)*D*4,cudaMemcpyHostToDevice));
        if(response)check_multi(cudaMemcpy(qy.get(),e->test_y_point_major,size_t(Q)*MY*4,cudaMemcpyHostToDevice));
        if(geometry)check_multi(cudaMemcpy(qt.get(),e->test_tangent_point_major,size_t(Q)*MM*D*4,cudaMemcpyHostToDevice));
        check_multi(cudaMemcpy(hs.get(),e->h_scales,size_t(nh)*4,cudaMemcpyHostToDevice));
        check_multi(cudaMemcpy(rs.get(),e->rho_scales,size_t(nr)*4,cudaMemcpyHostToDevice));
#endif
        check_multi(cudaMemset(pair_count.get(),0,4));check_multi(cudaMemset(points.get(),0,size_t(Q)*channels*4));
        check_multi(cudaMemset(sums.get(),0,size_t(S)*16));check_multi(cudaMemset(coords.get(),0,size_t(S)*MY*4));
        if(d.readout.stats_iteration!=m->stats.valid_iterations) {
            owner_fused_eval_train_stats_guarded_kernel<<<(N+255)/256,256>>>(d.x.get(),d.tan,D,MM,N,xnorm.get(),tdot.get(),zero.get());
#if MANOR_M_MULTI_PERSISTENT_READOUT
            d.readout.stats_iteration=m->stats.valid_iterations;
#endif
        }
        cudaDeviceProp prop;check_multi(cudaGetDeviceProperties(&prop,d.device));int jt=0;size_t shared=0;
        const size_t tiers[]={std::min(size_t(prop.sharedMemPerBlockOptin),size_t(prop.sharedMemPerMultiprocessor)/3),
            std::min(size_t(prop.sharedMemPerBlockOptin),size_t(prop.sharedMemPerMultiprocessor)/2),size_t(prop.sharedMemPerBlockOptin)};
        for(size_t budget:tiers){for(int n:{128,96,64,48,32,16}){
            const size_t bytes=owner_tiled_shared_bytes(S,D,MM,MY,n,8,response,geometry,false);
            if(bytes<=budget){jt=n;shared=bytes;break;}}
            if(jt)break;}
        require_multi(jt>0,"Readout does not fit shared memory.");
        check_multi(cudaFuncSetAttribute(owner_fused_eval_tiled_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
        const float* prepared_gram=nullptr;
#if MANOR_M_MULTI_CACHED_GRAM
        if(geometry) {
            const size_t count=size_t(N)*D*(D+1)/2, bytes=count*sizeof(float);
            size_t free=0,total=0;check_multi(cudaMemGetInfo(&free,&total));
            if(d.gram_bytes
#if !MANOR_M_MULTI_PERSISTENT_READOUT
                    ||
                    (bytes<=d.gram_budget&&free>=bytes+m->memory.reserve_bytes+64*1024*1024)) {
#else
                    ) {
#endif
                if(!d.gram_bytes){d.readout_gram.allocate(count,"multi tangent Gram cache");d.gram_bytes=bytes;}
                if(d.gram_iteration!=m->stats.valid_iterations) {
                    multi_prepare_tangent_gram<<<(N+7)/8,256>>>(d.tan,N,D,MM,d.readout_gram.get());
                    check_multi(cudaGetLastError());d.gram_iteration=m->stats.valid_iterations;
                }
                prepared_gram=d.readout_gram.get();
                check_multi(cudaFuncSetAttribute(multi_cached_gram_eval_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared)));
            }
        }
#endif
        auto launch=[&](int blocks,const int* ib,const int* offsets,const int* slots,const float* distances,
                        int* count,int* append,float* append_min){
#if MANOR_M_MULTI_CACHED_GRAM
            if(prepared_gram) {
                multi_cached_gram_eval_kernel<<<blocks,256,shared>>>(qx.get(),qy.get(),qt.get(),nullptr,
                    d.x.get(),d.y.get(),d.cen,d.tan,xnorm.get(),tdot.get(),prepared_gram,d.owner_mask.get(),
                    D,MM,MY,N,Q,h,rho,hs.get(),rs.get(),nh,S,jt,e->sum_sparse_threshold,
                    sparse_count,e->fallback_exponent_scale,e->epsilon_weight,
                    e->center_h_scale,e->center_rho_scale,response,geometry,false,false,sums.get(),coords.get(),
                    ib,offsets,slots,distances,count,append,append_min,zero.get(),points.get());
                check_multi(cudaGetLastError());return;
            }
#endif
            owner_fused_eval_tiled_kernel<<<blocks,256,shared>>>(qx.get(),qy.get(),qt.get(),nullptr,
                d.x.get(),d.y.get(),d.cen,d.tan,xnorm.get(),tdot.get(),d.owner_mask.get(),
                D,MM,MY,N,Q,h,rho,hs.get(),rs.get(),nh,S,jt,e->sum_sparse_threshold,
                sparse_count,e->fallback_exponent_scale,e->epsilon_weight,
                e->center_h_scale,e->center_rho_scale,response,geometry,false,false,sums.get(),coords.get(),
                ib,offsets,slots,distances,count,append,append_min,zero.get(),points.get());
            check_multi(cudaGetLastError());};
        launch(Q,nullptr,nullptr,nullptr,nullptr,pair_count.get(),pairs.get(),minq.get());
        int sparse=0;check_multi(cudaMemcpy(&sparse,pair_count.get(),4,cudaMemcpyDeviceToHost));
        require_multi(sparse>=0&&size_t(sparse)<=size_t(Q)*S,"Sparse pair count out of range.");
        if(sparse){
            std::vector<int> raw(size_t(sparse)*2),order(sparse),groups,offsets,flat(sparse);
            std::vector<float> raw_min(size_t(sparse)*2),flat_min(size_t(sparse)*2);
            check_multi(cudaMemcpy(raw.data(),pairs.get(),raw.size()*4,cudaMemcpyDeviceToHost));
            check_multi(cudaMemcpy(raw_min.data(),minq.get(),raw_min.size()*4,cudaMemcpyDeviceToHost));
            std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return raw[2*size_t(a)]<raw[2*size_t(b)];});
            for(int p=0;p<sparse;++p){int i=order[p],query=raw[2*size_t(i)];
                if(groups.empty()||groups.back()!=query){groups.push_back(query);offsets.push_back(p);}
                flat[p]=raw[2*size_t(i)+1];flat_min[2*size_t(p)]=raw_min[2*size_t(i)];flat_min[2*size_t(p)+1]=raw_min[2*size_t(i)+1];}
            offsets.push_back(sparse);std::vector<int> packed;
            packed.insert(packed.end(),groups.begin(),groups.end());packed.insert(packed.end(),offsets.begin(),offsets.end());packed.insert(packed.end(),flat.begin(),flat.end());
            check_multi(cudaMemcpy(pairs.get(),packed.data(),packed.size()*4,cudaMemcpyHostToDevice));
            check_multi(cudaMemcpy(minq.get(),flat_min.data(),flat_min.size()*4,cudaMemcpyHostToDevice));
            launch(int(groups.size()),pairs.get(),pairs.get()+groups.size(),pairs.get()+groups.size()*2+1,
                minq.get(),nullptr,nullptr,nullptr);
        }
        if(contributions) {
            check_multi(cudaMemcpy(contributions,points.get(),size_t(Q)*channels*4,cudaMemcpyDeviceToHost));
        } else multi_finish_readout(points.get(),Q,S,MY,zero.get(),one.get(),sums.get(),coords.get(),initial,
            response?mse:nullptr,response?coordinate:nullptr,geometry?tangent:nullptr);
    });
}

extern "C" MANOR_M_CUDA_API int manor_m_multi_reduce_queries(ManorMMultiGpu* m,int32_t slot,
    const float* contributions,int32_t Q,int32_t S,int32_t initial,
    float* mse,float* coordinate,float* tangent) {
    return multi_call([&] {
        require_multi(m&&!m->poisoned&&slot>=0&&size_t(slot)<m->devices.size()&&contributions,
            "Invalid query-reduction model/device.");
        require_multi(Q>0&&Q<=65536&&S>0&&S<=9&&(initial==0||initial==1)&&
            ((mse&&coordinate&&!tangent)||(!mse&&!coordinate&&tangent)),"Invalid query reduction shape/output.");
        check_multi(cudaSetDevice(m->devices[slot]->device));const int MY=m->shape.MY;
#if MANOR_M_MULTI_PERSISTENT_READOUT
        auto& buffer=m->devices[slot]->readout;
        auto &points=buffer.points,&sums=buffer.sums,&coords=buffer.coords;
        auto &zero=buffer.zero,&one=buffer.one;
#else
        OwnerCudaBuffer<float> points,sums,coords;OwnerCudaBuffer<int> zero,one;
        points.allocate(size_t(Q)*S*(MY+4),"multi ordered query gather");
        sums.allocate(size_t(S)*4,"multi gathered metric sums");coords.allocate(size_t(S)*MY,"multi gathered coordinate sums");
        zero.allocate(1,"multi gathered active");one.allocate(1,"multi gathered valid");
        check_multi(cudaMemset(zero.get(),0,4));const int valid=1;check_multi(cudaMemcpy(one.get(),&valid,4,cudaMemcpyHostToDevice));
#endif
        check_multi(cudaMemcpy(points.get(),contributions,size_t(Q)*S*(MY+4)*4,cudaMemcpyHostToDevice));
        multi_finish_readout(points.get(),Q,S,MY,zero.get(),one.get(),sums.get(),coords.get(),initial,mse,coordinate,tangent);
    });
}
