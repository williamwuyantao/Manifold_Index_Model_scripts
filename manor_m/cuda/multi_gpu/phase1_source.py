"""Build a parallel adapter around the unchanged Phase-1 kernel source."""
from pathlib import Path
import argparse


def generate(source, output, hooks):
    text=source.read_text()
    def replace(before,after,count=1):
        nonlocal text
        if text.count(before)!=count:
            raise ValueError('Phase1 adapter source marker changed: '+before[:80])
        text=text.replace(before,after)
    replace('#include <type_traits>', '#include <type_traits>\n#include <thread>\n#include <mutex>\n#include <condition_variable>')
    replace('#include "../../matlab/cuda_mex/manor_m_phase1_mex.cu"',
            '#include "'+str(source.parent.parent.parent/'matlab/cuda_mex/manor_m_phase1_mex.cu')+'"')
    replace('static void phase1_require(', '#include "'+str(hooks)+'"\n\nstatic void phase1_require(')
    replace('bool generated_index = false)\n{', 'bool generated_index = false)\n{\n    if(multi_p1_thread) return multi_p1_thread->exchange->request;')
    replace('    if (!destination || count == 0) return;\n    STABLE_CUDA_CHECK(cudaMemcpy(',
        '    if (!destination || count == 0) return;\n'
        '    if(multi_p1_thread){\n'
        '        const size_t width=count/multi_p1_thread->exchange->request.N;\n'
        '        const size_t offset=size_t(multi_p1_thread->first)*width;\n'
        '        destination+=offset;source+=offset;\n'
        '        count=size_t(multi_p1_thread->last-multi_p1_thread->first)*width;\n'
        '    }\n    STABLE_CUDA_CHECK(cudaMemcpy(')
    replace('for (int q0 = 0; q0 < request.anchors; q0 += BQ) {\n                const int B = std::min(BQ, request.anchors - q0);',
        'for (int q0 = multi_p1_first(); q0 < multi_p1_last(request.anchors); q0 += BQ) {\n'
        '                const int B = std::min(BQ, multi_p1_last(request.anchors) - q0);',2)
    names=[('d_tangent','request.D*request.MM*4'),('d_center','request.D*4'),
        ('d_radius','4'),('d_axial','4'),('d_num_neigh','4')]
    names += [('d_selected_'+n, '1' if n=='covered' else '4') for n in
              ('scale','index','cut','covered','geom','stable_h','neighbor','split','h_a','h_b')]
    names += [('d_fallback'+name[1:],width) for name,width in list(names)]
    names += [('d_neighborhood_radius_limit','4')]
    spans=','.join('{'+name+',size_t('+width+')}' for name,width in names)
    replace('        const int blocks = (request.N + threads - 1) / threads;\n        profile.begin("owner_rule");',
        '        if(multi_p1_thread)multi_p1_thread->exchange->selection(multi_p1_thread->rank,{'+spans+'});\n'
        '        const int blocks = (request.N + threads - 1) / threads;\n        profile.begin("owner_rule");')
    # Optional debug host outputs are also written by disjoint query ranges.
    replace('for (int anchor = 0; anchor < request.N; ++anchor) {\n            const int scale = selected_scale_host[anchor];',
        'for (int anchor = multi_p1_first(); anchor < multi_p1_last(request.N); ++anchor) {\n            const int scale = selected_scale_host[anchor];')
    replace('        std::array<unsigned long long, 3> owner_stats_host{};',
        '        if(multi_p1_thread)multi_p1_thread->exchange->finish();\n        std::array<unsigned long long, 3> owner_stats_host{};')
    text += r'''
namespace {thread_local std::string multi_p1_error;}
extern "C" MANOR_M_CUDA_API const char* manor_m_multi_phase1_error(){return multi_p1_error.c_str();}
extern "C" MANOR_M_CUDA_API int manor_m_multi_phase1(
    ManorMCudaContext* const* contexts,int count,const float* x,int64_t n,int d,
    const int32_t* index,int k,const int32_t* cuts,int cut_count,
    const ManorMCudaPhase1Options* options,ManorMCudaPhase1Output* output,
    const ManorMCudaP1CapacityOptions* capacity,ManorMCudaPhase1Stats* stats,
    ManorMCudaP1CapacityPlan* plans) {
    try{
        if(!contexts||count<1||count>32||n<count||!capacity||!stats||!plans)
            throw std::invalid_argument("Invalid parallel Phase1 descriptor.");
        MultiP1Exchange exchange(count);
        exchange.request=phase1_validate(x,n,d,2,index,n,k,&k,1,cuts,cut_count,options,output);
        for(int i=0;i<count;++i){
            if(capacity[i].mode!=3)throw std::invalid_argument("Parallel Phase1 requires whole-row streamed/local-mask mode.");
            if(manor_m_cuda_internal_context_device(contexts[i],&exchange.devices[i]))
                throw std::invalid_argument("Invalid parallel Phase1 context.");
            for(int j=0;j<i;++j)if(exchange.devices[i]==exchange.devices[j])
                throw std::invalid_argument("Duplicate parallel Phase1 device.");
        }
        for(int t=0;t<count;++t)for(int s=0;s<count;++s)if(s!=t){
            int can=0;STABLE_CUDA_CHECK(cudaDeviceCanAccessPeer(&can,exchange.devices[t],exchange.devices[s]));
            if(can){STABLE_CUDA_CHECK(cudaSetDevice(exchange.devices[t]));
                const auto e=cudaDeviceEnablePeerAccess(exchange.devices[s],0);
                if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)STABLE_CUDA_CHECK(e);
                if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
                exchange.peer[t][s]=true;}
        }
        std::vector<std::thread> threads;
        for(int rank=0;rank<count;++rank)threads.emplace_back([&,rank]{
            MultiP1Thread scope{&exchange,rank,int(n*rank/count),int(n*(rank+1)/count)};
            multi_p1_thread=&scope;
            try{
                const int status=manor_m_cuda_phase1_hybrid_bounded(contexts[rank],x,n,d,2,index,n,k,
                    &k,1,cuts,cut_count,options,output,&stats[rank],&capacity[rank],&plans[rank]);
                if(status)exchange.abort(manor_m_cuda_last_error());
            }catch(const std::exception& e){exchange.abort(e.what());}
            multi_p1_thread=nullptr;
        });
        for(auto& thread:threads)thread.join();
        exchange.check();multi_p1_error.clear();return 0;
    }catch(const std::exception& e){multi_p1_error=e.what();return 1;}
}
'''
    output.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('hooks',type=Path)
    a=p.parse_args();generate(a.source.resolve(),a.output.resolve(),a.hooks.resolve())
