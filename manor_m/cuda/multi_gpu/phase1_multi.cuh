// Included by the separately generated Phase-1 translation unit after its
// unchanged request type. Original single-GPU files are never rewritten.
struct MultiP1Span { void* pointer; size_t row_bytes; };
struct MultiP1Exchange {
    Phase1CoreRequest request;
    int count, selected=0, finished=0;
    std::vector<int> devices;
    std::vector<std::vector<MultiP1Span>> spans;
    std::vector<std::vector<bool>> peer;
    std::mutex mutex;
    std::condition_variable condition;
    bool exchanged=false, cancelled=false;
    std::string error;
    void* relay=nullptr;
    static constexpr size_t relay_bytes=16*1024*1024;
    explicit MultiP1Exchange(int n):count(n),devices(n),spans(n),peer(n,std::vector<bool>(n,false)) {
        STABLE_CUDA_CHECK(cudaHostAlloc(&relay,relay_bytes,cudaHostAllocDefault));
    }
    ~MultiP1Exchange(){if(relay)cudaFreeHost(relay);}
    void abort(const std::string& why) {
        std::lock_guard<std::mutex> lock(mutex);
        if(!cancelled){cancelled=true;error=why;}
        condition.notify_all();
    }
    void check() {if(cancelled)throw std::runtime_error(error);}
    void copy() {
        const int N=request.N;
        for(size_t f=0;f<spans.front().size();++f){
            const size_t row=spans[0][f].row_bytes;
            for(int s=0;s<count;++s){
                const size_t first=size_t(N)*s/count,last=size_t(N)*(s+1)/count;
                const size_t bytes=(last-first)*row,offset=first*row;
                for(int t=0;t<count;++t)if(s!=t){
                    char* from=static_cast<char*>(spans[s][f].pointer)+offset;
                    char* to=static_cast<char*>(spans[t][f].pointer)+offset;
                    if(peer[t][s]){
                        STABLE_CUDA_CHECK(cudaSetDevice(devices[t]));
                        STABLE_CUDA_CHECK(cudaMemcpyPeerAsync(to,devices[t],from,devices[s],bytes,nullptr));
                    }else for(size_t k=0;k<bytes;k+=relay_bytes){
                        const size_t n=std::min(relay_bytes,bytes-k);
                        STABLE_CUDA_CHECK(cudaSetDevice(devices[s]));
                        STABLE_CUDA_CHECK(cudaMemcpy(relay,from+k,n,cudaMemcpyDeviceToHost));
                        STABLE_CUDA_CHECK(cudaSetDevice(devices[t]));
                        STABLE_CUDA_CHECK(cudaMemcpy(to+k,relay,n,cudaMemcpyHostToDevice));
                    }
                }
            }
        }
        for(int d:devices){STABLE_CUDA_CHECK(cudaSetDevice(d));STABLE_CUDA_CHECK(cudaDeviceSynchronize());}
    }
    void selection(int rank,std::vector<MultiP1Span> fields){
        STABLE_CUDA_CHECK(cudaDeviceSynchronize());
        std::unique_lock<std::mutex> lock(mutex);check();
        spans[rank]=std::move(fields);
        if(++selected==count){
            lock.unlock();
            try{copy();}catch(const std::exception& e){abort(e.what());throw;}
            lock.lock();exchanged=true;condition.notify_all();
        }else condition.wait(lock,[&]{return exchanged||cancelled;});
        check();lock.unlock();STABLE_CUDA_CHECK(cudaSetDevice(devices[rank]));
    }
    void finish(){
        std::unique_lock<std::mutex> lock(mutex);check();
        if(++finished==count)condition.notify_all();
        else condition.wait(lock,[&]{return finished==count||cancelled;});
        check();
    }
};
struct MultiP1Thread {MultiP1Exchange* exchange;int rank,first,last;};
static thread_local MultiP1Thread* multi_p1_thread=nullptr;
static int multi_p1_first(){return multi_p1_thread?multi_p1_thread->first:0;}
static int multi_p1_last(int n){return multi_p1_thread?multi_p1_thread->last:n;}
