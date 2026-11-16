#pragma once

// Private, bounded diagnostic only. Included after OwnerCudaBuffer is defined.
// It observes the existing evaluator; it never replaces metric atomic adds.
#include <cstdio>
#include <cstdlib>

__device__ double* owner_eval_audit_predictions = nullptr;
__device__ double* owner_eval_audit_weighted_predictions = nullptr;
__device__ double* owner_eval_audit_weights = nullptr;
__device__ double* owner_eval_audit_squared_errors = nullptr;
__device__ float* owner_eval_audit_errors = nullptr;
__device__ float* owner_eval_audit_coordinate_errors = nullptr;
__device__ unsigned int* owner_eval_audit_visits = nullptr;
__device__ int owner_eval_audit_queries = 0;

__device__ __forceinline__ void owner_eval_audit_prediction(
    int slot, int query, int response, int responses,
    double weighted_prediction, double prediction, double squared_error)
{
    if (owner_eval_audit_errors == nullptr) return;
    const size_t index = ((size_t)slot * owner_eval_audit_queries + query)
        * responses + response;
    owner_eval_audit_weighted_predictions[index] = weighted_prediction;
    owner_eval_audit_predictions[index] = prediction;
    owner_eval_audit_coordinate_errors[index] = (float)squared_error;
}

__device__ __forceinline__ void owner_eval_audit_total(
    int slot, int query, double weight, double squared_error, float error)
{
    if (owner_eval_audit_errors == nullptr) return;
    const size_t index = (size_t)slot * owner_eval_audit_queries + query;
    owner_eval_audit_weights[index] = weight;
    owner_eval_audit_squared_errors[index] = squared_error;
    owner_eval_audit_errors[index] = error;
    atomicAdd(owner_eval_audit_visits + index, 1u);
}

class OwnerEvalContributionAudit {
    OwnerCudaBuffer<double> predictions_, weighted_predictions_, weights_, squared_errors_;
    OwnerCudaBuffer<float> errors_, coordinate_errors_;
    OwnerCudaBuffer<unsigned int> visits_;
    std::string prefix_;
    size_t points_ = 0, slots_ = 0;
    int queries_ = 0, responses_ = 0;
    bool installed_ = false;

    void write_file(const char* suffix, const void* values, size_t bytes) {
        const std::string path = prefix_ + suffix;
        FILE* stream = std::fopen(path.c_str(), "wbx");
        if (!stream)
            throw OwnerMexError("manor_m_phase2_mex:evalAudit", "Cannot create new audit evidence: " + path);
        const size_t written = std::fwrite(values, 1, bytes, stream);
        const int closed = std::fclose(stream);
        if (written != bytes || closed != 0)
            throw OwnerMexError("manor_m_phase2_mex:evalAudit", "Incomplete audit evidence: " + path);
    }
    template<class T> void download(const char* suffix, const T* device, size_t count) {
        std::vector<T> values(count);
        owner_cuda_check(cudaMemcpy(values.data(), device, count * sizeof(T),
            cudaMemcpyDeviceToHost), "download evaluation audit");
        write_file(suffix, values.data(), count * sizeof(T));
    }
public:
    OwnerEvalContributionAudit(int iteration, int queries, size_t slots,
                              int responses, bool response_eval) {
        const char* prefix = std::getenv("MFM_EVAL_CONTRIBUTION_PREFIX");
        if (iteration != 4 || !prefix || !*prefix) return;
        if (!response_eval || queries < 1 || queries > 10000 || slots < 1 ||
            slots > 81 || responses < 1 || responses > MAX_MY_MANOR || prefix[0] != '/')
            throw OwnerMexError("manor_m_phase2_mex:evalAudit", "Expected bounded K4 response audit and absolute new output prefix.");
        prefix_ = prefix; queries_ = queries; slots_ = slots; responses_ = responses;
        points_ = slots * (size_t)queries;
        const size_t values = points_ * responses;
        double* predictions = predictions_.allocate(values, "allocate audit predictions");
        double* weighted = weighted_predictions_.allocate(values, "allocate audit numerators");
        double* weights = weights_.allocate(points_, "allocate audit denominators");
        double* squared = squared_errors_.allocate(points_, "allocate audit FP64 errors");
        float* errors = errors_.allocate(points_, "allocate audit FP32 errors");
        float* coordinates = coordinate_errors_.allocate(values, "allocate audit coordinate errors");
        unsigned int* visits = visits_.allocate(points_, "allocate audit visits");
        owner_cuda_check(cudaMemset(predictions, 0xff, values * sizeof(double)), "initialize audit predictions");
        owner_cuda_check(cudaMemset(weighted, 0xff, values * sizeof(double)), "initialize audit numerators");
        owner_cuda_check(cudaMemset(weights, 0xff, points_ * sizeof(double)), "initialize audit denominators");
        owner_cuda_check(cudaMemset(squared, 0xff, points_ * sizeof(double)), "initialize audit FP64 errors");
        owner_cuda_check(cudaMemset(errors, 0xff, points_ * sizeof(float)), "initialize audit FP32 errors");
        owner_cuda_check(cudaMemset(coordinates, 0xff, values * sizeof(float)), "initialize audit coordinate errors");
        owner_cuda_check(cudaMemset(visits, 0, points_ * sizeof(unsigned int)), "initialize audit visits");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_queries, &queries, sizeof(queries)), "install audit query count");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_predictions, &predictions, sizeof(predictions)), "install audit predictions");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_weighted_predictions, &weighted, sizeof(weighted)), "install audit numerators");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_weights, &weights, sizeof(weights)), "install audit denominators");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_squared_errors, &squared, sizeof(squared)), "install audit FP64 errors");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_coordinate_errors, &coordinates, sizeof(coordinates)), "install audit coordinate errors");
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_visits, &visits, sizeof(visits)), "install audit visits");
        // Active pointer installed last; other symbols are never read unless set.
        owner_cuda_check(cudaMemcpyToSymbol(owner_eval_audit_errors, &errors, sizeof(errors)), "activate audit");
        installed_ = true;
    }
    ~OwnerEvalContributionAudit() {
        if (installed_) {
            float* none = nullptr;
            cudaMemcpyToSymbol(owner_eval_audit_errors, &none, sizeof(none));
        }
    }
    void save(const float* sums, const float* coordinate_sums) {
        if (!installed_) return;
        const uint64_t shape[3] = {(uint64_t)slots_, (uint64_t)queries_, (uint64_t)responses_};
        write_file(".shape.u64", shape, sizeof(shape));
        download(".predictions.f64", predictions_.get(), points_ * responses_);
        download(".weighted_predictions.f64", weighted_predictions_.get(), points_ * responses_);
        download(".weights.f64", weights_.get(), points_);
        download(".squared_errors.f64", squared_errors_.get(), points_);
        download(".errors.f32", errors_.get(), points_);
        download(".coordinate_errors.f32", coordinate_errors_.get(), points_ * responses_);
        download(".visits.u32", visits_.get(), points_);
        download(".sums.f32", sums, slots_ * 4);
        download(".coordinate_sums.f32", coordinate_sums, slots_ * responses_);
    }
};
