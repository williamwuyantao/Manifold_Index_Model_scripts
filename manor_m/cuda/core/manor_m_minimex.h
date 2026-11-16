#ifndef MANOR_M_MINIMEX_H
#define MANOR_M_MINIMEX_H

/*
 * A deliberately tiny, private mxArray compatibility layer used only while
 * the verified Phase-2 execution body is crossing into the shared library.
 * It is not part of the public ABI and contains no MATLAB code or linkage.
 *
 * The transition keeps one source of truth for every CUDA launch and output
 * packing rule.  The public C adapter builds these private array views, calls
 * the unchanged execution body, copies requested fields to caller-owned
 * buffers, and destroys the views.  Once Phase-2 three-way parity is locked,
 * the host body can be refactored into direct descriptors without changing
 * kernels and numerical order at the same time.
 */

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using mwSize = std::size_t;
using mwIndex = std::size_t;
using mxLogical = uint8_t;
using uint64_T = uint64_t;

enum mxClassID {
    mxUNKNOWN_CLASS = 0,
    mxCELL_CLASS,
    mxSTRUCT_CLASS,
    mxLOGICAL_CLASS,
    mxCHAR_CLASS,
    mxDOUBLE_CLASS,
    mxSINGLE_CLASS,
    mxINT8_CLASS,
    mxUINT8_CLASS,
    mxINT16_CLASS,
    mxUINT16_CLASS,
    mxINT32_CLASS,
    mxUINT32_CLASS,
    mxINT64_CLASS,
    mxUINT64_CLASS
};

constexpr int mxREAL = 0;

struct ManorMMiniMexError : std::runtime_error {
    ManorMMiniMexError(std::string identifier, std::string message)
        : std::runtime_error(std::move(message)), identifier_(std::move(identifier)) {}
    const std::string& identifier() const noexcept { return identifier_; }
private:
    std::string identifier_;
};

struct mxArray {
    mxClassID class_id = mxUNKNOWN_CLASS;
    std::vector<mwSize> dimensions{0, 0};
    void* data = nullptr;
    bool owns_data = false;
    bool complex = false;
    bool sparse = false;
    std::unordered_map<std::string, mxArray*> fields;
    std::vector<mxArray*> cells;
    std::string text;

    ~mxArray()
    {
        if (owns_data) ::operator delete(data);
        for (auto& entry : fields) delete entry.second;
        for (mxArray* value : cells) delete value;
    }
};

inline size_t manor_m_minimex_class_bytes(mxClassID class_id)
{
    switch (class_id) {
    case mxLOGICAL_CLASS:
    case mxINT8_CLASS:
    case mxUINT8_CLASS: return 1;
    case mxINT16_CLASS:
    case mxUINT16_CLASS: return 2;
    case mxSINGLE_CLASS:
    case mxINT32_CLASS:
    case mxUINT32_CLASS: return 4;
    case mxDOUBLE_CLASS:
    case mxINT64_CLASS:
    case mxUINT64_CLASS: return 8;
    default: return 0;
    }
}

inline size_t manor_m_minimex_element_count(const std::vector<mwSize>& dimensions)
{
    size_t count = 1;
    for (mwSize dimension : dimensions) {
        if (dimension != 0 && count > std::numeric_limits<size_t>::max() / dimension)
            throw ManorMMiniMexError("manor_m:minimex:overflow", "array size exceeds SIZE_MAX");
        count *= dimension;
    }
    return count;
}

inline mxArray* manor_m_minimex_view(
    mxClassID class_id, void* data, std::vector<mwSize> dimensions)
{
    auto* result = new mxArray;
    result->class_id = class_id;
    result->data = data;
    result->dimensions = std::move(dimensions);
    return result;
}

inline mxArray* mxCreateNumericArray(
    mwSize dimension_count, const mwSize* dimensions, mxClassID class_id, int)
{
    auto* result = new mxArray;
    result->class_id = class_id;
    result->dimensions.assign(dimensions, dimensions + dimension_count);
    const size_t bytes = manor_m_minimex_element_count(result->dimensions)
        * manor_m_minimex_class_bytes(class_id);
    if (bytes > 0) {
        result->data = ::operator new(bytes);
        result->owns_data = true;
        std::memset(result->data, 0, bytes);
    }
    return result;
}

inline mxArray* mxCreateNumericMatrix(
    mwSize rows, mwSize columns, mxClassID class_id, int complexity)
{
    const mwSize dimensions[] = {rows, columns};
    return mxCreateNumericArray(2, dimensions, class_id, complexity);
}

struct ManorMMiniMexOutputBinding {
    void* data;
    size_t bytes;
    mxClassID class_id;
};
inline thread_local std::unordered_map<std::string, ManorMMiniMexOutputBinding>
    manor_m_minimex_output_bindings;

inline mxArray* manor_m_minimex_named_output(
    const char* name, mwSize rank, const mwSize* dimensions, mxClassID class_id,
    bool zero_fill)
{
    const auto found = manor_m_minimex_output_bindings.find(name);
    if (found == manor_m_minimex_output_bindings.end())
        return mxCreateNumericArray(rank, dimensions, class_id, mxREAL);
    std::vector<mwSize> shape(dimensions, dimensions + rank);
    const size_t count = manor_m_minimex_element_count(shape);
    const size_t element_bytes = manor_m_minimex_class_bytes(class_id);
    if (element_bytes && count > std::numeric_limits<size_t>::max() / element_bytes)
        throw ManorMMiniMexError("manor_m_cuda:outputBuffer", "Caller output size overflow.");
    const size_t bytes = count * element_bytes;
    if (!element_bytes || (bytes && !found->second.data) ||
        found->second.class_id != class_id || bytes != found->second.bytes)
        throw ManorMMiniMexError("manor_m_cuda:outputBuffer", "Named caller output has the wrong layout.");
    if (bytes && zero_fill) std::memset(found->second.data, 0, bytes);
    return manor_m_minimex_view(class_id, found->second.data, std::move(shape));
}

inline mxArray* mxCreateDoubleMatrix(mwSize rows, mwSize columns, int complexity)
{
    return mxCreateNumericMatrix(rows, columns, mxDOUBLE_CLASS, complexity);
}

inline mxArray* mxCreateDoubleScalar(double value)
{
    mxArray* result = mxCreateDoubleMatrix(1, 1, mxREAL);
    static_cast<double*>(result->data)[0] = value;
    return result;
}

inline mxArray* mxCreateLogicalArray(mwSize count, const mwSize* dimensions)
{
    return mxCreateNumericArray(count, dimensions, mxLOGICAL_CLASS, mxREAL);
}

inline mxArray* mxCreateLogicalMatrix(mwSize rows, mwSize columns)
{
    const mwSize dimensions[] = {rows, columns};
    return mxCreateLogicalArray(2, dimensions);
}

inline mxArray* mxCreateLogicalScalar(bool value)
{
    mxArray* result = mxCreateLogicalMatrix(1, 1);
    static_cast<mxLogical*>(result->data)[0] = value ? 1u : 0u;
    return result;
}

inline mxArray* mxCreateStructMatrix(
    mwSize rows, mwSize columns, int field_count, const char* const* fields)
{
    auto* result = new mxArray;
    result->class_id = mxSTRUCT_CLASS;
    result->dimensions = {rows, columns};
    for (int index = 0; index < field_count; ++index)
        result->fields.emplace(fields[index], nullptr);
    return result;
}

inline mxArray* mxCreateCellMatrix(mwSize rows, mwSize columns)
{
    auto* result = new mxArray;
    result->class_id = mxCELL_CLASS;
    result->dimensions = {rows, columns};
    result->cells.resize(rows * columns, nullptr);
    return result;
}

inline mxArray* mxCreateString(const char* text)
{
    auto* result = new mxArray;
    result->class_id = mxCHAR_CLASS;
    result->text = text ? text : "";
    result->dimensions = {1, result->text.size()};
    return result;
}

inline void mxSetField(mxArray* array, mwIndex, const char* name, mxArray* value)
{
    auto found = array->fields.find(name);
    if (found != array->fields.end() && found->second != value) delete found->second;
    array->fields[name] = value;
}

inline const mxArray* mxGetField(const mxArray* array, mwIndex, const char* name)
{
    if (!array || array->class_id != mxSTRUCT_CLASS) return nullptr;
    const auto found = array->fields.find(name);
    return found == array->fields.end() ? nullptr : found->second;
}

inline mxArray* mxGetField(mxArray* array, mwIndex index, const char* name)
{
    return const_cast<mxArray*>(mxGetField(
        static_cast<const mxArray*>(array), index, name));
}

inline void mxSetCell(mxArray* array, mwIndex index, mxArray* value)
{
    if (index >= array->cells.size())
        throw ManorMMiniMexError("manor_m:minimex:cell", "cell index out of range");
    delete array->cells[index];
    array->cells[index] = value;
}

inline void mxDestroyArray(mxArray* array) { delete array; }
inline void* mxGetData(const mxArray* array) { return array ? array->data : nullptr; }
inline double* mxGetPr(const mxArray* array)
{
    return array ? static_cast<double*>(array->data) : nullptr;
}
inline mxLogical* mxGetLogicals(const mxArray* array)
{
    return array ? static_cast<mxLogical*>(array->data) : nullptr;
}
inline mxClassID mxGetClassID(const mxArray* array)
{
    return array ? array->class_id : mxUNKNOWN_CLASS;
}
inline mwSize mxGetNumberOfDimensions(const mxArray* array)
{
    return array ? array->dimensions.size() : 0;
}
inline const mwSize* mxGetDimensions(const mxArray* array)
{
    return array ? array->dimensions.data() : nullptr;
}
inline mwSize mxGetNumberOfElements(const mxArray* array)
{
    return array ? manor_m_minimex_element_count(array->dimensions) : 0;
}
inline mwSize mxGetM(const mxArray* array)
{
    return !array || array->dimensions.empty() ? 0 : array->dimensions[0];
}
inline mwSize mxGetN(const mxArray* array)
{
    if (!array || array->dimensions.size() < 2) return 1;
    size_t columns = 1;
    for (size_t index = 1; index < array->dimensions.size(); ++index)
        columns *= array->dimensions[index];
    return columns;
}

inline bool mxIsComplex(const mxArray* array) { return array && array->complex; }
inline bool mxIsSparse(const mxArray* array) { return array && array->sparse; }
inline bool mxIsSingle(const mxArray* array) { return array && array->class_id == mxSINGLE_CLASS; }
inline bool mxIsDouble(const mxArray* array) { return array && array->class_id == mxDOUBLE_CLASS; }
inline bool mxIsInt32(const mxArray* array) { return array && array->class_id == mxINT32_CLASS; }
inline bool mxIsUint32(const mxArray* array) { return array && array->class_id == mxUINT32_CLASS; }
inline bool mxIsLogical(const mxArray* array) { return array && array->class_id == mxLOGICAL_CLASS; }
inline bool mxIsStruct(const mxArray* array) { return array && array->class_id == mxSTRUCT_CLASS; }
inline bool mxIsLogicalScalar(const mxArray* array)
{
    return mxIsLogical(array) && mxGetNumberOfElements(array) == 1;
}
inline bool mxIsLogicalScalarTrue(const mxArray* array)
{
    return mxIsLogicalScalar(array) && mxGetLogicals(array)[0] != 0;
}
inline bool mxIsNumeric(const mxArray* array)
{
    if (!array) return false;
    return array->class_id >= mxDOUBLE_CLASS && array->class_id <= mxUINT64_CLASS;
}

inline double mxGetScalar(const mxArray* array)
{
    if (!array || mxGetNumberOfElements(array) == 0) return 0.0;
    switch (array->class_id) {
    case mxLOGICAL_CLASS: return static_cast<const mxLogical*>(array->data)[0];
    case mxDOUBLE_CLASS: return static_cast<const double*>(array->data)[0];
    case mxSINGLE_CLASS: return static_cast<const float*>(array->data)[0];
    case mxINT8_CLASS: return static_cast<const int8_t*>(array->data)[0];
    case mxUINT8_CLASS: return static_cast<const uint8_t*>(array->data)[0];
    case mxINT16_CLASS: return static_cast<const int16_t*>(array->data)[0];
    case mxUINT16_CLASS: return static_cast<const uint16_t*>(array->data)[0];
    case mxINT32_CLASS: return static_cast<const int32_t*>(array->data)[0];
    case mxUINT32_CLASS: return static_cast<const uint32_t*>(array->data)[0];
    case mxINT64_CLASS: return static_cast<double>(static_cast<const int64_t*>(array->data)[0]);
    case mxUINT64_CLASS: return static_cast<double>(static_cast<const uint64_t*>(array->data)[0]);
    default: return 0.0;
    }
}

[[noreturn]] inline void mexErrMsgIdAndTxt(
    const char* identifier, const char* format, ...)
{
    char message[2048];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    throw ManorMMiniMexError(identifier ? identifier : "manor_m:minimex", message);
}

inline int mexPrintf(const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    const int result = std::vfprintf(stdout, format, arguments);
    va_end(arguments);
    return result;
}

#endif
