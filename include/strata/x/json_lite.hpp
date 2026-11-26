// include/strata/x/json_lite.hpp - moex (2026-11-24): a small strict JSON reader for the files a Hugging Face checkpoint
// carries (config.json, model.safetensors.index.json, a safetensors shard's header), so strata-x reads a release as it
// is published (moex/docs/ERNIE55_PLAN.md). Objects (members kept in file order), arrays, strings (every escape; \u with
// surrogate pairs to UTF-8), numbers (their double, and an exact int64 when written as an integer that fits), true,
// false, null. Strict RFC 8259: no comments, no trailing commas, no leading zeros, no NaN / Infinity, nesting at most 64.
// Numbers are read with std::from_chars, so the C locale's decimal point never matters.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::json {

struct Value {
    enum class Type : uint8_t { Null, Bool, Num, Str, Arr, Obj };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    int64_t i = 0;          ///< the number when written as an integer that fits in int64 (is_int)
    bool is_int = false;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;   ///< file order; find() is a linear scan

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_num() const { return type == Type::Num; }
    bool is_str() const { return type == Type::Str; }
    bool is_arr() const { return type == Type::Arr; }
    bool is_obj() const { return type == Type::Obj; }

    /// an object's member, or nullptr (also when this is not an object)
    const Value* find(const std::string& key) const;
    /// typed reads of an object's member: the default when absent or of another type (a number for an integer getter
    /// must be integral)
    double num_or(const std::string& key, double def) const;
    int64_t int_or(const std::string& key, int64_t def) const;
    bool bool_or(const std::string& key, bool def) const;
    std::string str_or(const std::string& key, const std::string& def) const;
};

/// Parses n bytes as one JSON value (whitespace around it allowed, nothing else after it). On failure: false, and err
/// says what and at which byte.
bool parse(const char* data, size_t n, Value& out, std::string& err);

}  // namespace strata::json
