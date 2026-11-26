// tests/x/json_lite_test.cpp - moex (2026-11-24): include/strata/x/json_lite.hpp against cases written by hand -
// every escape, surrogate pairs, the number grammar's edges, nesting, and the strictness rules - no model, no GPU.
#include "strata/x/json_lite.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using strata::json::Value;

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

static bool parse(const std::string& s, Value& v, std::string& err) { return strata::json::parse(s.data(), s.size(), v, err); }

static void good(const std::string& s, const char* what) {
    Value v;
    std::string err;
    if (!parse(s, v, err)) {
        std::printf("FAIL: %s: %s\n", what, err.c_str());
        ++g_fail;
    }
}

static void bad(const std::string& s, const char* what) {
    Value v;
    std::string err;
    if (parse(s, v, err)) {
        std::printf("FAIL: %s: accepted\n", what);
        ++g_fail;
    }
}

int main() {
    Value v;
    std::string err;
    // a config.json-like object
    check(parse(R"({"a": 1, "b": -2.5e3, "c": true, "d": null, "e": [128, -2, 0], "f": {"g": "x"}, "h": 4096})", v, err), "object");
    check(v.int_or("a", 0) == 1 && v.num_or("b", 0) == -2500.0 && v.bool_or("c", false) && v.find("d")->is_null(), "members");
    check(v.find("e")->arr.size() == 3 && v.find("e")->arr[1].i == -2 && v.find("f")->str_or("g", "") == "x", "nested");
    check(v.int_or("b", 7) == 7, "a non-integer is not an int");
    check(v.int_or("zz", 9) == 9 && v.str_or("a", "dflt") == "dflt", "absent / wrong type: the default");
    // the integers of a shard header: exact past 2^32
    check(parse(R"({"data_offsets": [1951890376, 9007199254740993]})", v, err), "big ints");
    check(v.find("data_offsets")->arr[0].i == 1951890376LL && v.find("data_offsets")->arr[1].i == 9007199254740993LL, "exact int64");
    // strings: every escape, \u, a surrogate pair, raw UTF-8 passed through
    check(parse(R"(["\"\\\/\b\f\n\r\t", "é中", "😀", "é中"])", v, err), "strings");
    check(v.arr[0].str == "\"\\/\b\f\n\r\t", "escapes");
    check(v.arr[1].str == "\xc3\xa9\xe4\xb8\xad", "\\u to UTF-8");
    check(v.arr[2].str == "\xf0\x9f\x98\x80", "a surrogate pair");
    check(v.arr[3].str == "\xc3\xa9\xe4\xb8\xad", "raw UTF-8");
    // numbers
    good("0", "zero");
    good("-0", "minus zero");
    good("1e-6", "an exponent");
    good("1.0E+2", "E+");
    good("  [ ]  ", "whitespace around");
    check(parse("1e-6", v, err) && std::fabs(v.num - 1e-6) < 1e-21 && !v.is_int, "1e-6");
    check(parse("160000.0", v, err) && v.num == 160000.0 && !v.is_int, "a float written with a point is not an int");
    check(parse("-9223372036854775808", v, err) && v.is_int && v.i == INT64_MIN, "int64 min");
    check(parse("9223372036854775808", v, err) && !v.is_int && v.num == 9223372036854775808.0, "past int64: a double");
    // strictness
    bad("01", "a leading zero");
    bad("1.", "a bare point");
    bad(".5", "no integer part");
    bad("1e", "an empty exponent");
    bad("-", "a bare minus");
    bad("[1,]", "a trailing comma (array)");
    bad(R"({"a":1,})", "a trailing comma (object)");
    bad("{a:1}", "an unquoted key");
    bad(R"({"a" 1})", "a missing colon");
    bad("[1 2]", "a missing comma");
    bad("tru", "a short literal");
    bad("NaN", "NaN");
    bad("Infinity", "Infinity");
    bad("// c\n1", "a comment");
    bad("\"a\nb\"", "a raw newline in a string");
    bad(R"("\x")", "an unknown escape");
    bad(R"("\ud83d")", "a lone high surrogate");
    bad(R"("\ude00")", "a lone low surrogate");
    bad(R"("\ud83dA")", "a high surrogate with no low half");
    bad("\"abc", "an unterminated string");
    bad("[1] 2", "text after the value");
    bad("", "empty input");
    std::string deep(65, '[');
    deep += std::string(65, ']');
    bad(deep, "nesting deeper than 64");
    std::string ok64(64, '[');
    ok64 += std::string(64, ']');
    good(ok64, "nesting of 64");
    std::printf(g_fail ? "%d FAILED\n" : "json_lite: all cases pass\n", g_fail);
    return g_fail ? 1 : 0;
}
