// src/x/json_lite.cpp - moex (2026-11-24): include/strata/x/json_lite.hpp's reader.
#include "strata/x/json_lite.hpp"

#include <charconv>
#include <cstring>

namespace strata::json {

namespace {

constexpr int kMaxDepth = 64;

struct Parser {
    const char* p;
    const char* b;
    const char* e;
    std::string& err;

    bool fail(const char* what) {
        err = std::string(what) + " at byte " + std::to_string((size_t) (p - b));
        return false;
    }
    void ws() {
        while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool lit(const char* s, size_t n) {
        if ((size_t) (e - p) < n || std::memcmp(p, s, n) != 0) return fail("an unknown literal");
        p += n;
        return true;
    }
    static void utf8(std::string& o, uint32_t c) {
        if (c < 0x80) {
            o += (char) c;
        } else if (c < 0x800) {
            o += (char) (0xC0 | (c >> 6));
            o += (char) (0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            o += (char) (0xE0 | (c >> 12));
            o += (char) (0x80 | ((c >> 6) & 0x3F));
            o += (char) (0x80 | (c & 0x3F));
        } else {
            o += (char) (0xF0 | (c >> 18));
            o += (char) (0x80 | ((c >> 12) & 0x3F));
            o += (char) (0x80 | ((c >> 6) & 0x3F));
            o += (char) (0x80 | (c & 0x3F));
        }
    }
    bool hex4(uint32_t& v) {
        if (e - p < 4) return fail("a short \\u escape");
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return fail("a bad hex digit in a \\u escape");
        }
        return true;
    }
    bool string(std::string& o) {
        ++p;   // the opening quote
        o.clear();
        for (;;) {
            if (p >= e) return fail("an unterminated string");
            const unsigned char c = (unsigned char) *p;
            if (c == '"') {
                ++p;
                return true;
            }
            if (c < 0x20) return fail("a control character in a string");
            if (c != '\\') {   // a run of plain bytes (UTF-8 passes through as it is)
                const char* s = p;
                while (p < e && *p != '"' && *p != '\\' && (unsigned char) *p >= 0x20) ++p;
                o.append(s, (size_t) (p - s));
                continue;
            }
            if (++p >= e) return fail("an unterminated escape");
            const char x = *p++;
            switch (x) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t u = 0;
                    if (!hex4(u)) return false;
                    if (u >= 0xD800 && u <= 0xDBFF) {   // a high surrogate: its low half must follow
                        uint32_t lo = 0;
                        if (e - p < 2 || p[0] != '\\' || p[1] != 'u') return fail("a lone high surrogate");
                        p += 2;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo > 0xDFFF) return fail("a high surrogate without its low half");
                        u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (u >= 0xDC00 && u <= 0xDFFF) {
                        return fail("a lone low surrogate");
                    }
                    utf8(o, u);
                    break;
                }
                default: return fail("an unknown escape");
            }
        }
    }
    bool number(Value& v) {
        const char* s = p;
        if (p < e && *p == '-') ++p;
        if (p >= e) return fail("a bare minus sign");
        if (*p == '0') {
            ++p;
            if (p < e && *p >= '0' && *p <= '9') return fail("a leading zero");
        } else if (*p >= '1' && *p <= '9') {
            while (p < e && *p >= '0' && *p <= '9') ++p;
        } else {
            return fail("a number without digits");
        }
        bool integral = true;
        if (p < e && *p == '.') {
            integral = false;
            ++p;
            if (p >= e || *p < '0' || *p > '9') return fail("no digits after a decimal point");
            while (p < e && *p >= '0' && *p <= '9') ++p;
        }
        if (p < e && (*p == 'e' || *p == 'E')) {
            integral = false;
            ++p;
            if (p < e && (*p == '+' || *p == '-')) ++p;
            if (p >= e || *p < '0' || *p > '9') return fail("no digits in an exponent");
            while (p < e && *p >= '0' && *p <= '9') ++p;
        }
        v.type = Value::Type::Num;
        if (integral) {
            int64_t iv = 0;
            const auto r = std::from_chars(s, p, iv);
            if (r.ec == std::errc() && r.ptr == p) {
                v.i = iv;
                v.is_int = true;
                v.num = (double) iv;
                return true;
            }
        }
        double d = 0.0;
        const auto r = std::from_chars(s, p, d);
        if (r.ec != std::errc() || r.ptr != p) return fail("a number out of range");
        v.num = d;
        return true;
    }
    bool value(Value& v, int depth) {
        if (depth >= kMaxDepth) return fail("nesting deeper than 64");   // (depth 0 is the outermost value)
        ws();
        if (p >= e) return fail("a missing value");
        switch (*p) {
            case '{': {
                ++p;
                v.type = Value::Type::Obj;
                ws();
                if (p < e && *p == '}') {
                    ++p;
                    return true;
                }
                for (;;) {
                    ws();
                    if (p >= e || *p != '"') return fail("an object key that is not a string");
                    std::string k;
                    if (!string(k)) return false;
                    ws();
                    if (p >= e || *p != ':') return fail("a missing colon");
                    ++p;
                    v.obj.emplace_back(std::move(k), Value{});
                    if (!value(v.obj.back().second, depth + 1)) return false;
                    ws();
                    if (p < e && *p == ',') {
                        ++p;
                        continue;
                    }
                    if (p < e && *p == '}') {
                        ++p;
                        return true;
                    }
                    return fail("a missing comma or closing brace");
                }
            }
            case '[': {
                ++p;
                v.type = Value::Type::Arr;
                ws();
                if (p < e && *p == ']') {
                    ++p;
                    return true;
                }
                for (;;) {
                    v.arr.emplace_back();
                    if (!value(v.arr.back(), depth + 1)) return false;
                    ws();
                    if (p < e && *p == ',') {
                        ++p;
                        continue;
                    }
                    if (p < e && *p == ']') {
                        ++p;
                        return true;
                    }
                    return fail("a missing comma or closing bracket");
                }
            }
            case '"': v.type = Value::Type::Str; return string(v.str);
            case 't': v.type = Value::Type::Bool; v.b = true; return lit("true", 4);
            case 'f': v.type = Value::Type::Bool; v.b = false; return lit("false", 5);
            case 'n': v.type = Value::Type::Null; return lit("null", 4);
            default: return number(v);
        }
    }
};

}  // namespace

const Value* Value::find(const std::string& key) const {
    if (type != Type::Obj) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

double Value::num_or(const std::string& key, double def) const {
    const Value* v = find(key);
    return v && v->is_num() ? v->num : def;
}

int64_t Value::int_or(const std::string& key, int64_t def) const {
    const Value* v = find(key);
    return v && v->is_num() && v->is_int ? v->i : def;
}

bool Value::bool_or(const std::string& key, bool def) const {
    const Value* v = find(key);
    return v && v->is_bool() ? v->b : def;
}

std::string Value::str_or(const std::string& key, const std::string& def) const {
    const Value* v = find(key);
    return v && v->is_str() ? v->str : def;
}

bool parse(const char* data, size_t n, Value& out, std::string& err) {
    out = Value{};
    Parser ps{data, data, data + n, err};
    if (!ps.value(out, 0)) return false;
    ps.ws();
    if (ps.p != ps.e) return ps.fail("text after the value");
    return true;
}

}  // namespace strata::json
