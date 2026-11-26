// moex/ref/recorder.cpp - see recorder.hpp.
#include "recorder.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace moex {

// ------------------------------------------------------------------ recorder

bool Recorder::open(const std::string& dir, const std::string& match, std::string& err) {
    std::string cmd = "mkdir -p '" + dir + "'";
    if (system(cmd.c_str()) != 0) {
        err = "mkdir failed";
        return false;
    }
    bin_ = fopen((dir + "/dump.bin").c_str(), "wb");
    idx_ = fopen((dir + "/index.tsv").c_str(), "w");
    match_ = std::regex(match);
    match_all_ = match == ".";
    if (!bin_ || !idx_) {
        err = "cannot write " + dir;
        return false;
    }
    return true;
}

bool Recorder::attach_oracle(const std::string& dir, std::string& err) {
    FILE* f = fopen((dir + "/index.tsv").c_str(), "r");
    if (!f) {
        err = "no oracle index in " + dir;
        return false;
    }
    std::unordered_map<std::string, int> occ;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        char name[512], type[16];
        int step;
        long long n0, n1, n2, n3;
        unsigned long long off;
        if (sscanf(line, "%d\t%511[^\t]\t%15s\t%lld\t%lld\t%lld\t%lld\t%llu", &step, name, type, &n0, &n1, &n2, &n3, &off) != 8) continue;
        const std::string base = std::to_string(step) + "|" + name;
        const int o = occ[base]++;
        oracle_[base + "|" + std::to_string(o)] = ORec{off, n0 * n1 * n2 * n3, std::string(type) == "i32"};
    }
    fclose(f);
    oracle_fd_ = ::open((dir + "/dump.bin").c_str(), O_RDONLY);
    if (oracle_fd_ < 0) {
        err = "no oracle dump.bin in " + dir;
        return false;
    }
    oracle_bin_ = "";   // marks the oracle as attached
    return true;
}

void Recorder::close() {
    if (bin_) fclose(bin_);
    if (idx_) fclose(idx_);
    bin_ = idx_ = nullptr;
    if (oracle_fd_ >= 0) ::close(oracle_fd_);
    oracle_fd_ = -1;
}

void Recorder::set_step(int s) {
    step_ = s;
    occ_.clear();
}

bool Recorder::want(const std::string& n) const { return bin_ && (match_all_ || std::regex_search(n, match_)); }

void Recorder::write(const std::string& name, const char* type, const int64_t ne[4], const void* d, size_t bytes) {
    fwrite(d, 1, bytes, bin_);
    fprintf(idx_, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%llu\n", step_, name.c_str(), type, (long long) ne[0],
            (long long) ne[1], (long long) ne[2], (long long) ne[3], (unsigned long long) off_);
    off_ += bytes;
}

std::string Recorder::key(const std::string& n) {
    const int o = occ_[n]++;
    return std::to_string(step_) + "|" + n + "|" + std::to_string(o);
}

void Recorder::check(const std::string& stat_name, double rel) {
    auto it = stats_.find(stat_name);
    if (it == stats_.end()) {
        order_.push_back(stat_name);
        it = stats_.emplace(stat_name, Stat{}).first;
    }
    Stat& st = it->second;
    if (rel > st.worst || st.step < 0) {
        st.worst = rel;
        st.step = step_;
    }
    st.n++;
    st.sum += rel;
}

static std::string rec_name(const char* name, int il) {
    return il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name);
}

bool Recorder::wants(const char* name, int il) const { return want(rec_name(name, il)); }

void Recorder::f32(const char* name, int il, float* d, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const std::string n = rec_name(name, il);
    const int64_t cnt = ne0 * ne1 * ne2 * ne3;
    if (want(n)) {
        const int64_t ne[4] = {ne0, ne1, ne2, ne3};
        write(n, "f32", ne, d, sizeof(float) * cnt);
    }
    if (!oracle_bin_) return;
    const std::string k = key(n);
    auto it = oracle_.find(k);
    const int o = occ_[n] - 1;
    const std::string sn = o ? n + "#" + std::to_string(o) : n;
    if (it == oracle_.end() || it->second.i32) return;
    // the oracle pads some records (indexer_score: to a multiple of the pool block): compare the shared prefix
    if (it->second.n != cnt && n.rfind("indexer_score", 0) != 0) {
        check(sn + " (SHAPE " + std::to_string(cnt) + " vs " + std::to_string(it->second.n) + ")", 1.0);
        return;
    }
    const int64_t m = std::min<int64_t>(cnt, it->second.n);
    std::vector<float> y(m);
    if (pread(oracle_fd_, y.data(), m * sizeof(float), (off_t) it->second.off) != (ssize_t) (m * sizeof(float))) abort();
    double num = 0, den = 0;
    bool fin = true;
    for (int64_t i = 0; i < m; ++i) {
        if (!std::isfinite(y[i]) || !std::isfinite(d[i])) {
            if (std::isfinite(y[i]) != std::isfinite(d[i])) fin = false;
            continue;
        }
        const double e = (double) d[i] - (double) y[i];
        num += e * e;
        den += (double) y[i] * (double) y[i];
    }
    const double rel = !fin ? 1.0 : den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    check(sn, rel);
    std::copy(y.begin(), y.end(), d);   // teacher forcing
}

void Recorder::i32(const char* name, int il, int32_t* d, int64_t ne0, int64_t ne1) {
    const std::string n = rec_name(name, il);
    if (want(n)) {
        const int64_t ne[4] = {ne0, ne1, 1, 1};
        write(n, "i32", ne, d, sizeof(int32_t) * ne0 * ne1);
    }
    if (!oracle_bin_) return;
    const std::string k = key(n);
    auto it = oracle_.find(k);
    if (it == oracle_.end() || !it->second.i32 || it->second.n != ne0 * ne1) return;
    std::vector<int32_t> y(ne0 * ne1);
    if (pread(oracle_fd_, y.data(), y.size() * sizeof(int32_t), (off_t) it->second.off) != (ssize_t) (y.size() * sizeof(int32_t))) abort();
    std::vector<int32_t> a(d, d + ne0 * ne1), b = y;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    check(n, a == b ? 0.0 : 1.0);
    std::copy(y.begin(), y.end(), d);
}

int Recorder::report(double tol, FILE* out) const {
    int bad = 0;
    fprintf(out, "%-36s %11s %5s %11s %6s\n", "record (op output, oracle inputs)", "worst rel", "step", "mean rel", "steps");
    for (const auto& n : order_) {
        const Stat& st = stats_.at(n);
        const bool over = st.worst > tol;
        bad += over;
        fprintf(out, "%-36s %11.3e %5d %11.3e %6d%s\n", n.c_str(), st.worst, st.step, st.n ? st.sum / st.n : 0.0, st.n,
                over ? "   <-- over tol" : "");
    }
    return bad;
}

}  // namespace moex
