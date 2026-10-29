// Minimal JSON reader for case files (objects, arrays, numbers, strings, true,
// false, null). Enough for app/bc_case's input; no writer, no unicode escapes
// beyond \uXXXX in the ASCII range.
#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace igabem {

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  Type type = Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  static Json parse(const std::string& text) {
    size_t i = 0;
    Json j = value(text, i);
    skip(text, i);
    if (i != text.size()) fail(text, i, "trailing characters");
    return j;
  }
  static Json load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
  }

  bool isNull() const { return type == Null; }
  bool isObject() const { return type == Object; }
  bool isArray() const { return type == Array; }
  bool isNumber() const { return type == Number; }
  bool isString() const { return type == String; }
  bool has(const std::string& k) const { return type == Object && obj.count(k) && !obj.at(k).isNull(); }
  const Json& operator[](const std::string& k) const {
    static const Json null;
    if (type != Object) throw std::runtime_error("json: not an object (key " + k + ")");
    auto it = obj.find(k);
    return it == obj.end() ? null : it->second;
  }
  const Json& operator[](size_t i) const { return arr.at(i); }
  size_t size() const { return type == Array ? arr.size() : obj.size(); }

  double asNumber() const {
    if (type != Number) throw std::runtime_error("json: number expected");
    return num;
  }
  int asInt() const { return int(asNumber()); }
  bool asBool() const {
    if (type != Bool) throw std::runtime_error("json: boolean expected");
    return b;
  }
  const std::string& asString() const {
    if (type != String) throw std::runtime_error("json: string expected");
    return str;
  }
  std::vector<double> asNumbers() const {
    if (type == Number) return {num};
    std::vector<double> v;
    for (const Json& x : arr) v.push_back(x.asNumber());
    return v;
  }
  double get(const std::string& k, double dflt) const { return has(k) ? (*this)[k].asNumber() : dflt; }
  std::string get(const std::string& k, const std::string& dflt) const {
    return has(k) ? (*this)[k].asString() : dflt;
  }

 private:
  [[noreturn]] static void fail(const std::string& t, size_t i, const std::string& what) {
    size_t line = 1;
    for (size_t k = 0; k < i && k < t.size(); ++k) line += t[k] == '\n';
    throw std::runtime_error("json: " + what + " at line " + std::to_string(line));
  }
  static void skip(const std::string& t, size_t& i) {
    for (;;) {
      while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
      if (i + 1 < t.size() && t[i] == '/' && t[i + 1] == '/') {  // // comments, for hand-written cases
        while (i < t.size() && t[i] != '\n') ++i;
        continue;
      }
      return;
    }
  }
  static std::string string(const std::string& t, size_t& i) {
    if (t[i] != '"') fail(t, i, "string expected");
    std::string s;
    for (++i; i < t.size() && t[i] != '"'; ++i) {
      if (t[i] != '\\') { s += t[i]; continue; }
      if (++i >= t.size()) break;
      switch (t[i]) {
        case 'n': s += '\n'; break;
        case 't': s += '\t'; break;
        case 'r': s += '\r'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'u': s += char(std::strtol(t.substr(i + 1, 4).c_str(), nullptr, 16)); i += 4; break;
        default: s += t[i];
      }
    }
    if (i >= t.size()) fail(t, i, "unterminated string");
    ++i;
    return s;
  }
  static Json value(const std::string& t, size_t& i) {
    skip(t, i);
    if (i >= t.size()) fail(t, i, "unexpected end");
    Json j;
    const char c = t[i];
    if (c == '{') {
      j.type = Object;
      ++i;
      skip(t, i);
      if (t[i] == '}') { ++i; return j; }
      for (;;) {
        skip(t, i);
        const std::string k = string(t, i);
        skip(t, i);
        if (t[i] != ':') fail(t, i, "':' expected");
        ++i;
        j.obj[k] = value(t, i);
        skip(t, i);
        if (t[i] == ',') { ++i; continue; }
        if (t[i] == '}') { ++i; return j; }
        fail(t, i, "',' or '}' expected");
      }
    }
    if (c == '[') {
      j.type = Array;
      ++i;
      skip(t, i);
      if (t[i] == ']') { ++i; return j; }
      for (;;) {
        j.arr.push_back(value(t, i));
        skip(t, i);
        if (t[i] == ',') { ++i; continue; }
        if (t[i] == ']') { ++i; return j; }
        fail(t, i, "',' or ']' expected");
      }
    }
    if (c == '"') { j.type = String; j.str = string(t, i); return j; }
    if (t.compare(i, 4, "true") == 0) { j.type = Bool; j.b = true; i += 4; return j; }
    if (t.compare(i, 5, "false") == 0) { j.type = Bool; i += 5; return j; }
    if (t.compare(i, 4, "null") == 0) { i += 4; return j; }
    char* end = nullptr;
    j.num = std::strtod(t.c_str() + i, &end);
    if (end == t.c_str() + i) fail(t, i, "value expected");
    j.type = Number;
    i = size_t(end - t.c_str());
    return j;
  }
};

}  // namespace igabem
