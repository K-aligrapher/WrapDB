#include <charconv>
#include <set>

#include "sql/ast.h"

namespace warpdb {

ExprP make_expr(Expr::Kind k) {
  auto e = std::make_shared<Expr>();
  e->kind = k;
  return e;
}

const std::string& Expr::str() const {
  if (!cache_.empty()) return cache_;
  std::string s;
  auto join = [&](size_t from) {
    std::string r;
    for (size_t i = from; i < args.size(); ++i) {
      if (i > from) r += ",";
      r += args[i]->str();
    }
    return r;
  };
  switch (kind) {
    case Col:
      s = tbl >= 0 ? "t" + std::to_string(tbl) + ".c" + std::to_string(col) : (table.empty() ? "" : table + ".") + name;
      break;
    case Int: s = std::to_string(ival); break;
    case Float: {
      char b[48];
      std::snprintf(b, sizeof b, "%.17g", fval);
      s = std::string(b) + "f";
      break;
    }
    case Str: s = "'" + name + "'"; break;
    case Date: s = "date(" + std::to_string(ival) + ")"; break;
    case Interval: s = "interval(" + std::to_string(ival) + name + ")"; break;
    case Bin: s = "(" + args[0]->str() + " " + name + " " + args[1]->str() + ")"; break;
    case Not: s = "not(" + args[0]->str() + ")"; break;
    case Neg: s = "neg(" + args[0]->str() + ")"; break;
    case Between: s = std::string(negated ? "not_" : "") + "between(" + join(0) + ")"; break;
    case In: s = std::string(negated ? "not_" : "") + "in(" + join(0) + ")"; break;
    case Like: s = std::string(negated ? "not_" : "") + "like(" + join(0) + ")"; break;
    case Case: s = "case(" + join(0) + ")"; break;
    case Func:
    case Agg: s = name + "(" + join(0) + ")"; break;
    case Star: s = "*"; break;
  }
  cache_ = s.empty() ? "?" : s;
  return cache_;
}

namespace {

struct Token {
  enum K { Ident, Num, Str, Sym, End } kind = End;
  std::string text;  // identifiers lowercased; strings unescaped
  bool quoted = false;
  size_t pos = 0, end = 0;
};

std::vector<Token> lex(const std::string& s) {
  std::vector<Token> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    char c = s[i];
    if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
    if (c == '-' && i + 1 < n && s[i + 1] == '-') {
      while (i < n && s[i] != '\n') ++i;
      continue;
    }
    Token t;
    t.pos = i;
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t j = i;
      while (j < n && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_' || s[j] == '$')) ++j;
      t.kind = Token::Ident;
      t.text = to_lower(s.substr(i, j - i));
      i = j;
    } else if (c == '"') {
      size_t j = s.find('"', i + 1);
      if (j == std::string::npos) throw Error("unterminated quoted identifier");
      t.kind = Token::Ident;
      t.quoted = true;
      t.text = to_lower(s.substr(i + 1, j - i - 1));
      i = j + 1;
    } else if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
      size_t j = i;
      while (j < n && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      if (j < n && s[j] == '.') {
        ++j;
        while (j < n && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      }
      if (j < n && (s[j] == 'e' || s[j] == 'E')) {
        size_t k = j + 1;
        if (k < n && (s[k] == '+' || s[k] == '-')) ++k;
        if (k < n && std::isdigit(static_cast<unsigned char>(s[k]))) {
          while (k < n && std::isdigit(static_cast<unsigned char>(s[k]))) ++k;
          j = k;
        }
      }
      t.kind = Token::Num;
      t.text = s.substr(i, j - i);
      i = j;
    } else if (c == '\'') {
      std::string v;
      size_t j = i + 1;
      for (;;) {
        if (j >= n) throw Error("unterminated string literal");
        if (s[j] == '\'') {
          if (j + 1 < n && s[j + 1] == '\'') { v += '\''; j += 2; continue; }
          break;
        }
        v += s[j++];
      }
      t.kind = Token::Str;
      t.text = v;
      i = j + 1;
    } else {
      t.kind = Token::Sym;
      if (i + 1 < n) {
        std::string two = s.substr(i, 2);
        if (two == "<=" || two == ">=" || two == "<>" || two == "!=" || two == "||") {
          t.text = two == "!=" ? "<>" : two;
          i += 2;
          t.end = i;
          out.push_back(t);
          continue;
        }
      }
      t.text = std::string(1, c);
      ++i;
    }
    t.end = i;
    out.push_back(std::move(t));
  }
  Token e;
  e.kind = Token::End;
  e.pos = e.end = n;
  out.push_back(e);
  return out;
}

const std::set<std::string>& reserved() {
  static const std::set<std::string> r = {
      "select", "from", "where", "group", "by", "order", "having", "limit", "join", "inner", "left", "right",
      "outer", "cross", "full", "on", "union", "and", "or", "not", "as", "asc", "desc", "case", "when", "then",
      "else", "end", "between", "in", "like", "is", "null", "distinct", "explain", "analyze", "using"};
  return r;
}

class Parser {
 public:
  explicit Parser(const std::string& sql) : sql_(sql), t_(lex(sql)) {}

  Select parse() {
    Select s;
    if (kw("explain")) {
      ++p_;
      s.explain = true;
      if (kw("analyze")) { ++p_; s.analyze = true; }
    }
    expect_kw("select");
    if (kw("distinct")) { ++p_; s.distinct = true; }
    parse_items(s);
    expect_kw("from");
    parse_from(s);
    if (kw("where")) { ++p_; s.where = parse_expr(); }
    if (kw("group")) {
      ++p_;
      expect_kw("by");
      do { s.group_by.push_back(parse_expr()); } while (sym(","));
    }
    if (kw("having")) { ++p_; s.having = parse_expr(); }
    if (kw("order")) {
      ++p_;
      expect_kw("by");
      do {
        OrderItem o;
        o.e = parse_expr();
        if (kw("asc")) ++p_;
        else if (kw("desc")) { ++p_; o.desc = true; }
        s.order_by.push_back(o);
      } while (sym(","));
    }
    if (kw("limit")) {
      ++p_;
      if (cur().kind != Token::Num) fail("expected number after LIMIT");
      s.limit = std::stoll(cur().text);
      ++p_;
    }
    while (cur().kind == Token::Sym && cur().text == ";") ++p_;
    if (cur().kind != Token::End) fail("unexpected token");
    return s;
  }

 private:
  const std::string& sql_;
  std::vector<Token> t_;
  size_t p_ = 0;

  const Token& cur() const { return t_[p_]; }
  bool kw(const char* w) const { return cur().kind == Token::Ident && !cur().quoted && cur().text == w; }
  bool sym(const char* s) {
    if (cur().kind == Token::Sym && cur().text == s) { ++p_; return true; }
    return false;
  }
  bool peek_sym(const char* s) const { return cur().kind == Token::Sym && cur().text == s; }
  [[noreturn]] void fail(const std::string& msg) const {
    std::string near = cur().kind == Token::End ? "end of input" : "'" + sql_.substr(cur().pos, cur().end - cur().pos) + "'";
    throw Error("syntax error: " + msg + " near " + near + " (position " + std::to_string(cur().pos) + ")");
  }
  void expect_kw(const char* w) {
    if (!kw(w)) fail(std::string("expected ") + w);
    ++p_;
  }
  void expect_sym(const char* s) {
    if (!sym(s)) fail(std::string("expected '") + s + "'");
  }
  std::string ident() {
    if (cur().kind != Token::Ident) fail("expected identifier");
    return t_[p_++].text;
  }
  bool alias_follows() const {
    return cur().kind == Token::Ident && (cur().quoted || !reserved().count(cur().text));
  }

  void parse_items(Select& s) {
    if (peek_sym("*")) {
      ++p_;
      s.star = true;
      return;
    }
    do {
      SelectItem it;
      size_t b = cur().pos;
      it.e = parse_expr();
      it.text = sql_.substr(b, t_[p_ - 1].end - b);
      if (kw("as")) { ++p_; it.alias = ident(); }
      else if (alias_follows()) it.alias = ident();
      s.items.push_back(it);
    } while (sym(","));
  }

  void parse_table(Select& s) {
    TableRef r;
    r.name = ident();
    if (kw("as")) { ++p_; r.alias = ident(); }
    else if (alias_follows()) r.alias = ident();
    if (r.alias.empty()) r.alias = r.name;
    s.from.push_back(r);
  }

  void parse_from(Select& s) {
    parse_table(s);
    for (;;) {
      if (sym(",")) { parse_table(s); continue; }
      if (kw("inner")) ++p_;
      else if (kw("left") || kw("right") || kw("full") || kw("outer") || kw("cross"))
        fail("only INNER JOIN is supported");
      if (kw("join")) {
        ++p_;
        parse_table(s);
        expect_kw("on");
        s.on_conds.push_back(parse_expr());
        continue;
      }
      break;
    }
  }

  ExprP bin(const std::string& op, ExprP a, ExprP b) {
    auto e = make_expr(Expr::Bin);
    e->name = op;
    e->args = {std::move(a), std::move(b)};
    return e;
  }

  ExprP parse_expr() { return parse_or(); }

  ExprP parse_or() {
    auto l = parse_and();
    while (kw("or")) { ++p_; l = bin("or", l, parse_and()); }
    return l;
  }
  ExprP parse_and() {
    auto l = parse_not();
    while (kw("and")) { ++p_; l = bin("and", l, parse_not()); }
    return l;
  }
  ExprP parse_not() {
    if (kw("not")) {
      ++p_;
      auto e = make_expr(Expr::Not);
      e->args = {parse_not()};
      return e;
    }
    return parse_cmp();
  }
  ExprP parse_cmp() {
    auto l = parse_add();
    static const char* ops[] = {"=", "<>", "<=", ">=", "<", ">"};
    for (auto op : ops)
      if (peek_sym(op)) {
        ++p_;
        return bin(op, l, parse_add());
      }
    bool neg = false;
    size_t save = p_;
    if (kw("not")) { neg = true; ++p_; }
    if (kw("between")) {
      ++p_;
      auto e = make_expr(Expr::Between);
      e->negated = neg;
      auto lo = parse_add();
      expect_kw("and");
      auto hi = parse_add();
      e->args = {l, lo, hi};
      return e;
    }
    if (kw("in")) {
      ++p_;
      expect_sym("(");
      if (kw("select")) fail("subqueries are not supported yet");
      auto e = make_expr(Expr::In);
      e->negated = neg;
      e->args.push_back(l);
      do { e->args.push_back(parse_add()); } while (sym(","));
      expect_sym(")");
      return e;
    }
    if (kw("like")) {
      ++p_;
      auto e = make_expr(Expr::Like);
      e->negated = neg;
      e->args = {l, parse_add()};
      return e;
    }
    if (neg) p_ = save;
    return l;
  }

  static ExprP fold_date(const ExprP& d, const ExprP& iv, bool sub) {
    int y;
    unsigned m, dd;
    civil_from_days(static_cast<int32_t>(d->ival), y, m, dd);
    int64_t n = sub ? -iv->ival : iv->ival;
    auto e = make_expr(Expr::Date);
    if (iv->name == "day") {
      e->ival = d->ival + n;
    } else {
      int64_t months = (iv->name == "year" ? 12 : 1) * n;
      int64_t tot = static_cast<int64_t>(y) * 12 + (m - 1) + months;
      int ny = static_cast<int>(tot >= 0 ? tot / 12 : -((-tot + 11) / 12));
      unsigned nm = static_cast<unsigned>(tot - static_cast<int64_t>(ny) * 12) + 1;
      e->ival = days_from_civil(ny, nm, std::min(dd, days_in_month(ny, nm)));
    }
    return e;
  }

  ExprP parse_add() {
    auto l = parse_mul();
    for (;;) {
      std::string op;
      if (peek_sym("+")) op = "+";
      else if (peek_sym("-")) op = "-";
      else break;
      ++p_;
      auto r = parse_mul();
      if (l->kind == Expr::Date && r->kind == Expr::Interval) l = fold_date(l, r, op == "-");
      else l = bin(op, l, r);
    }
    return l;
  }
  ExprP parse_mul() {
    auto l = parse_unary();
    for (;;) {
      std::string op;
      if (peek_sym("*")) op = "*";
      else if (peek_sym("/")) op = "/";
      else if (peek_sym("%")) op = "%";
      else break;
      ++p_;
      l = bin(op, l, parse_unary());
    }
    return l;
  }
  ExprP parse_unary() {
    if (peek_sym("-")) {
      ++p_;
      auto x = parse_unary();
      if (x->kind == Expr::Int) { x->ival = -x->ival; return x; }
      if (x->kind == Expr::Float) { x->fval = -x->fval; return x; }
      auto e = make_expr(Expr::Neg);
      e->args = {x};
      return e;
    }
    if (peek_sym("+")) { ++p_; return parse_unary(); }
    return parse_primary();
  }

  ExprP parse_primary() {
    const Token& t = cur();
    if (t.kind == Token::Num) {
      ++p_;
      if (t.text.find_first_of(".eE") != std::string::npos) {
        auto e = make_expr(Expr::Float);
        e->fval = std::stod(t.text);
        return e;
      }
      auto e = make_expr(Expr::Int);
      e->ival = std::stoll(t.text);
      return e;
    }
    if (t.kind == Token::Str) {
      ++p_;
      auto e = make_expr(Expr::Str);
      e->name = t.text;
      return e;
    }
    if (peek_sym("(")) {
      ++p_;
      if (kw("select")) fail("subqueries are not supported yet");
      auto e = parse_expr();
      expect_sym(")");
      return e;
    }
    if (t.kind != Token::Ident) fail("unexpected token in expression");
    // keyword-like primaries
    if (!t.quoted) {
      if (t.text == "date" && t_[p_ + 1].kind == Token::Str) {
        ++p_;
        auto e = make_expr(Expr::Date);
        int32_t d;
        if (!parse_date(cur().text, d)) fail("invalid date literal");
        e->ival = d;
        ++p_;
        return e;
      }
      if (t.text == "interval") {
        ++p_;
        auto e = make_expr(Expr::Interval);
        std::string unit;
        if (cur().kind == Token::Str) {
          std::string v = cur().text;
          ++p_;
          size_t k = 0;
          e->ival = std::stoll(v, &k);
          while (k < v.size() && v[k] == ' ') ++k;
          unit = v.substr(k);
        } else if (cur().kind == Token::Num) {
          e->ival = std::stoll(cur().text);
          ++p_;
        } else fail("expected interval value");
        if (unit.empty() && cur().kind == Token::Ident) unit = t_[p_++].text;
        unit = to_lower(unit);
        if (!unit.empty() && unit.back() == 's') unit.pop_back();
        if (unit != "day" && unit != "month" && unit != "year") fail("interval unit must be day, month or year");
        e->name = unit;
        return e;
      }
      if (t.text == "case") return parse_case();
      if (t.text == "extract") {
        ++p_;
        expect_sym("(");
        std::string field = ident();
        expect_kw("from");
        auto e = make_expr(Expr::Func);
        if (field != "year" && field != "month" && field != "day") fail("EXTRACT supports year, month, day");
        e->name = field;
        e->args = {parse_expr()};
        expect_sym(")");
        return e;
      }
      if (t.text == "null") fail("NULL is not supported");
    }
    std::string first = t.text;
    ++p_;
    if (peek_sym("(")) {  // function call
      ++p_;
      bool is_agg = first == "sum" || first == "count" || first == "avg" || first == "min" || first == "max";
      auto e = make_expr(is_agg ? Expr::Agg : Expr::Func);
      e->name = first;
      if (kw("distinct")) fail("DISTINCT aggregates are not supported yet");
      if (peek_sym("*")) {
        ++p_;
        if (first != "count") fail("* only allowed in count(*)");
        e->args.push_back(make_expr(Expr::Star));
      } else if (!peek_sym(")")) {
        do { e->args.push_back(parse_expr()); } while (sym(","));
      }
      expect_sym(")");
      if (!is_agg && first != "year" && first != "month" && first != "day" && first != "abs")
        throw Error("unknown function: " + first);
      if (is_agg && e->args.size() != 1) fail("aggregate takes exactly one argument");
      return e;
    }
    auto e = make_expr(Expr::Col);
    if (peek_sym(".")) {
      ++p_;
      e->table = first;
      e->name = ident();
    } else {
      e->name = first;
    }
    return e;
  }

  ExprP parse_case() {
    ++p_;  // case
    auto e = make_expr(Expr::Case);
    if (!kw("when")) fail("only searched CASE (CASE WHEN ...) is supported");
    while (kw("when")) {
      ++p_;
      e->args.push_back(parse_expr());
      expect_kw("then");
      e->args.push_back(parse_expr());
    }
    if (kw("else")) {
      ++p_;
      e->has_else = true;
      e->args.push_back(parse_expr());
    }
    expect_kw("end");
    return e;
  }
};

}  // namespace

Select parse_sql(const std::string& sql) {
  Parser p(sql);
  return p.parse();
}

}  // namespace warpdb
