#include "rawtext-splitter.h"

using namespace std;

namespace ari = boost::algorithm;
using Tokenizer = boost::tokenizer<boost::char_separator<char>>;

vector<string> RawtextSplitter::Split(const string & text) {
  vector<string> splitted_tokens;
  string text_copy(text);
  ari::replace_all(text_copy, "‘", "'");
  ari::replace_all(text_copy, "’", "'");
  ari::replace_all(text_copy, "“", "\"");
  ari::replace_all(text_copy, "”", "\"");
  ari::replace_all(text_copy, "'s", " xqyzs");
  boost::char_separator<char> sep(" ", punctuation_.c_str());
  ari::replace_all(text_copy, "xqyzs", "'s");
  Tokenizer tok(text_copy, sep);
  for (const auto & e: tok) {
    splitted_tokens.push_back(e);
  }

  return splitted_tokens;
}
