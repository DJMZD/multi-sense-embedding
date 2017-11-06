#include "vocab.h"

#include <iostream>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace ari = boost::algorithm;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;
using Tokenizer = boost::tokenizer<boost::char_separator<char>>;

Vocab::Vocab(const string & corpus_path, const unsigned size) {
  map<string, unsigned> frequency;
  unsigned num_lines = 0;
  unsigned num_words = 0;

  const fs::path path(corpus_path);
  using recur_it = fs::recursive_directory_iterator;
  for (const auto & pa: boost::make_iterator_range(recur_it(path), {})) {
    if (fs::is_directory(pa)) {
      cout << pa << endl;
    } else {
      ifstream ifs(pa.path().string());
      string line;
      getline(ifs, line);  // for header
      while (getline(ifs, line)) {
        if (line.empty()) { continue; }
        ari::replace_all(line, "‘", "'");
        ari::replace_all(line, "’", "'");
        ari::replace_all(line, "“", "\"");
        ari::replace_all(line, "”", "\"");
        ari::replace_all(line, "'s", " xqyzs");
        string punctuation = "/⁄\\()\"':,.;<>~!@#$%^&*|+=[]{}`?-…-‹›«»‐-‘’“”";
        boost::char_separator<char> sep(" ", punctuation.c_str());
        ari::replace_all(line, "xqyzs", "'s");
        Tokenizer tok(line, sep);
        for (auto & word: tok) {
          ++frequency[ari::to_lower_copy(word)];
        }
        ++num_lines;
        num_words += distance(tok.begin(), tok.end());
      }
    }
  }
  num_words_ = num_words;

  vector<pair<unsigned, string>> frequencies;
  for (const auto & e: frequency) {
    frequencies.emplace_back(e.second, e.first);
  }

  sort(frequencies.begin(), frequencies.end(), [](const auto & lhs, const auto & rhs) {
    if (lhs.first > rhs.first) { return true; }
    else if (lhs.first < rhs.first) { return false; }
    else { return lhs.second < rhs.second; }
  });

  frequencies_.emplace_back(num_words);
  frequencies_.emplace_back(num_lines);
  frequencies_.emplace_back(num_lines);
  num_to_words_.emplace_back("<UNK>");
  num_to_words_.emplace_back("<s>");
  num_to_words_.emplace_back("</s>");
  word_to_nums_["<UNK>"] = 0;
  word_to_nums_["<s>"] = 1;
  word_to_nums_["</s>"] = 2;
  unsigned num_kind_of_words = 3;
  for (auto & p: frequencies) {
    frequencies_.emplace_back(p.first);
    frequencies_[0] -= p.first;
    num_to_words_.emplace_back(p.second);
    word_to_nums_[p.second] = num_kind_of_words++;
    if (frequencies_.size() >= size + 3) {
      break;
    }
  }
  size_ = frequencies_.size();
}

unsigned Vocab::frequency(const unsigned id) {
  if (id > frequencies_.size()) {
    ostringstream oss;
    oss << __FILE__ << ": " << __LINE__ << "ERROR: "
        << "Id{" << id << "} is out of range";
    throw range_error(oss.str());
  } else {
    return frequencies_[id];
  }
}

string Vocab::word(const unsigned id) {
  if (id > num_to_words_.size()) {
    ostringstream oss;
    oss << __FILE__ << ": " << __LINE__ << "ERROR: "
        << "Id{" << id << "} is out of range";
    throw range_error(oss.str());
  } else {
    return num_to_words_[id];
  }
}

unsigned Vocab::id(const string & word) {
  auto it = word_to_nums_.find(word);
  if (it == word_to_nums_.end()) {
    return 0;  // <UNK>
  } else {
    return it->second;
  }
}

vector<string> Vocab::ConvertToWords(const vector<unsigned> ids) {
  vector<string> words;
  for (auto & id: ids) {
    words.emplace_back(word(id));
  }
  return words;
}

vector<unsigned> Vocab::ConvertToIds(const vector<string> & words) {
  vector<unsigned> ids;
  for (auto & word: words) {
    ids.emplace_back(id(word));
  }
  return ids;
}
