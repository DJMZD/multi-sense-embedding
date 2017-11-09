#ifndef RAWTEXT_SPLITTER_H
#define RAWTEXT_SPLITTER_H

#include <vector>
#include <string>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

class RawtextSplitter {
public:
  RawtextSplitter(const std::string & punctuation): punctuation_(punctuation) {}
  std::vector<std::string> Split(const std::string & text);
private:
  std::string punctuation_;
};

#endif
