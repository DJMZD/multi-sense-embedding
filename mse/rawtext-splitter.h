#ifndef RAWTEXT_SPLITTER_H
#define RAWTEXT_SPLITTER_H

#include <vector>
#include <string>

#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/string.hpp>

class RawtextSplitter {
public:
  RawtextSplitter() {}
  RawtextSplitter(const std::string & punctuation): punctuation_(punctuation) {}
  std::vector<std::string> Split(const std::string & text);
private:
  std::string punctuation_;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & punctuation_;
  }
};

#endif
