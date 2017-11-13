#ifndef VOCAB_H_
#define VOCAB_H_

#include <memory>
#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

class RawtextSplitter;

// \brief Make a vocab instance from a corpus of rawtext
// \details Word(s) can be converted to id(s) according the rank of frequency
// and vice versa.
class Vocab {
public:
  // \brief Make a vocab instance from a corpus of rawtext
  //
  // \param corpus_path The root path of the corpus
  // \param size The maximum size of words except <UNK>, <s> and </s>
  Vocab() {};
  Vocab(const std::string & corpus_path, const unsigned size,
    boost::shared_ptr<RawtextSplitter> text_splitter);
  ~Vocab() {};

  unsigned frequency(const unsigned id) const;
  std::string word(const unsigned id) const;
  unsigned id(const std::string & word) const;
  unsigned size() const { return size_; }
  unsigned long num_words() const { return num_words_; }
  boost::shared_ptr<RawtextSplitter> text_splitter() const { return text_splitter_; }
  std::vector<std::string> ConvertToWords(const std::vector<unsigned> ids) const;
  std::vector<unsigned> ConvertToIds(const std::vector<std::string> & words) const;
private:
  std::vector<unsigned> frequencies_;
  std::vector<std::string> num_to_words_;
  std::map<std::string, unsigned> word_to_nums_;
  unsigned size_;
  unsigned long num_words_;
  boost::shared_ptr<RawtextSplitter> text_splitter_;

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned) {
    ar & frequencies_;
    ar & num_to_words_;
    ar & word_to_nums_;
    ar & size_;
    ar & num_words_;
    ar & text_splitter_;
  }
};

#endif
