#ifndef VOCAB_H_
#define VOCAB_H_

#include <vector>
#include <map>

class Vocab {
public:
  Vocab(const std::string & corpus_path, const unsigned size);
  ~Vocab() {};

  unsigned frequency(const unsigned id);
  std::string word(const unsigned id);
  unsigned id(const std::string & word);
  unsigned size() { return size_; }
  unsigned long num_words() { return num_words_; }
  std::vector<std::string> ConvertToWords(const std::vector<unsigned> ids);
  std::vector<unsigned> ConvertToIds(const std::vector<std::string> & words);
private:
  std::vector<unsigned> frequencies_;
  std::vector<std::string> num_to_words_;
  std::map<std::string, unsigned> word_to_nums_;
  unsigned size_;
  unsigned long num_words_;
};

#endif
