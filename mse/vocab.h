#ifndef VOCAB_H_
#define VOCAB_H_

#include <vector>
#include <unordered_map>

class Vocab {
public:
  Vocab(const std::string & corpus_path, const unsigned size);
  ~Vocab();

  unsigned Vocab::frequency(const unsigned id);
  string Vocab::word(const unsigned id);
  unsigned Vocab::id(const string & word);
  vector<string> Vocab::ConvertToWords(const vector<unsigned> ids);
  vector<unsigned> Vocab::ConvertToIds(const string & words);
private:
  std::vector<unsigned> frequencies_;
  std::vector<std::string> num_to_words_;
  std::unordered_map<std::string, unsigned> word_to_nums_;
};

#endif
