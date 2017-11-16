#ifndef MS_EMBEDDING_TRAINER_H_
#define MS_EMBEDDING_TRAINER_H_

#include <memory>
#include <vector>

#include <boost/property_tree/ptree.hpp>

#include <Eigen/Dense>

class Vocab;
class RawtextSplitter;

class MSEmbeddingTrainer {
public:
  // TODO: more constructors
  MSEmbeddingTrainer() {}
  MSEmbeddingTrainer(const Vocab & vocab, const boost::property_tree::ptree & config);
  ~MSEmbeddingTrainer() {}
  void Train(const Vocab & vocab, const boost::property_tree::ptree & config);
private:
  std::vector<int> SplitStringToIds(const Vocab & vocab,
                          const std::string & line, const float sampling);
  Eigen::VectorXf GetContextEmbedding(const std::vector<unsigned> & ids,
                    const unsigned i, const unsigned context_size,
                    const unsigned emb_size, bool & is_empty);
  int SampleSense(const int w_id, float gamma,
        const Eigen::VectorXf & context_emb, const unsigned max_sense_num);
  void UpdateParameters(const std::vector<unsigned> & ids, const unsigned idx,
         const unsigned context_size, Eigen::VectorXf & now_sense_emb,
         const float alpha, const unsigned neg_sample_count);
  float CalculateSigmoid(float x);
  std::vector<float> sigmoid_table_;
  std::vector<unsigned> unigram_table_;
  Eigen::MatrixXf global_embeddings_;
  std::map<unsigned, std::vector<Eigen::VectorXf>> sense_embeddings_;
  std::map<unsigned, std::vector<Eigen::VectorXf>> sense_contexts_;
  std::map<unsigned, std::vector<unsigned>> sense_counts_;
  // for fast mode
  unsigned exp_table_size_ = 1000;
  unsigned exp_table_max_ = 6;
  double unigram_power = 0.75;
  unsigned unigram_table_size_ = 1e8;
};

#endif
