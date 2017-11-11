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
  MSEmbeddingTrainer(const boost::property_tree::ptree & config);
  ~MSEmbeddingTrainer() {};
  void Train(const Vocab & vocab, const boost::property_tree::ptree & config);
private:
  std::vector<unsigned> SplitStringToIds(const Vocab & vocab,
                          const std::string & line, const float sampling);
  Eigen::VectorXf GetContextEmbedding(const std::vector<unsigned> & ids,
                    const unsigned i, const unsigned context_size,
                    const unsigned emb_size);
  unsigned SampleSense(const unsigned w_id, float gamma,
             const Eigen::VectorXf & context_emb, const unsigned max_sense_num);
  std::vector<float> sigmoid_table_;
  Eigen::MatrixXf global_embeddings_;
  std::map<unsigned, std::vector<Eigen::VectorXf>> sense_embeddings_;
  std::map<unsigned, std::vector<unsigned>> sense_counts_;
  // for fast mode
  unsigned table_size_ = 1000;
  unsigned table_max_ = 6;
};

#endif
