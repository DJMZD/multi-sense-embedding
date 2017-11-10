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
private:
  std::vector<float> sigmoid_table_;
  Eigen::MatrixXf global_embedding_;
  std::map<unsigned, std::vector<Eigen::VectorXf>> sense_embeddings_;
  std::map<unsigned, std::vector<unsigned>> sense_counts_;
};

#endif
