#include "ms-embedding-trainer.h"

#include <sstream>
#include <numeric>
#include <random>
#include <boost/property_tree/ptree.hpp>

using namespace std;
using namespace Eigen;

namespace pt = boost::property_tree;

MSEmbeddingTrainer::MSEmbeddingTrainer(const pt::ptree & config) {
  // Calculate sigmoid value beforehand
  // sigmoid(x)
  //   = sigmoid_table_[int((x + kTableMmax) * kTableSize / (2 * kTableMmax))]
  if (config.get<unsigned>("Train.fast_mode")) {
    const unsigned kTableSize = 1000;
    const unsigned kTableMmax = 6;
    for (unsigned i = 0; i < kTableSize; ++i) {
      auto exp_x = exp((float(i) / kTableSize * 2 - 1) * kTableMmax);
      sigmoid_table_[i] = exp_x / (exp_x + 1);
    }
  }
  const auto vocab_size = config.get<unsigned>("Train.vocab_size");
  const auto embed_size = config.get<unsigned>("Train.embed_size");
  const auto scale = config.get<float>("Train.scale");
  global_embedding_ = MatrixXf::Random(vocab_size, embed_size).array() * scale;
}
