#include "ms-embedding-trainer.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>

#include "vocab.h"
#include "rawtext-splitter.h"

using namespace std;
using namespace Eigen;

namespace ari = boost::algorithm;
namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

MSEmbeddingTrainer::MSEmbeddingTrainer(const pt::ptree & config) {
  // Calculate sigmoid value beforehand
  // sigmoid(x)
  //   = sigmoid_table_[int((x + kTableMmax) * kTableSize / (2 * kTableMmax))]
  if (config.get<unsigned>("Train.fast_mode")) {
    table_size_ = 1000;
    table_max_ = 6;
    for (unsigned i = 0; i < table_size_; ++i) {
      auto exp_x = exp((float(i) / table_size_ * 2 - 1) * table_max_);
      sigmoid_table_.push_back(exp_x / (exp_x + 1));
    }
  }
  const auto vocab_size = config.get<unsigned>("Train.vocab_size");
  const auto emb_size = config.get<unsigned>("Train.emb_size");
  const auto scale = config.get<float>("Train.scale");
  global_embeddings_ = MatrixXf::Random(vocab_size, emb_size) * scale;
}

void MSEmbeddingTrainer::Train(const Vocab & vocab, const pt::ptree & config) {
  const auto train_path = config.get<string>("Corpus.train_path");
  const auto vocab_size = config.get<unsigned>("Train.vocab_size");
  const auto emb_size = config.get<unsigned>("Train.emb_size");
  const auto context_size = config.get<unsigned>("Train.context_size");
  const auto max_sense_num = config.get<unsigned>("Train.max_sense_num");
  const auto max_iter_num = config.get<unsigned>("Train.max_iter");
  const auto sampling = config.get<float>("Train.sampling");
  const auto alpha = config.get<float>("Train.alpha");
  const auto gamma = config.get<float>("Train.gamma");
  const auto is_fase_mode = config.get<bool>("Train.fast_mode");
  for (unsigned i = 0; i < max_iter_num; ++i) {
    cerr << "Iter " << i << endl;
    const fs::path path(train_path);
    using recur_it = fs::recursive_directory_iterator;
    for (const auto & p: boost::make_iterator_range(recur_it(path), {})) {
      if (fs::is_directory(p)) {
        cout << p << endl;
      } else {
        vector<int> word_senses;
        ifstream ifs(p.path().string());
        string line;
        while (getline(ifs, line)) {
          auto ids = SplitStringToIds(vocab, line, sampling);
          if (ids.empty()) { continue; }
          for (unsigned i = 0; i < ids.size(); ++i) {
            auto now_id = ids[i];
            auto context_emb = GetContextEmbedding(ids, i, context_size,
                                 emb_size);
            if (!context_emb.size()) {
              word_senses.push_back(-1);
              continue;
            }
            auto & now_embs = sense_embeddings_[now_id];
            auto & now_cns = sense_counts_[now_id];
            auto sense = SampleSense(now_id, gamma, context_emb, max_sense_num);
            if (sense >= now_embs.size()) {
              now_embs.push_back(context_emb);
              now_cns.push_back(1);
              word_senses.push_back(sense);
            } else {
              now_embs[sense] += context_emb * alpha;
              ++now_cns[sense];
              word_senses.push_back(sense);
            }
          }
        }
      }
    }
  }
}

vector<int> MSEmbeddingTrainer::SplitStringToIds(const Vocab & vocab,
                                       const string & line,
                                       const float sampling) {
  vector<int> ids;
  // for empty line, header and footer
  if (line.empty() || line.find("<doc") == 0 || line.find("</doc") == 0) {
    return ids;
  }
  auto tokens = vocab.text_splitter()->Split(line);
  for (const auto & t: tokens) {
    auto w_id = vocab.id(ari::to_lower_copy(t));
    auto w_count = vocab.frequency(w_id);
    // subsampling
    if (sampling) {
      auto prob = sqrt(sampling * w_count / vocab.num_words())
                  + sampling * w_count / vocab.num_words();
      random_device seed_gen;
      mt19937 engine(seed_gen());
      uniform_real_distribution<float> dist(0, 1.0);
      auto ran = dist(engine);
      if (ran <= prob) {
        ids.push_back(w_id);
      } else {
        ids.push_back(-1);
      }
    }
  }
  return ids;
}

VectorXf MSEmbeddingTrainer::GetContextEmbedding(const vector<int> & ids,
                               const unsigned i, const unsigned context_size,
                               const unsigned emb_size) {
  VectorXf context_emb = VectorXf::Zero(emb_size);
  unsigned l_pos = 0;
  unsigned r_pos = ids.size();
  if (i > context_size / 2) { l_pos = i - context_size / 2; }
  if (i + context_size / 2 < ids.size()) { r_pos = i + context_size / 2; }
  for (unsigned pos = l_pos; pos <= r_pos; ++pos) {
    context_emb += global_embeddings_.row(pos);
  }
  context_emb = context_emb / (r_pos - l_pos + 1);

  return context_emb;
}

int MSEmbeddingTrainer::SampleSense(const int w_id, float gamma,
                               const VectorXf & context_emb,
                               const unsigned max_sense_num) {
  auto & now_embs = sense_embeddings_[w_id];
  auto & now_cns = sense_counts_[w_id];
  if (now_embs.empty()) { return 0; }

  vector<float> probablities(now_embs.size());
  for (unsigned i = 0; i < probablities.size(); ++i) {
    const auto & now_emb = now_embs[i];
    const auto & now_cn = now_cns[i];
    float sim = context_emb.dot(now_emb);
    // TODO: normal mode
    auto table_index = int((sim + table_max_) * table_size_ / (2 * table_max_));
    probablities[i] = now_cn * sigmoid_table_[table_index];
    if (i) { probablities[i] += probablities[i - i]; }
  }
  if (now_cns.size() < max_sense_num) {
    auto new_sense_prob = probablities.back() + gamma;
    probablities.push_back(new_sense_prob);
  }

  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_real_distribution<float> dist(0, probablities.back());
  auto ran = dist(engine);

  int sense = 0;
  while (sense < probablities.size()) {
    if (ran <= probablities[sense++]) { break; }
  }
  return sense - 1;
}
