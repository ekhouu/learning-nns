#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

#include "network.hpp"

/* my goal here is generally to just write a functional nn but w/ nothing fancy,
 * i want to learn the basics before moving into the more abstracted py stuff
 */

/* -- NEURON --
 *
 * x_1 ----- |____________|
 *           |            |
 * x_2 ----- |            |
 * 	     |            |
 * x_3 ----- |            |
 *           |            |
 * x_4 ----- |____________|
 *
 * threshhold : \sum x_i w_i > T
 * */

// LIB

void populateWeights(std::vector<double> &weights, size_t num_inputs,
                     std::mt19937 &rng) {
  weights.resize(num_inputs);
  double s = 1.0 / std::sqrt((double)num_inputs);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (auto &w : weights)
    w = dist(rng) * s;
}

Neuron::Neuron(size_t num_inputs, std::mt19937 &rng, double b) : bias{b} {
  populateWeights(weights, num_inputs, rng);
}

void Neuron::debug_weights() const {
  for (auto &w : weights) {
    std::cout << std::setw(8) << w;
  }
}

// TODO: make implementation more modular (i.e. make it easier to load saved NNs
// instead of regen & override)
Layer::Layer(size_t w, size_t prev, std::mt19937 &rng) {
  for (size_t i = 0; i < w; i++) {
    neurons.push_back(Neuron(prev, rng, 0.00));
  }

  a.resize(w, 0);
  z.resize(w, 0);
  deltas.resize(w, 0);
}

void Layer::debug() const {
  for (auto &n : neurons) {
    std::cout << std::setw(3) << "{";
    n.debug_weights();
    std::cout << std::setw(3) << "}\n";
  }
}

/* NETWORK INITIALIZER FORMAT
 * Network (
 *   {
 *      { layer_width }
 *      TODO: finish transferring neuron stuff to layer
 *      then we can populate this
 *   }
 * )
 * */
NetworkConfig::NetworkConfig(size_t in_size, std::vector<LayerConfig> l)
    : layers{l}, input_size{in_size} {
  layer_count = layers.size();
}

Network::Network(NetworkConfig net) {
  if (net.layer_count < 1) {
    throw std::runtime_error("NetworkConfig has <1 layer");
  }
  if (net.input_size < 1) {
    throw std::runtime_error("Net input size <1");
  }

  layers.push_back(Layer(net.layers[0].width, net.input_size, rng));
  for (size_t i = 1; i < net.layer_count; i++) {
    layers.push_back(Layer(net.layers[i].width, net.layers[i - 1].width, rng));
  }
}

void Network::debug() const {
  for (size_t i = 0; i < layers.size(); i++) {
    std::cout << "Layer " << i << ":\n";
    layers[i].debug();
  }
}

std::string standardize(const std::string &text) {
  std::string result = text;

  result.erase(std::remove_if(result.begin(), result.end(),
                              [](char c) { return (unsigned char)c > 127; }),
               result.end());

  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  return result;
}

// NGRAMS
std::map<std::string, int> extract_ngrams(const std::string &filename, int n) {
  std::ifstream file(filename);
  std::map<std::string, int> ngrams;
  std::string line;

  // TODO: count all words in file & add parameter to for unique iden weighing
  while (std::getline(file, line)) {
    // DONE: normalize text
    line = standardize(line);
    for (size_t i = 0; i + n <= line.length(); i++) {
      std::string ngram = line.substr(i, n);
      ngrams[ngram]++;
    }
  }
  return ngrams;
}

std::vector<std::pair<std::string, int>>
get_top_k(const std::map<std::string, int> &ngrams, int k) {
  std::vector<std::pair<std::string, int>> sorted(ngrams.begin(), ngrams.end());
  std::partial_sort(
      sorted.begin(), sorted.begin() + k, sorted.end(),
      [](const auto &a, const auto &b) { return a.second > b.second; });
  sorted.resize(k);
  return sorted;
}

std::map<std::string, int>
combine_ngrams(const std::vector<std::map<std::string, int>> &ngram_maps) {
  std::map<std::string, int> out;
  for (auto &map : ngram_maps) {
    for (const auto &[key, value] : map) {
      out[key] += value;
    }
  }

  return out;
}

int top_ngrams(std::vector<std::string> &paths, NGramConfig nconf) {
  std::vector<std::map<std::string, int>> bi, tri;
  for (auto &p : paths) {
    bi.push_back(extract_ngrams(p, 2));
    tri.push_back(extract_ngrams(p, 3));
  }

  std::map<std::string, int> bigrams = combine_ngrams(bi);
  std::map<std::string, int> trigrams = combine_ngrams(tri);

  json ngrams_data;
  ngrams_data["bigrams"] = get_top_k(bigrams, nconf.bk);
  ngrams_data["trigrams"] = get_top_k(trigrams, nconf.tk);
  std::ofstream("out/ngrams.json") << ngrams_data.dump(2);

  return 1;
}

NGramConfig::NGramConfig(size_t b, size_t t) : bk{b}, tk{t} {
  features = b + t;
}
