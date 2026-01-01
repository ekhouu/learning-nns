#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

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

/* testing network
 *
 * note: every neuron on one layer is connected to every neuron of the next
 * layer, we store network as an array
 *
 * */

NetworkConfig netconf{1, {{2}, {5}, {3}}};
Network network{netconf};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
