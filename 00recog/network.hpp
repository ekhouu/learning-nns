#pragma once

#include <random>
#include <vector>

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
                     std::mt19937 &rng);

struct Neuron {
  // should be width of prev layer
  std::vector<double> weights;
  // boundary : w*x + b
  double bias;

  Neuron(size_t num_inputs, std::mt19937 &rng, double b);

  void debug_weights() const;
};

struct Layer {
  std::vector<Neuron> neurons;
  size_t width;

  // z = \sum_i (w_i x_i) + b (affine)
  std::vector<double> a, z, deltas;

  Layer(size_t w, size_t prev, std::mt19937 &rng);

  void debug() const;
};

struct LayerConfig {
  size_t width;
};

struct NetworkConfig {
  std::vector<LayerConfig> layers;
  size_t input_size;
  size_t layer_count;

  NetworkConfig(size_t in_size, std::vector<LayerConfig> l);
};

/* NETWORK INITIALIZER FORMAT
 * Network (
 *   {
 *      { layer_width }
 * TODO: change to
 * 	{ size_t layer_width, vector<vector<double>> weights, etc}
 *   }
 * )
 * */
struct Network {
  std::vector<Layer> layers;
  std::mt19937 rng{std::random_device{}()};

  Network(NetworkConfig net);

  void debug() const;
};
