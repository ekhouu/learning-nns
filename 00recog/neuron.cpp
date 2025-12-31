#include <cmath>
#include <iomanip>
#include <iostream>
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

void populateWeights(std::vector<float> &weights, size_t num_inputs,
                     std::mt19937 &rng) {
  weights.resize(num_inputs);
  float s = 1.f / std::sqrt((float)num_inputs);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (auto &w : weights)
    w = dist(rng) * s;
}

struct Neuron {
  // should be width of prev layer
  std::vector<float> weights;
  // boundary : w*x + b
  double bias;
  // temp --> move to layer
  double a, z, delta;

  Neuron(size_t num_inputs, std::mt19937 &rng,
         double b, /* TODO: remove a,z,delta and move to layer*/
         double a, double z, double delta)
      : bias{b} {
    populateWeights(weights, num_inputs, rng);
  };

  const void debug_weights() {
    for (auto &w : weights) {
      std::cout << std::setw(8) << w;
    }
  }
};

struct Layer {
  std::vector<Neuron> neurons;
  size_t width;

  // z = \sum_i (w_i x_i) + b (affine)
  // TODO: move from neuron to layer
  // std::vector<double> a, z, deltas;

  Layer(size_t w, size_t prev, std::mt19937 &rng) : width{w} {
    for (int i = 0; i < w; i++) {
      neurons.push_back(Neuron(prev, rng, 0.00, 0.00, 0.00, 0.00));
    }
  }

  const void debug() {
    for (auto &n : neurons) {
      std::cout << std::setw(3) << "{";
      n.debug_weights();
      std::cout << std::setw(3) << "}\n";
    }
  }
};

struct Network {
  std::vector<Layer> layers;

  const void debug() {
    for (int i = 0; i < layers.size(); i++) {
      std::cout << "Layer " << i << ":\n";
      layers[i].debug();
    }
  }
};

// BASIC NEURON FUCKERY

/* testing network
 *
 * note: every neuron on one layer is connected to every neuron of the next
 * layer, we store network as an array
 *
 * */

std::mt19937 rng(std::random_device{}());

Layer A(5, 2, rng);
Layer B(2, 5, rng);

Network network{{A, B}};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
