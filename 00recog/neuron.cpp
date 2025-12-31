#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
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
                     std::mt19937 &rng) {
  weights.resize(num_inputs);
  double s = 1.0 / std::sqrt((double)num_inputs);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (auto &w : weights)
    w = dist(rng) * s;
}

struct Neuron {
  // should be width of prev layer
  std::vector<double> weights;
  // boundary : w*x + b
  double bias;

  Neuron(size_t num_inputs, std::mt19937 &rng, double b) : bias{b} {
    populateWeights(weights, num_inputs, rng);
  };

  void debug_weights() const {
    for (auto &w : weights) {
      std::cout << std::setw(8) << w;
    }
  }
};

struct Layer {
  std::vector<Neuron> neurons;
  size_t width;

  // z = \sum_i (w_i x_i) + b (affine)
  std::vector<double> a, z, deltas;

  Layer(size_t w, size_t prev, std::mt19937 &rng) : width{w} {
    for (size_t i = 0; i < w; i++) {
      neurons.push_back(Neuron(prev, rng, 0.00));
    }

    // aside: look into changing inital values of a,z,deltas;
    // for now just keeping them at 0, dont think theyre important
    // since they get redone every time anyway
    a.resize(w, 0);
    z.resize(w, 0);
    deltas.resize(w, 0);
  }

  void debug() const {
    for (auto &n : neurons) {
      std::cout << std::setw(3) << "{";
      n.debug_weights();
      std::cout << std::setw(3) << "}\n";
    }
  }
};

struct LayerConfig {
  size_t width;
};

struct NetworkConfig {
  std::vector<LayerConfig> layers;
  size_t input_size;
  size_t layer_count;

  NetworkConfig(size_t in_size, std::vector<LayerConfig> l)
      : layers{l}, input_size{in_size} {
    layer_count = layers.size();
  }
};

/* NETWORK INITIALIZER FORMAT
 * Network (
 *   {
 *      { layer_width }
 *      TODO: finish transferring neuron stuff to layer
 *      then we can populate this
 *   }
 * )
 * */
struct Network {
  std::vector<Layer> layers;
  std::mt19937 rng{std::random_device{}()};

  Network(NetworkConfig net) {
    if (net.layer_count < 1) {
      throw std::runtime_error("netconfig has <1 layer");
    }
    if (net.input_size < 1) {
      throw std::runtime_error("net input size <1");
    }
    layers.push_back(Layer(net.layers[0].width, net.input_size, rng));
    for (size_t i = 1; i < net.layer_count; i++) {
      layers.push_back(
          Layer(net.layers[i].width, net.layers[i - 1].width, rng));
    }
  }

  void debug() const {
    for (size_t i = 0; i < layers.size(); i++) {
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

NetworkConfig netconf{1, {{2}, {5}, {3}}};
Network network{netconf};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
