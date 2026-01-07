#include "network.hpp"
#include <fstream>
#include <iostream>
#include <vector>
// #include <iomanip>

#include "json.hpp"
using json = nlohmann::json;

/*
NetworkConfig netconf{1, {{2}, {5}, {3}}};
Network network{netconf};

int main() {
  std::cout << std::fixed << std::setprecision(4);
  network.debug();
}
*/

void softmax(Layer &last_layer) {
  double max = *std::max_element(last_layer.z.begin(), last_layer.z.end());
  double sum = 0;
  for (double &z : last_layer.z) {
    sum += exp(z - max);
  }
  for (size_t i = 0; i < last_layer.width; i++) {
    last_layer.a[i] = exp(last_layer.z[i] - max) / sum;
  }
}

std::vector<double> forward(Network &net, const std::vector<double> &input) {
  // z[j] = bias + dot(weights, prev_input)
  // if hidden a[j] = tanh(z[j])
  // else dont touch a
  // after finished, run softmox and fill out layer a[k] with probs
  size_t height = net.layers.size();

  for (size_t i = 0; i < height; i++) {
    Layer *layer = &net.layers[i];
    const std::vector<double> *prev_activations =
        (i == 0) ? &input : &net.layers[i - 1].a;

    // TODO: rm
    std::cout << "DEBUG: Layer " << i << " width=" << layer->width
              << " neurons=" << layer->neurons.size()
              << " prev_size=" << prev_activations->size() << std::endl;

    for (size_t j = 0; j < layer->width; j++) {

      layer->z[j] = layer->neurons[j].bias;
      for (size_t k = 0; k < prev_activations->size(); k++) {
        layer->z[j] += layer->neurons[j].weights[k] * (*prev_activations)[k];
      }
    }
    if (!layer->xput) {
      for (size_t j = 0; j < layer->width; j++) {
        layer->a[j] = tanh(layer->z[j]);
      }
    }
  }
  Layer *last_layer = &net.layers[height - 1];
  // softmax
  softmax(*last_layer);
  return last_layer->a;
}

// TODO: backprop / penalization / whatever it is

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 175, triN = 125;
NGramConfig ngramconf(biN, triN);

std::string FEATURIZED = "./out/batches";

int main() {
  // top_ngrams(LANG, ngramconf);
  std::cout << "DEBUG: started main";
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  size_t n = ngram_config.features;

  // featurize_dataset(LANG, ngram_config, 100000, FEATURIZED);

  NetworkConfig netconf{n, {{350}, {175}, {2}}};
  Network network{netconf};

  // TEMP -> manually set
  network.layers[0].xput = true;
  network.layers[2].xput = true;

  /* === FORWARD PASS
   *
   * */
  int tot = 0, wrong = 0, borderc = 0, borderw = 0, strong = 0;

  // accuracy :
  // if correct value >> incorrect value

  std::ifstream batch1("./out/batches1_batch.json");
  std::ifstream batch8("./out/batches8_batch.json");
  json batch1dat = json::parse(batch1);
  json batch8dat = json::parse(batch8);
  batch1dat.update(batch8dat, true);

  for (const auto &[filename, words] : batch1dat["data"].items()) {
    bool is_de = filename.find("de.txt") != std::string::npos;

    for (const auto &[word, feats] : words.items()) {
      std::vector<double> input = feats;
      std::vector<double> out = forward(network, input);

      int expected = is_de ? 0 : 1;

      std::cout << "Text: \"" << word << "\"\n";
      std::cout << "Expected: " << (expected ? "FR" : "DE")
                << ", Prediction: " << (expected ? "FR" : "DE") << '\n';
      std::cout << "Confidence: DE=" << out[0] << ", FR=" << out[1] << '\n';
    }
  }
}
