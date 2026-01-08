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

// DONE? tentatively
// MAYBE redo backprop / penalization / whatever it is
void backprop(Network &net, const std::vector<double> &e,
              std::vector<double> &input, double rate) {
  size_t h = net.layers.size();
  net.layers[h - 1].deltas = e;

  for (size_t i = h - 2; i > 0; i--) {
    Layer *layer = &net.layers[i];
    Layer *next = &net.layers[i + 1];
    for (size_t j = 0; j < layer->width; j++) {
      double sum = 0;
      for (size_t k = 0; k < next->width; k++) {
        sum += next->neurons[k].weights[j] * next->deltas[k];
      }
      double dtanh = 1.0 - layer->a[j] * layer->a[j];
      layer->deltas[j] = sum * dtanh;
    }
  }

  for (size_t i = 0; i < h; i++) {
    Layer *layer = &net.layers[i];
    // TODO: look into how these things build up
    // i always feel kind of bad putting an extra if statement into loops
    // if it isn't optimized out, id like to stop doing that, since that seems
    // like an extra burden for no reason, even if the effect is probably very
    // very small, it just annoys me man
    std::vector<double> *prev;
    if (i == 0)
      prev = &input;
    else
      prev = &net.layers[i - 1].a;

    for (size_t j = 0; j < layer->width; j++) {
      for (size_t k = 0; k < prev->size(); k++) {
        double gradient = layer->deltas[j] * (*prev)[k];
        layer->neurons[j].weights[k] -= rate * gradient;
      }
      layer->neurons[j].bias -= rate * layer->deltas[j];
    }
  }
}

// TODO: get rid of and find better approach
std::string batch_helper(size_t i) {
  return "./out/batches" + std::to_string(i) + "_batch.json";
}

std::string DE_PATH = "./data/de.txt";
std::string FR_PATH = "./data/fr.txt";

std::vector<std::string> LANG = {DE_PATH, FR_PATH};

size_t biN = 350, triN = 150;
NGramConfig ngramconf(biN, triN);

std::string FEATURIZED = "./out/batches";

double LEARNING_RATE = 0.05;
size_t SAMPLE_SIZE = 100000;

// TODO: #1 priority - reorganize & modularize this crazy logic
int main() {
  top_ngrams(LANG, ngramconf);
  std::cout << "DEBUG: started main\n";
  std::string NGRAMS_PATH = "out/ngrams.json";
  NGramConfig ngram_config = load_ngrams(NGRAMS_PATH);
  size_t n = ngram_config.features;

  featurize_dataset(LANG, ngram_config, 100000, FEATURIZED);

  NetworkConfig netconf{n, {{512}, {256}, {64}, {8}, {2}}};
  Network network{netconf};

  // TEMP -> manually set
  network.layers[0].xput = true;
  network.layers[2].xput = true;

  /* === FORWARD PASS
   *
   * */
  // TODO: stat gathering
  // int tot = 0, wrong = 0, borderc = 0, borderw = 0, strong = 0;
  int tot = 0, e = 0;
  // accuracy :
  // if correct value >> incorrect value

  // TODO: somewhat major, please create a better helper that sorts languages
  // dynamically instead of only knowing german or not german
  std::vector<std::tuple<bool, std::string, std::vector<double>>> all_samples;
  for (size_t i = 1; i <= 8; i++) {
    std::ifstream file(batch_helper(i));
    if (file.is_open()) {
      json dat = json::parse(file);
      for (const auto &[filename, words] : dat["data"].items()) {
        bool is_de = filename.find("de.txt") != std::string::npos;
        for (const auto &[word, feats] : words.items()) {
          all_samples.push_back({is_de, word, feats});
        }
      }
      dat.clear();
      file.close();
      std::cout << "DEBUG: closed batch " + std::to_string(i) + '\n';
    }
  }

  std::cout << "DEBUG: assembled pre-shuffle dataset!\n";

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(all_samples.begin(), all_samples.end(), g);

  size_t total_samples =
      std::min(static_cast<size_t>(SAMPLE_SIZE), all_samples.size());
  size_t train_size = static_cast<size_t>(total_samples * 0.8);
  size_t val_size = total_samples - train_size;

  std::vector<std::tuple<bool, std::string, std::vector<double>>> train_samples(
      all_samples.begin(), all_samples.begin() + train_size);
  std::vector<std::tuple<bool, std::string, std::vector<double>>> val_samples(
      all_samples.begin() + train_size, all_samples.begin() + total_samples);

  all_samples.clear(); // since this is a memory sucker

  std::cout << "Training on " << train_size << " samples, validating on "
            << val_size << " samples\n";

  int BATCH_SIZE = 250;
  int display = SAMPLE_SIZE / (BATCH_SIZE * 0.8);
  for (const auto &[is_de, word, feats] : train_samples) {

    double de_expect = is_de ? 1 : 0;
    double fr_expect = is_de ? 0 : 1;

    std::vector<double> expected = {de_expect, fr_expect};

    std::vector<double> input = feats;
    std::vector<double> out = forward(network, input);

    std::vector<double> error(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
      error[i] = out[i] - expected[i];
    }

    // CREDIT: nice logging was generated by claude since i am not good at
    // designing nice looking stuff lol
    int predicted = out[0] > out[1] ? 0 : 1; // 0=German, 1=French
    int actual = is_de ? 0 : 1;
    bool correct = predicted == actual;
    double confidence = std::max(out[0], out[1]);
    double loss = 0.5 * (error[0] * error[0] + error[1] * error[1]); // MSE

    tot++;
    if (tot == BATCH_SIZE) {
      std::cout << "=== ROUND " << e << " (" << std::to_string(e) << "/"
                << std::to_string(display) << ")" << std::endl;
      std::cout << (correct ? "y" : "n") << " " << (actual == 0 ? "DE" : "FR")
                << "→" << (predicted == 0 ? "DE" : "FR")
                << " conf:" << std::fixed << std::setprecision(3) << confidence
                << " loss:" << loss << " | \"" << word.substr(0, 30)
                << (word.length() > 30 ? "..." : "") << "\"" << std::endl;
      std::cout << "out[0]=" << out[0] << " out[1]=" << out[1]
                << " sum=" << (out[0] + out[1]) << std::endl;
      tot = 0;
      e++;
    }
    /* === BACKWARD PASS
     * go through layers BACKWARDS
     * (as long as not last layer):
     *  deltas = next.weights^T * next.deltas * (1 - curr.a^2)
     * */
    backprop(network, error, input, LEARNING_RATE);
  }

  int val_right = 0;
  double val_loss = 0.0;

  std::cout << "VALIDATION TIME!\n";

  for (const auto &[is_de, words, feats] : val_samples) {
    double de_expect = is_de ? 1 : 0;
    double fr_expect = is_de ? 0 : 1;
    std::vector<double> expected = {de_expect, fr_expect};

    std::vector<double> input = feats;
    std::vector<double> out = forward(network, input);

    std::vector<double> error(expected.size());

    for (size_t i = 0; i < expected.size(); i++) {
      error[i] = out[i] - expected[i];
    }

    int predicted = out[0] > out[1] ? 0 : 1;
    int actual = is_de ? 0 : 1;
    bool correct = predicted == actual;
    double loss = 0.5 * (error[0] * error[0] + error[1] * error[1]);
    if (correct)
      val_right++;
    val_loss += loss;
  }

  double accuracy = (double)val_right / val_size * 100.0;
  double avg_loss = val_loss / val_size;
  std::cout << "Validation Accuracy: " << std::fixed << std::setprecision(2)
            << accuracy << "% (" << val_right << "/" << val_size << ")\n";
  std::cout << "Average Validation Loss: " << avg_loss << "\n";
}
